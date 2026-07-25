"""Unified dataset layer (dendroprop.data).

No network access / no dataset downloads: every source is exercised against a
tiny synthetic fixture (a fake SHD `.h5` written with h5py, a fake NMNIST `.bin`).
The preprocessor is anchored to a vendored copy of the reference count-binning.
"""
from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from dendroprop.data import (
    Augment,
    CachedSpikeDataset,
    SpikeDataset,
    bin_events,
    build_cache,
    cache_key,
    channel_shift,
    consecutive_pool_map,
    decode_nmnist_events,
    default_params,
    iterate_batches,
    load_cached,
    nmnist_pool_map,
    nmnist_pool_n_channels,
    spike_dropout,
)

_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Fixtures: synthetic dataset files                                           #
# --------------------------------------------------------------------------- #
def _write_fake_shd_h5(path: Path, samples, labels):
    """Write a minimal SHD-format HDF5 (spikes/units, spikes/times, labels)."""
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        spikes = f.create_group("spikes")
        du = spikes.create_dataset(
            "units", (len(samples),), dtype=h5py.vlen_dtype(np.dtype("int64"))
        )
        dt = spikes.create_dataset(
            "times", (len(samples),), dtype=h5py.vlen_dtype(np.dtype("float64"))
        )
        for i, (u, t) in enumerate(samples):
            du[i] = np.asarray(u, dtype=np.int64)
            dt[i] = np.asarray(t, dtype=np.float64)
        f.create_dataset("labels", data=np.asarray(labels, dtype=np.int64))


def _fake_shd_data_path(tmp_path: Path):
    """A data_path whose hdspikes/shd_train.h5 holds two known SHD samples."""
    rng = np.random.default_rng(0)
    # sample 0 (label 3): a few spikes; sample 1 (label 7): a few spikes.
    s0 = (rng.integers(0, 700, size=40), rng.uniform(0.0, 1.399, size=40))
    s1 = (rng.integers(0, 700, size=25), rng.uniform(0.0, 1.399, size=25))
    _write_fake_shd_h5(
        tmp_path / "hdspikes" / "shd_train.h5", [s0, s1], labels=[3, 7]
    )
    return tmp_path


def _nmnist_event_bytes(x, y, pol, time):
    """One NMNIST AER event as 5 bytes (time < 2**16 kept in the low two cols)."""
    return bytes([x & 0xFF, y & 0xFF, (pol << 7) & 0xFF, (time >> 8) & 0xFF, time & 0xFF])


# --------------------------------------------------------------------------- #
# 1. bin_events basics                                                        #
# --------------------------------------------------------------------------- #
def test_bin_events_basic_counts_and_padding():
    # channels 0,1 map to out-channels 0,0 (collapse 2); channel 2 -> out 1.
    cmap = consecutive_pool_map(4, 2)  # [0,0,1,1]
    units = np.array([0, 1, 0])       # two hit out-ch 0 at bin 0, plus another
    times_ms = np.array([1.0, 1.0, 1.0])
    x = bin_events(
        units, times_ms, T_bins=3, bin_size_ms=4.0, channel_map=cmap,
        n_channels_out=2, max_duration_ms=12.0,
    )
    assert x.shape == (3, 2)
    assert x.dtype == np.uint8
    assert x[0, 0] == 3           # three events accumulate in (bin0, out-ch0)
    assert x[1:].sum() == 0        # zero-padded tail


def test_bin_events_drops_out_of_window_and_binarizes():
    cmap = consecutive_pool_map(2, 1)
    units = np.array([0, 0, 1])
    times_ms = np.array([1.0, 1.0, 999.0])  # third event past the window
    x = bin_events(
        units, times_ms, T_bins=2, bin_size_ms=4.0, channel_map=cmap,
        n_channels_out=2, max_duration_ms=8.0, binarize=True,
    )
    assert x[0, 0] == 1            # two events, binarized to 1
    assert x.sum() == 1           # the out-of-window event was dropped


# --------------------------------------------------------------------------- #
# 2. Parity (anchor the preprocessor to the reference count-binning)           #
# --------------------------------------------------------------------------- #
def _legacy_bin_shd_reference(units, times_sec, bin_size_ms, collapse_factor,
                              n_units_in, max_duration_ms, dtype=np.float32):
    """Reference count-binning (non-binarize path) that ``preprocess.bin_events`` must
    reproduce exactly (float cast, counts <= 255). The ``binarize`` branch is omitted —
    this anchor only exercises the count path."""
    units = np.asarray(units, dtype=np.int64)
    times_ms = np.asarray(times_sec, dtype=np.float64) * 1000.0
    if max_duration_ms is not None:
        keep = (times_ms >= 0.0) & (times_ms < max_duration_ms)
        times_ms, units = times_ms[keep], units[keep]
        T_bins = max(1, int(np.ceil(max_duration_ms / bin_size_ms)))
    else:
        keep = times_ms >= 0.0
        times_ms, units = times_ms[keep], units[keep]
        T_bins = int(np.floor(times_ms.max() / bin_size_ms)) + 1 if times_ms.size else 1
    n_channels_out = (n_units_in + collapse_factor - 1) // collapse_factor
    t_idx = (times_ms / bin_size_ms).astype(np.int64)
    np.minimum(t_idx, T_bins - 1, out=t_idx)
    c_idx = units // collapse_factor
    valid = (c_idx >= 0) & (c_idx < n_channels_out) & (t_idx >= 0)
    t_idx, c_idx = t_idx[valid], c_idx[valid]
    x = np.zeros((T_bins, n_channels_out), dtype=dtype)
    np.add.at(x, (t_idx, c_idx), 1)
    return x


def test_bin_events_parity_with_legacy_shd():
    bin_shd_sample = _legacy_bin_shd_reference

    rng = np.random.default_rng(42)
    units = rng.integers(0, 700, size=500)
    times_sec = rng.uniform(0.0, 1.399, size=500)
    cmap = consecutive_pool_map(700, 5)

    mine = bin_events(
        units, times_sec * 1000.0, T_bins=350, bin_size_ms=4.0, channel_map=cmap,
        n_channels_out=140, max_duration_ms=1400.0,
    ).astype(np.float32)
    legacy = bin_shd_sample(
        units, times_sec, bin_size_ms=4.0, collapse_factor=5,
        n_units_in=700, max_duration_ms=1400.0, dtype=np.float32,
    )
    assert mine.shape == legacy.shape == (350, 140)
    assert mine.max() > 1                       # non-trivial (has real counts)
    np.testing.assert_array_equal(mine, legacy)


# --------------------------------------------------------------------------- #
# 3. Channel maps                                                             #
# --------------------------------------------------------------------------- #
def test_consecutive_pool_map():
    cmap = consecutive_pool_map(700, 5)
    assert cmap.shape == (700,)
    assert int(cmap.max()) + 1 == 140
    assert cmap[0] == 0 and cmap[4] == 0 and cmap[5] == 1 and cmap[699] == 139


def test_nmnist_pool_map():
    cmap = nmnist_pool_map(34, 34, 2, pool=2)
    assert cmap.shape == (2312,)
    assert int(cmap.max()) + 1 == 578 == nmnist_pool_n_channels(34, 34, 2, 2)
    # channel for (x=3, y=5, pol=1) -> pooled (x'=1, y'=2, pol=1)
    in_ch = 1 * 34 * 34 + 5 * 34 + 3
    assert cmap[in_ch] == 1 * (17 * 17) + 2 * 17 + 1


# --------------------------------------------------------------------------- #
# 4. Cache build + load round-trip                                            #
# --------------------------------------------------------------------------- #
def test_cache_build_and_load_roundtrip(tmp_path):
    data_path = _fake_shd_data_path(tmp_path / "shd")
    params = default_params("shd")
    root = tmp_path / "cache"

    d = build_cache("shd", "train", params, data_path=str(data_path), root=root)
    assert (d / "X.npy").is_file() and (d / "params.json").is_file()

    meta = json.loads((d / "params.json").read_text())
    assert meta["n_inputs"] == 140 and meta["n_classes"] == 20
    assert meta["T"] == 350 and meta["N"] == 2 and meta["count_dtype"] == "uint8"

    ds = load_cached("shd", "train", params, data_path=str(data_path), root=root)
    assert isinstance(ds, SpikeDataset)
    assert len(ds) == 2
    assert ds.n_inputs == 140 and ds.n_classes == 20 and ds.T == 350
    assert np.asarray(ds._X).dtype == np.uint8

    x0, y0 = ds[0]
    assert x0.shape == (350, 140) and x0.dtype == np.float32
    assert y0 in (3, 7)


def test_cache_hit_no_source_reread(tmp_path):
    data_path = _fake_shd_data_path(tmp_path / "shd")
    params = default_params("shd")
    root = tmp_path / "cache"

    d = build_cache("shd", "train", params, data_path=str(data_path), root=root)
    xnpy = d / "X.npy"
    mtime0 = xnpy.stat().st_mtime_ns

    # Remove the source entirely: a warm cache must not touch it.
    import shutil

    shutil.rmtree(data_path)
    ds = load_cached("shd", "train", params, data_path=str(data_path), root=root)
    assert len(ds) == 2
    assert xnpy.stat().st_mtime_ns == mtime0  # not rebuilt


def test_cache_key_stability_and_separation(tmp_path):
    params = default_params("shd")
    k1 = cache_key("shd", "train", params)
    k2 = cache_key("shd", "train", dict(params))
    assert k1 == k2 and len(k1) == 16          # deterministic, same params

    other = dict(params, bin_size_ms=10.0)
    assert cache_key("shd", "train", other) != k1
    assert cache_key("shd", "test", params) != k1  # split is part of the key

    data_path = _fake_shd_data_path(tmp_path / "shd")
    root = tmp_path / "cache"
    d1 = build_cache("shd", "train", params, data_path=str(data_path), root=root)
    d2 = build_cache("shd", "train", other, data_path=str(data_path), root=root)
    assert d1 != d2
    assert json.loads((d1 / "params.json").read_text())["params"] == params


# --------------------------------------------------------------------------- #
# 5. Augment-on-read (cache stays clean)                                      #
# --------------------------------------------------------------------------- #
def test_augment_off_returns_clean_and_stable(tmp_path):
    data_path = _fake_shd_data_path(tmp_path / "shd")
    params = default_params("shd")
    root = tmp_path / "cache"
    ds = load_cached("shd", "train", params, data_path=str(data_path), root=root)

    x_a, _ = ds[0]
    x_b, _ = ds[0]
    np.testing.assert_array_equal(x_a, x_b)                    # no per-read mutation
    np.testing.assert_array_equal(x_a, np.asarray(ds._X[0], dtype=np.float32))


def test_channel_shift_semantics():
    x = np.arange(12, dtype=np.float32).reshape(3, 4)  # (T=3, C=4)
    # the three possible outcomes of a +/-1 shift (drop + zero-fill, no wrap)
    out_pos = np.zeros_like(x)
    out_pos[:, 1:] = x[:, :3]           # shift +1: channel c -> c+1
    out_neg = np.zeros_like(x)
    out_neg[:, :3] = x[:, 1:]           # shift -1: channel c -> c-1
    outcomes = {
        tuple(x.ravel()),               # shift 0
        tuple(out_pos.ravel()),
        tuple(out_neg.ravel()),
    }
    seen = {
        tuple(channel_shift(x, 1, np.random.default_rng(s)).ravel())
        for s in range(50)
    }
    assert seen == outcomes             # fn only ever produces these three
    # shift_range<=0 is a no-op passthrough
    np.testing.assert_array_equal(channel_shift(x, 0, np.random.default_rng(0)), x)


def test_spike_dropout_only_drops_nonzero(tmp_path):
    x = np.zeros((10, 10), dtype=np.float32)
    x[::2, ::2] = 5.0
    rng = np.random.default_rng(7)
    out = spike_dropout(x, p_drop=0.5, rng=rng)
    zero_before = x == 0
    assert np.all(out[zero_before] == 0)           # zeros never become nonzero
    dropped = (x != 0) & (out == 0)
    assert dropped.sum() > 0                        # something was dropped
    assert 0.2 < dropped.sum() / (x != 0).sum() < 0.8  # ~p


def test_augment_does_not_bake_into_cache(tmp_path):
    data_path = _fake_shd_data_path(tmp_path / "shd")
    params = default_params("shd")
    root = tmp_path / "cache"
    aug = Augment(spike_dropout=0.5)
    ds = load_cached(
        "shd", "train", params, data_path=str(data_path), root=root,
        augment=aug, rng=np.random.default_rng(3),
    )
    clean = np.asarray(ds._X[0], dtype=np.float32)
    x_aug, _ = ds[0]
    # augmented read only ever zeros existing spikes (subset of clean support)
    assert np.all((x_aug != 0) <= (clean != 0))
    # the underlying memmap is untouched by the read
    np.testing.assert_array_equal(np.asarray(ds._X[0], dtype=np.float32), clean)


# --------------------------------------------------------------------------- #
# 6. Batch iterator                                                           #
# --------------------------------------------------------------------------- #
def test_iterate_batches_shapes_and_coverage(tmp_path):
    data_path = _fake_shd_data_path(tmp_path / "shd")
    params = default_params("shd")
    root = tmp_path / "cache"
    ds = load_cached("shd", "train", params, data_path=str(data_path), root=root)

    seen = 0
    for X, y in iterate_batches(ds, batch_size=1):
        assert X.ndim == 3 and X.shape[1:] == (350, 140) and X.dtype == np.float32
        assert y.dtype == np.int64
        seen += X.shape[0]
    assert seen == len(ds)


# --------------------------------------------------------------------------- #
# 7. NMNIST binary reader + spatial pool                                      #
# --------------------------------------------------------------------------- #
def test_decode_nmnist_events():
    raw = _nmnist_event_bytes(x=3, y=5, pol=1, time=100)
    units, times_ms = decode_nmnist_events(raw, duration_ms=300.0)
    assert units.tolist() == [1 * 34 * 34 + 5 * 34 + 3]     # channel 1329
    assert times_ms.tolist() == [round(100 * 300 / 336040)]  # rescaled


def test_nmnist_cache_pooled_shape(tmp_path):
    # one sample of class 7 with a single event -> pooled channel present.
    label_dir = tmp_path / "nm" / "Train" / "7"
    label_dir.mkdir(parents=True)
    (label_dir / "s.bin").write_bytes(_nmnist_event_bytes(x=3, y=5, pol=1, time=100))

    params = default_params("nmnist")
    root = tmp_path / "cache"
    ds = load_cached("nmnist", "train", params, data_path=str(tmp_path / "nm"), root=root)
    assert ds.n_inputs == 578 and ds.n_classes == 10
    x, y = ds[0]
    assert y == 7 and x.shape[1] == 578
    pooled_ch = 1 * (17 * 17) + 2 * 17 + 1
    assert x[:, pooled_ch].sum() == 1


# --------------------------------------------------------------------------- #
# 8. Protocol conformance                                                     #
# --------------------------------------------------------------------------- #
def test_protocol_conformance(tmp_path):
    data_path = _fake_shd_data_path(tmp_path / "shd")
    params = default_params("shd")
    ds = load_cached("shd", "train", params, data_path=str(data_path), root=tmp_path / "c")
    assert isinstance(ds, SpikeDataset)
    assert isinstance(ds, CachedSpikeDataset)
    assert isinstance(ds.cache_key(), str)


# --------------------------------------------------------------------------- #
# 9. TensorFlow is severed                                                    #
# --------------------------------------------------------------------------- #
def test_no_tensorflow_in_data_path():
    src = _ROOT / "src"
    code = "import dendroprop.data, sys; assert 'tensorflow' not in sys.modules"
    r = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PYTHONPATH": str(src)},
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, f"stdout={r.stdout!r} stderr={r.stderr!r}"


def test_requirements_no_tensorflow_and_legacy_retired():
    req = (_ROOT / "requirements.txt").read_text().splitlines()
    active = [ln.strip() for ln in req if ln.strip() and not ln.strip().startswith("#")]
    assert not any(ln.lower().startswith("tensorflow") for ln in active)

    # No legacy top-level data/ package and no reference-model dirs.
    assert not (_ROOT / "data").exists()
    for d in ("Original", "DFA", "RFA"):
        assert not (_ROOT / d).exists()


# =========================================================================== #
# 10. Hardening from adversarial review (coverage gaps + regression guards)    #
# =========================================================================== #
def _write_fake_ssc_gz(data_path: Path, samples, labels, split="valid"):
    """Write a fake SSC archive: ssc_<split>.h5.gz (same HDF5 layout as SHD)."""
    data_path.mkdir(parents=True, exist_ok=True)
    h5_tmp = data_path / f"ssc_{split}.h5"
    _write_fake_shd_h5(h5_tmp, samples, labels)
    raw = h5_tmp.read_bytes()
    h5_tmp.unlink()
    with gzip.open(data_path / f"ssc_{split}.h5.gz", "wb") as g:
        g.write(raw)


def test_bin_events_uint8_saturates_at_255():
    # >255 events into a single (bin, channel) cell must saturate, not wrap mod 256.
    cmap = consecutive_pool_map(1, 1)
    units = np.zeros(300, dtype=np.int64)
    times_ms = np.ones(300)
    x = bin_events(
        units, times_ms, T_bins=1, bin_size_ms=4.0, channel_map=cmap,
        n_channels_out=1, max_duration_ms=8.0,
    )
    assert x.dtype == np.uint8
    assert x[0, 0] == 255                       # saturating clip (raw count 300)
    assert np.array(300).astype(np.uint8) == 44  # sanity: uncast wrap would give 44


def test_bin_events_max_duration_none_no_window_drop():
    cmap = consecutive_pool_map(2, 1)
    units = np.array([0, 1, 0])
    times_ms = np.array([1.0, 5.0, 999.0])      # late event: clamped, not dropped
    x = bin_events(
        units, times_ms, T_bins=3, bin_size_ms=4.0, channel_map=cmap,
        n_channels_out=2, max_duration_ms=None,
    )
    assert x.shape == (3, 2)
    assert x.sum() == 3                          # nothing dropped (no window)
    assert x[2, 0] == 1                          # 999ms clamped into the last bin


def test_ssc_cache_roundtrip(tmp_path):
    data_path = tmp_path / "ssc"
    _write_fake_ssc_gz(data_path, [(np.array([0, 5, 699]), np.array([0.1, 0.2, 0.3]))] * 2,
                       labels=[3, 7], split="valid")
    params = default_params("ssc")
    ds = load_cached("ssc", "valid", params, data_path=str(data_path), root=tmp_path / "c")
    assert ds.n_classes == 35                    # SSC-specific
    assert ds.T == 250 and ds.n_inputs == 140    # 1000ms / 4ms window, 700->140
    assert len(ds) == 2
    # the gz->h5 decompress branch ran
    assert (data_path / "ssc_valid.h5").is_file()
    with pytest.raises(ValueError):
        load_cached("ssc", "bogus", params, data_path=str(data_path), root=tmp_path / "c")


def test_build_cache_max_duration_none_uses_source_default(tmp_path):
    data_path = _fake_shd_data_path(tmp_path / "shd")
    params = dict(default_params("shd"), max_duration_ms=None)
    d = build_cache("shd", "train", params, data_path=str(data_path), root=tmp_path / "c")
    meta = json.loads((d / "params.json").read_text())
    assert meta["T"] == 350                      # 1400ms (SHD default) / 4ms


def test_build_cache_force_rebuilds_from_source(tmp_path):
    data_path = _fake_shd_data_path(tmp_path / "shd")  # 2 samples
    params = default_params("shd")
    root = tmp_path / "c"
    d = build_cache("shd", "train", params, data_path=str(data_path), root=root)
    assert json.loads((d / "params.json").read_text())["N"] == 2

    # grow the source to 3 samples
    rng = np.random.default_rng(1)
    s = [(rng.integers(0, 700, 20), rng.uniform(0, 1.399, 20)) for _ in range(3)]
    _write_fake_shd_h5(data_path / "hdspikes" / "shd_train.h5", s, labels=[3, 7, 3])

    # warm rebuild ignores the changed source; force=True re-reads it
    d_warm = build_cache("shd", "train", params, data_path=str(data_path), root=root)
    assert json.loads((d_warm / "params.json").read_text())["N"] == 2
    d_force = build_cache("shd", "train", params, data_path=str(data_path), root=root, force=True)
    assert json.loads((d_force / "params.json").read_text())["N"] == 3


def test_empty_dataset_path_and_warning(tmp_path, capsys):
    data_path = _fake_shd_data_path(tmp_path / "shd")  # labels 3, 7
    params = dict(default_params("shd"), target_classes=[99])  # no such class
    root = tmp_path / "c"
    d = build_cache("shd", "train", params, data_path=str(data_path), root=root)
    assert "0 samples" in capsys.readouterr().out       # warned, not silent

    meta = json.loads((d / "params.json").read_text())
    assert meta["N"] == 0
    ds = load_cached("shd", "train", params, data_path=str(data_path), root=root)
    assert len(ds) == 0 and ds.n_inputs == 140 and ds.T == 350
    assert list(iterate_batches(ds, batch_size=4)) == []  # no error over len 0


def test_tuple_target_classes_warm_cache_no_rebuild(tmp_path):
    # Regression: tuple params must canonicalize so a warm cache is recognized,
    # not rebuilt (which, with the source gone, would try to re-download).
    import shutil

    data_path = _fake_shd_data_path(tmp_path / "shd")
    params = dict(default_params("shd"), target_classes=(3, 7))  # a TUPLE
    root = tmp_path / "c"
    d = build_cache("shd", "train", params, data_path=str(data_path), root=root)
    mtime0 = (d / "X.npy").stat().st_mtime_ns

    shutil.rmtree(data_path)                     # source gone -> only a warm cache can serve
    ds = load_cached("shd", "train", params, data_path=str(data_path), root=root)
    assert len(ds) == 2
    assert (d / "X.npy").stat().st_mtime_ns == mtime0     # not rebuilt


def test_iterate_batches_drop_last_and_shuffle_determinism(tmp_path):
    data_path = _fake_shd_data_path(tmp_path / "shd")  # N=2
    params = default_params("shd")
    ds = load_cached("shd", "train", params, data_path=str(data_path), root=tmp_path / "c")

    # drop_last drops the short final batch; without it the tail is kept
    kept = [b[0].shape[0] for b in iterate_batches(ds, batch_size=2)]
    assert kept == [2]
    kept_drop = [b[0].shape[0] for b in iterate_batches(ds, batch_size=2, drop_last=True)]
    assert kept_drop == [2]
    ragged = [b[0].shape[0] for b in iterate_batches(ds, batch_size=3)]
    assert ragged == [2]                         # single short batch when B > N
    ragged_drop = [b[0].shape[0] for b in iterate_batches(ds, batch_size=3, drop_last=True)]
    assert ragged_drop == []                     # dropped

    # seeded shuffle is deterministic and preserves full coverage
    def labels(seed):
        out = []
        for _, y in iterate_batches(ds, batch_size=1, shuffle=True, rng=np.random.default_rng(seed)):
            out.extend(y.tolist())
        return out

    assert labels(5) == labels(5)                # same seed -> same order
    assert sorted(labels(5)) == sorted([3, 7])   # coverage preserved


def test_channel_shift_no_alias():
    x = np.arange(8, dtype=np.float32).reshape(2, 4)
    out = channel_shift(x, 0, np.random.default_rng(0))   # no-op branch
    np.testing.assert_array_equal(out, x)
    assert not np.shares_memory(out, x)          # a fresh copy, not an alias


def test_augment_channel_shift_read_through(tmp_path):
    data_path = _fake_shd_data_path(tmp_path / "shd")
    params = default_params("shd")
    root = tmp_path / "c"

    # channel-shift only: the read must equal channel_shift(clean, k, rng) with the same rng
    ds = load_cached("shd", "train", params, data_path=str(data_path), root=root,
                     augment=Augment(channel_shift=3), rng=np.random.default_rng(0))
    clean = np.array(ds._X[0], dtype=np.float32)
    x_aug, _ = ds[0]
    expected = channel_shift(clean, 3, np.random.default_rng(0))
    np.testing.assert_array_equal(x_aug, expected)
    assert Augment(channel_shift=3).active() and not Augment().active()

    # compose order is shift THEN dropout
    ds2 = load_cached("shd", "train", params, data_path=str(data_path), root=root,
                      augment=Augment(channel_shift=3, spike_dropout=0.3),
                      rng=np.random.default_rng(1))
    x2, _ = ds2[0]
    r = np.random.default_rng(1)
    expected2 = spike_dropout(channel_shift(clean, 3, r), 0.3, r)
    np.testing.assert_array_equal(x2, expected2)
