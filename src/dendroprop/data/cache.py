"""Preprocess-once, memmap-forever cache keyed by a params hash.

``build_cache`` runs a source adapter through :func:`~dendroprop.data.preprocess.
bin_events`, stacks the samples, and writes an uncompressed `.npy` cache plus a
JSON sidecar of the exact preprocessing params (whose hash is the cache key).
``load_cached`` memmaps the big count array read-only. A warm cache needs no
source, no download, and no TensorFlow.

Uncompressed `.npy` (not `.npz`) is deliberate: `np.load(mmap_mode='r')` pages the
big `X` from disk instead of inflating it whole. Counts are stored `uint8`
(cast to float on read) — ~4x smaller than float32 and exact for realistic counts.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Optional

import numpy as np

from .dataset import Augment, CachedSpikeDataset
from .preprocess import bin_events, consecutive_pool_map, nmnist_pool_map
from .sources import SOURCES


# Paper-default preprocessing params per dataset. `max_duration_ms=None` here means
# "use the source's default_max_duration_ms".
DEFAULT_PARAMS = {
    "shd": dict(
        bin_size_ms=4.0,
        max_duration_ms=1400.0,
        collapse_factor=5,
        binarize=False,
        target_classes=None,
        max_samples_per_class=None,
    ),
    "ssc": dict(
        bin_size_ms=4.0,
        max_duration_ms=1000.0,
        collapse_factor=5,
        binarize=False,
        target_classes=None,
        max_samples_per_class=None,
    ),
    "nmnist": dict(
        bin_size_ms=1.0,
        max_duration_ms=300.0,
        spatial_pool=2,
        binarize=False,
        target_classes=None,
        max_samples_per_class=None,
    ),
}


def default_params(name: str) -> dict:
    """A fresh copy of the paper-default params for dataset `name`."""
    if name not in DEFAULT_PARAMS:
        raise KeyError(f"Unknown dataset {name!r}; known: {list(DEFAULT_PARAMS)}")
    return dict(DEFAULT_PARAMS[name])


def default_cache_root() -> Path:
    """Root cache directory: `$DENDRO_CACHE_DIR` or `~/.cache/dendroprop`."""
    env = os.environ.get("DENDRO_CACHE_DIR")
    return Path(env) if env else Path.home() / ".cache" / "dendroprop"


def _canonical_params(params: dict) -> dict:
    """Round-trip params through JSON so every equivalent form normalizes identically.

    Tuples and lists both become lists, and any JSON-lossy value (e.g. a numpy
    scalar) is coerced instead of crashing the sidecar write. This makes the stored
    ``params.json`` and a later request compare equal, so a warm cache built with
    ``target_classes=(0,1)`` is recognized on reload rather than rebuilt every time.
    Downstream reads coerce with ``int()``/``float()``/``bool()``, so string-coerced
    values still resolve correctly.
    """
    return json.loads(json.dumps(params, sort_keys=True, default=str))


def cache_key(name: str, split: str, params: dict) -> str:
    """Deterministic 16-hex cache key for (name, split, params).

    A canonical (sorted-key) JSON dump hashed with sha1 — stable across processes.
    """
    blob = json.dumps(
        {"name": name, "split": split, "params": _canonical_params(params)},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def _channel_map(name: str, params: dict, n_units_in: int):
    """Build the spatial-pool channel map + output channel count from params."""
    if name in ("shd", "ssc"):
        cf = int(params.get("collapse_factor", 5))
        cmap = consecutive_pool_map(n_units_in, cf)
    elif name == "nmnist":
        pool = int(params.get("spatial_pool", 2))
        cmap = nmnist_pool_map(pool=pool)
    else:
        raise KeyError(f"No channel map for dataset {name!r}")
    return cmap, int(cmap.max()) + 1


def _cache_dir(name: str, split: str, params: dict, root: Optional[Path]) -> Path:
    root = Path(root) if root is not None else default_cache_root()
    return root / name / split / cache_key(name, split, params)


def _is_valid(d: Path, params: dict) -> bool:
    """A cache dir is valid iff all arrays + a sidecar with matching params exist."""
    if not d.is_dir():
        return False
    for f in ("X.npy", "y.npy", "lengths.npy", "params.json"):
        if not (d / f).is_file():
            return False
    try:
        meta = json.loads((d / "params.json").read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return meta.get("params") == params  # guard hash collisions / schema drift


def build_cache(
    name: str,
    split: str,
    params: dict,
    *,
    data_path: Optional[str] = None,
    root: Optional[Path] = None,
    force: bool = False,
) -> Path:
    """Preprocess (name, split) under `params` into a `.npy` cache; return its dir.

    Idempotent: if a valid cache already exists and `force` is False, returns it
    without touching the source. Builds into a sibling `.tmp` dir then atomically
    renames, so a crash never leaves a half-written cache in place.
    """
    if name not in SOURCES:
        raise KeyError(f"Unknown dataset {name!r}; known: {list(SOURCES)}")
    params = _canonical_params(params)
    d = _cache_dir(name, split, params, root)
    if not force and _is_valid(d, params):
        return d

    src = SOURCES[name](split, data_path=data_path)
    cmap, n_channels_out = _channel_map(name, params, src.n_units_in)

    bin_size = float(params.get("bin_size_ms", 4.0))
    max_dur = params.get("max_duration_ms", None)
    if max_dur is None:
        max_dur = src.default_max_duration_ms
    max_dur = float(max_dur)
    # Keep the event-time rescale target aligned with the bin window so events are
    # re-timed to the requested duration rather than silently dropped (NMNIST rescales
    # raw AER timestamps into [0, duration_ms]).
    if hasattr(src, "duration_ms"):
        src.duration_ms = max_dur
    T_bins = max(1, int(np.ceil(max_dur / bin_size)))
    binarize = bool(params.get("binarize", False))

    xs, ys, lens = [], [], []
    for units, times_ms, label in src.iter_samples(
        params.get("target_classes"), params.get("max_samples_per_class")
    ):
        x = bin_events(
            units,
            times_ms,
            T_bins=T_bins,
            bin_size_ms=bin_size,
            channel_map=cmap,
            n_channels_out=n_channels_out,
            max_duration_ms=max_dur,
            binarize=binarize,
            count_dtype=np.uint8,
        )
        populated = np.flatnonzero(x.any(axis=1))
        lens.append(int(populated[-1] + 1) if populated.size else 0)
        xs.append(x)
        ys.append(int(label))

    if xs:
        X = np.stack(xs, axis=0)
    else:
        X = np.zeros((0, T_bins, n_channels_out), dtype=np.uint8)
        print(
            f"[dendroprop.data] WARNING: {name}/{split} cache built with 0 samples "
            f"— the source yielded nothing (check data_path and target_classes). "
            f"This empty cache is treated as valid; pass force=True to rebuild once "
            f"the source is fixed.",
            flush=True,
        )
    y = np.asarray(ys, dtype=np.int64)
    lengths = np.asarray(lens, dtype=np.int64)

    meta = {
        "dataset": name,
        "split": split,
        "cache_key": cache_key(name, split, params),
        "params": params,
        "n_inputs": int(n_channels_out),
        "n_classes": int(src.n_classes),
        "T": int(T_bins),
        "N": int(y.shape[0]),
        "count_dtype": "uint8",
    }

    tmp = d.parent / (d.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    np.save(tmp / "X.npy", X)
    np.save(tmp / "y.npy", y)
    np.save(tmp / "lengths.npy", lengths)
    (tmp / "params.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    if d.exists():
        shutil.rmtree(d)
    os.replace(tmp, d)
    return d


def load_cached(
    name: str,
    split: str,
    params: dict,
    *,
    data_path: Optional[str] = None,
    root: Optional[Path] = None,
    augment: Optional[Augment] = None,
    rng: Optional[np.random.Generator] = None,
) -> CachedSpikeDataset:
    """Load (build if missing) a cached SpikeDataset with the big `X` memmapped."""
    d = build_cache(name, split, params, data_path=data_path, root=root)
    X = np.load(d / "X.npy", mmap_mode="r")
    y = np.load(d / "y.npy")
    lengths = np.load(d / "lengths.npy")
    meta = json.loads((d / "params.json").read_text())
    return CachedSpikeDataset(X, y, lengths, meta, augment=augment, rng=rng)
