"""
SSC (Spiking Speech Commands) count-binning preprocessing.

This is the SSC counterpart of ``data/shd_binned.py``. SSC ships in the exact
same Zenke-lab HDF5 format as SHD (700 cochlea channels; per-sample
``spikes/units`` + ``spikes/times`` + ``labels``), so the binning pipeline is
identical. The only dataset-level differences are:

  - 35 classes            (SHD has 20)
  - train/valid/test split (SHD has only train/test)
  - 1 s clips, so the default fixed window is 1000 ms (SHD uses 1400 ms)

Pipeline per sample (same as SHD):
  1. Read raw events (units, times in seconds) from the HDF5 file.
  2. Time binning:    bin index = floor(time_ms / bin_size_ms),
                      cell value = number of spikes in that (bin, channel).
  3. Spatial binning: sum-pool every `collapse_factor` consecutive input
                      channels into one output channel (e.g. 700 -> 140 with
                      collapse_factor=5).
  4. Length handling: every sample is zero-padded out to the same number of
                      time bins so all examples share (T_bins, n_channels_out).

Cross-references:
  - Paper § 4.1 Experimental setup
  - sparch dataloaders/spiking_datasets.py (BSD-3, Bittar & Garner)

Like ``shd_binned.py`` this module is intentionally standalone (only NumPy +
h5py/tables + stdlib) so it doesn't pull in the TensorFlow dependency that
``data/ssc.py`` carries transitively.
"""
from __future__ import annotations

import gzip
import os
import shutil
import urllib.request
from typing import List, Optional, Tuple

import numpy as np


N_UNITS_SSC = 700
N_CLASSES_SSC = 35
ZENKE_BASE_URL = "https://zenkelab.org/datasets"
_SPLIT_FILES = {
    "train": "ssc_train.h5.gz",
    "valid": "ssc_valid.h5.gz",
    "test": "ssc_test.h5.gz",
}


def _open_hdf5(hdf5_path: str):
    """Open an HDF5 file with PyTables if available, else h5py.

    Returns (file_handle, units_dataset, times_dataset, labels_array,
             speaker_array_or_None, use_h5py_flag).
    `speaker_array` is the per-sample speaker id (SSC stores it under
    extra/speaker); None if the file has no such field.
    Imports are deferred so the binning helpers can be used without hdf5 deps installed.
    """
    try:
        import tables  # noqa: WPS433
        fh = tables.open_file(hdf5_path, mode="r")
        # extra/speaker is optional metadata, unused by training. SSC encodes it
        # in a form that may not cast to int64, so degrade to None on any error.
        try:
            speaker = np.asarray(fh.root.extra.speaker[:]).astype(np.int64, copy=False)
        except (AttributeError, ValueError, TypeError, OSError, tables.NoSuchNodeError):
            speaker = None
        return (
            fh,
            fh.root.spikes.units,
            fh.root.spikes.times,
            np.asarray(fh.root.labels, dtype=np.int64),
            speaker,
            False,
        )
    except (ImportError, ValueError):
        try:
            import h5py  # noqa: WPS433
        except ImportError as e:
            raise ImportError(
                "Either pytables or h5py must be installed to read SSC HDF5 files. "
                "pip install h5py"
            ) from e
        fh = h5py.File(hdf5_path, mode="r")
        # extra/speaker is optional metadata, unused by training. SSC encodes it
        # in a form that may not cast to int64, so degrade to None on any error.
        speaker = None
        if "extra" in fh and "speaker" in fh["extra"]:
            try:
                speaker = np.asarray(fh["extra"]["speaker"][()]).astype(np.int64, copy=False)
            except (ValueError, TypeError, OSError):
                speaker = None
        return (
            fh,
            fh["spikes"]["units"],
            fh["spikes"]["times"],
            np.asarray(fh["labels"], dtype=np.int64),
            speaker,
            True,
        )


def _default_cache_dir() -> str:
    """Where SSC .h5.gz files live. Mirrors data/ssc.py's path priority, with
    SHD-style scratch fallbacks so the same code runs on the cluster."""
    if "SSC_DATA_PATH" in os.environ:
        return os.environ["SSC_DATA_PATH"]
    if "SSC_CACHE_DIR" in os.environ:
        return os.environ["SSC_CACHE_DIR"]
    for p in (
        "/home/staff/t/tibax/Downloads/SSC_data",
        "/share/neurocomputation/Tim/SSC_data",
        "/Users/tbax/Documents/SSC",
    ):
        if os.path.exists(p):
            return p
    if "SCRATCH" in os.environ:
        return os.path.join(os.environ["SCRATCH"], "SSC_data")
    if "TMPDIR" in os.environ and os.environ.get("TMPDIR") != "/tmp":
        return os.path.join(os.environ["TMPDIR"], "SSC_data")
    return os.path.expanduser("~/Downloads/SSC_data")


def _resolve_and_gunzip(split: str, cache_dir: Optional[str] = None) -> str:
    """Locate (downloading + gunzipping if needed) the SSC .h5 for `split`.

    SSC clips are large; if the .h5.gz is already present locally (the common
    case on the cluster) nothing is downloaded — it is only decompressed.
    Returns the path to the decompressed .h5 file.
    """
    if split not in _SPLIT_FILES:
        raise ValueError(f"Unknown split: {split}. Must be one of {list(_SPLIT_FILES)}")
    cache_dir = cache_dir or _default_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)

    gz_name = _SPLIT_FILES[split]
    gz_path = os.path.join(cache_dir, gz_name)
    h5_path = gz_path[:-3] if gz_path.endswith(".gz") else gz_path + ".h5"

    if not os.path.isfile(gz_path) and not os.path.isfile(h5_path):
        url = f"{ZENKE_BASE_URL}/{gz_name}"
        print(f"Downloading {url} -> {gz_path}", flush=True)
        urllib.request.urlretrieve(url, gz_path)

    if not os.path.isfile(h5_path) or (
        os.path.isfile(gz_path) and os.path.getctime(gz_path) > os.path.getctime(h5_path)
    ):
        print(f"Decompressing {gz_path}", flush=True)
        with gzip.open(gz_path, "rb") as fin, open(h5_path, "wb") as fout:
            shutil.copyfileobj(fin, fout)
    return h5_path


def bin_ssc_sample(
    units: np.ndarray,
    times_sec: np.ndarray,
    bin_size_ms: float = 4.0,
    collapse_factor: int = 5,
    n_units_in: int = N_UNITS_SSC,
    max_duration_ms: Optional[float] = 1000.0,
    binarize: bool = False,
    dtype=np.float32,
) -> np.ndarray:
    """Bin one SSC sample's raw events into a (T_bins, n_channels_out) count matrix.

    Identical binning to ``shd_binned.bin_shd_sample`` (the operation is
    dataset-agnostic); only the default window (1000 ms) differs.

    Args:
        units:           int array of input channel indices in [0, n_units_in).
        times_sec:       float array of spike timestamps in seconds.
        bin_size_ms:     time bin width, ms.
        collapse_factor: number of consecutive input channels summed into one
                         output channel (1 = no collapsing).
        n_units_in:      number of original input channels (700 for SSC).
        max_duration_ms: events past this are dropped (not squashed). Sets
                         T_bins = ceil(max_duration_ms / bin_size_ms). If None,
                         T_bins is set to the sample's own max bin index + 1.
        binarize:        if True, clip counts to {0, 1} after binning.
        dtype:           output dtype.

    Returns:
        x: ndarray of shape (T_bins, n_channels_out).
    """
    units = np.asarray(units, dtype=np.int64)
    times_ms = np.asarray(times_sec, dtype=np.float64) * 1000.0

    if max_duration_ms is not None:
        keep = (times_ms >= 0.0) & (times_ms < max_duration_ms)
        times_ms = times_ms[keep]
        units = units[keep]
        T_bins = max(1, int(np.ceil(max_duration_ms / bin_size_ms)))
    else:
        keep = times_ms >= 0.0
        times_ms = times_ms[keep]
        units = units[keep]
        if times_ms.size:
            T_bins = int(np.floor(times_ms.max() / bin_size_ms)) + 1
        else:
            T_bins = 1

    n_channels_out = (n_units_in + collapse_factor - 1) // collapse_factor

    t_idx = (times_ms / bin_size_ms).astype(np.int64)
    np.minimum(t_idx, T_bins - 1, out=t_idx)
    c_idx = units // collapse_factor

    valid = (c_idx >= 0) & (c_idx < n_channels_out) & (t_idx >= 0)
    t_idx = t_idx[valid]
    c_idx = c_idx[valid]

    x = np.zeros((T_bins, n_channels_out), dtype=dtype)
    np.add.at(x, (t_idx, c_idx), 1)
    if binarize:
        np.minimum(x, 1, out=x)
    return x


def apply_channel_shift(x: np.ndarray, shift_range: int) -> np.ndarray:
    """Shift all channels of one sample up/down by a uniform integer offset.

    Channel-axis analogue of the temporal-jitter augmentation, with drop +
    zero-fill (no clamping, no wrap):

    - One shift per sample (same offset for every timestep / all channels).
    - shift = np.random.randint(-shift_range, shift_range + 1)  # inclusive, may be 0
    - Positive shift s ("up"): channel c -> c+s; the top s channels are pushed
      out and discarded, the bottom s channels become 0. Negative shift is the
      mirror image.
    - Operates on axis 1 (channels) of an (T, C) array.

    Args:
        x:           (T, C) sample array.
        shift_range: max absolute shift in channels; <= 0 returns x unchanged.

    Returns:
        Shifted (T, C) array (same shape and dtype).
    """
    if shift_range <= 0:
        return np.asarray(x)
    x_np = np.asarray(x)
    C = x_np.shape[1]
    shift = int(np.random.randint(-shift_range, shift_range + 1))
    if shift == 0:
        return x_np
    out = np.zeros_like(x_np)
    if shift > 0:
        out[:, shift:] = x_np[:, : C - shift]
    else:
        k = -shift
        out[:, : C - k] = x_np[:, k:]
    return out


class SSCBinnedLoader:
    """SSC loader that returns dense count-binned tensors.

    No TF dependency; uses urllib for download and tables/h5py for HDF5 access.
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        bin_size_ms: float = 4.0,
        collapse_factor: int = 5,
        max_duration_ms: Optional[float] = 1000.0,
    ):
        if bin_size_ms <= 0:
            raise ValueError("bin_size_ms must be > 0")
        if collapse_factor < 1:
            raise ValueError("collapse_factor must be >= 1")

        self.data_path = data_path
        self.n_units = N_UNITS_SSC
        self.bin_size_ms = float(bin_size_ms)
        self.collapse_factor = int(collapse_factor)
        self.max_duration_ms = float(max_duration_ms) if max_duration_ms is not None else None
        self.n_channels_out = (self.n_units + self.collapse_factor - 1) // self.collapse_factor
        self.T_bins = (
            max(1, int(np.ceil(self.max_duration_ms / self.bin_size_ms)))
            if self.max_duration_ms is not None
            else None
        )

    def get_dataset_binned(
        self,
        split: str = "train",
        max_samples_per_class: Optional[int] = None,
        target_classes: Optional[List[int]] = None,
        binarize: bool = False,
        dtype=np.float32,
        with_speaker: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (X, y, lengths) — or (X, y, lengths, speakers) if with_speaker.
            X        : (N, T_bins, n_channels_out) float array
            y        : (N,)                         int64
            lengths  : (N,)                         int64, last populated time bin + 1
                       (useful if you want to mask the trailing zeros)
            speakers : (N,)                         int64 per-sample speaker id
        """
        if target_classes is None:
            target_classes = list(range(N_CLASSES_SSC))
        if split not in _SPLIT_FILES:
            raise ValueError(f"Unknown split: {split}. Must be one of {list(_SPLIT_FILES)}")
        hdf5_path = _resolve_and_gunzip(split, cache_dir=self.data_path)

        fh, units_ds, times_ds, labels, speaker, use_h5py = _open_hdf5(hdf5_path)
        if with_speaker and speaker is None:
            fh.close()
            raise ValueError(
                f"with_speaker=True but no extra/speaker field in ssc_{split}.h5"
            )

        xs: List[np.ndarray] = []
        ys: List[int] = []
        lens: List[int] = []
        spk: List[int] = []
        class_counts = {c: 0 for c in target_classes}

        for i in range(labels.shape[0]):
            label = int(labels[i])
            if label not in target_classes:
                continue
            if max_samples_per_class is not None and class_counts[label] >= max_samples_per_class:
                continue
            u = units_ds[i][:] if use_h5py else units_ds[i]
            t = times_ds[i][:] if use_h5py else times_ds[i]

            x = bin_ssc_sample(
                u, t,
                bin_size_ms=self.bin_size_ms,
                collapse_factor=self.collapse_factor,
                n_units_in=self.n_units,
                max_duration_ms=self.max_duration_ms,
                binarize=binarize,
                dtype=dtype,
            )
            populated = np.flatnonzero(x.any(axis=1))
            length = int(populated[-1] + 1) if populated.size else 0

            xs.append(x)
            ys.append(label)
            lens.append(length)
            if speaker is not None:
                spk.append(int(speaker[i]))
            class_counts[label] += 1

        fh.close()

        if self.T_bins is not None:
            X = np.stack(xs, axis=0)
        else:
            T_max = max(x.shape[0] for x in xs) if xs else 1
            X = np.zeros((len(xs), T_max, self.n_channels_out), dtype=dtype)
            for k, x in enumerate(xs):
                X[k, : x.shape[0]] = x

        y = np.asarray(ys, dtype=np.int64)
        lengths = np.asarray(lens, dtype=np.int64)
        speakers = np.asarray(spk, dtype=np.int64)

        print(
            f"[ssc_binned/{split}] N={X.shape[0]}, "
            f"shape={tuple(X.shape)}, "
            f"bin={self.bin_size_ms}ms, collapse={self.collapse_factor} "
            f"(700 -> {self.n_channels_out} channels), "
            f"mean count/bin={X.mean():.4f}, max count/bin={X.max():.0f}, "
            f"median length={int(np.median(lengths)) if len(lengths) else 0} bins",
            flush=True,
        )
        if with_speaker:
            return X, y, lengths, speakers
        return X, y, lengths


def load_ssc_binned(
    bin_size_ms: float = 4.0,
    collapse_factor: int = 5,
    max_duration_ms: Optional[float] = 1000.0,
    train_samples_per_class: Optional[int] = None,
    eval_samples_per_class: Optional[int] = None,
    target_classes: Optional[List[int]] = None,
    binarize: bool = False,
    dtype=np.float32,
    data_path: Optional[str] = None,
    eval_split: str = "test",
    return_speakers: bool = False,
):
    """Convenience: load SSC train + one eval split in the paper's preprocessing.

    Args:
        eval_split: which split to return as the evaluation set ("valid" or
                    "test"). SSC ships a separate validation split; use "valid"
                    for model selection / tuning and "test" for the final number.

    Returns:
        X_train, y_train, len_train, X_eval, y_eval, len_eval
        (+ spk_train, spk_eval appended when return_speakers=True)
    """
    if eval_split not in ("valid", "test"):
        raise ValueError("eval_split must be 'valid' or 'test'")
    loader = SSCBinnedLoader(
        data_path=data_path,
        bin_size_ms=bin_size_ms,
        collapse_factor=collapse_factor,
        max_duration_ms=max_duration_ms,
    )
    tr = loader.get_dataset_binned(
        "train",
        max_samples_per_class=train_samples_per_class,
        target_classes=target_classes,
        binarize=binarize,
        dtype=dtype,
        with_speaker=return_speakers,
    )
    ev = loader.get_dataset_binned(
        eval_split,
        max_samples_per_class=eval_samples_per_class,
        target_classes=target_classes,
        binarize=binarize,
        dtype=dtype,
        with_speaker=return_speakers,
    )
    if return_speakers:
        X_tr, y_tr, L_tr, spk_tr = tr
        X_ev, y_ev, L_ev, spk_ev = ev
        return X_tr, y_tr, L_tr, X_ev, y_ev, L_ev, spk_tr, spk_ev
    X_tr, y_tr, L_tr = tr
    X_ev, y_ev, L_ev = ev
    return X_tr, y_tr, L_tr, X_ev, y_ev, L_ev


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="SSC count-based preprocessing (sparch / Fabre et al. 2025 style)."
    )
    p.add_argument("--bin_size_ms", type=float, default=4.0,
                   help="Time bin width in ms (paper default 4.0; also try 10 or 14).")
    p.add_argument("--collapse_factor", type=int, default=5,
                   help="Spatial collapsing: sum-pool every N consecutive channels (paper default 5).")
    p.add_argument("--max_duration_ms", type=float, default=1000.0,
                   help="Fixed window length in ms; spikes past this are dropped (SSC clips are 1 s).")
    p.add_argument("--eval_split", choices=["valid", "test"], default="test")
    p.add_argument("--binarize", action="store_true",
                   help="Cap per-bin counts to {0,1} (default: keep counts).")
    p.add_argument("--data_path", type=str, default="",
                   help="Directory holding ssc_{train,valid,test}.h5.gz (default: auto-detect).")
    p.add_argument("--train_per_class", type=int, default=None)
    p.add_argument("--eval_per_class", type=int, default=None)
    args = p.parse_args()

    X_tr, y_tr, L_tr, X_ev, y_ev, L_ev = load_ssc_binned(
        bin_size_ms=args.bin_size_ms,
        collapse_factor=args.collapse_factor,
        max_duration_ms=args.max_duration_ms,
        binarize=args.binarize,
        data_path=args.data_path or None,
        eval_split=args.eval_split,
        train_samples_per_class=args.train_per_class,
        eval_samples_per_class=args.eval_per_class,
    )
    print(f"\nTrain X: {X_tr.shape}  y: {y_tr.shape}  lengths in [{L_tr.min()}, {L_tr.max()}]")
    print(f"{args.eval_split.capitalize():5s} X: {X_ev.shape}  y: {y_ev.shape}  "
          f"lengths in [{L_ev.min()}, {L_ev.max()}]")
