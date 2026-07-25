"""Per-dataset source adapters — raw events, no TensorFlow.

Each source exposes ``n_units_in``, ``n_classes``, ``default_max_duration_ms`` and
an ``iter_samples(target_classes, max_samples_per_class)`` generator yielding
``(units, times_ms, label)`` per sample. Downstream, :func:`~dendroprop.data.
preprocess.bin_events` turns each event list into a dense count tensor; spatial
pooling (SHD/SSC consecutive, NMNIST 2x2) is applied there via a channel map, so
the sources stay format-only.

HDF5 access (SHD/SSC) uses PyTables when available, else h5py — never TF. NMNIST
decodes the per-file 5-byte-column binary directly. This replaces the legacy
``data/shd.py`` (its only role, ``tensorflow.keras.utils.get_file``, is done here
with ``urllib``).
"""
from __future__ import annotations

import gzip
import os
import shutil
import urllib.request
from typing import Iterator, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# HDF5 helpers (shared by SHD + SSC), lifted from data/shd_binned.py           #
# --------------------------------------------------------------------------- #
def _open_hdf5(hdf5_path: str):
    """Open an SHD/SSC-format HDF5 file with PyTables if available, else h5py.

    Returns (file_handle, units_dataset, times_dataset, labels_array, use_h5py).
    Imports are deferred so the rest of the module works without hdf5 deps.
    """
    try:
        import tables  # noqa: WPS433

        fh = tables.open_file(hdf5_path, mode="r")
        return (
            fh,
            fh.root.spikes.units,
            fh.root.spikes.times,
            np.asarray(fh.root.labels, dtype=np.int64),
            False,
        )
    except (ImportError, ValueError):
        try:
            import h5py  # noqa: WPS433
        except ImportError as e:  # pragma: no cover - depends on env
            raise ImportError(
                "Either pytables or h5py must be installed to read SHD/SSC HDF5 "
                "files. pip install h5py"
            ) from e
        fh = h5py.File(hdf5_path, mode="r")
        return (
            fh,
            fh["spikes"]["units"],
            fh["spikes"]["times"],
            np.asarray(fh["labels"], dtype=np.int64),
            True,
        )


def _default_cache_dir() -> str:
    """Same priority order as the legacy loaders, for path compatibility."""
    if "SHD_CACHE_DIR" in os.environ:
        return os.environ["SHD_CACHE_DIR"]
    if os.path.exists("/share/neurocomputation/Tim/SHD_data"):
        return "/share/neurocomputation/Tim/SHD_data"
    if "SCRATCH" in os.environ:
        return os.path.join(os.environ["SCRATCH"], "data")
    if "TMPDIR" in os.environ and os.environ.get("TMPDIR") != "/tmp":
        return os.path.join(os.environ["TMPDIR"], "data")
    if os.path.exists("/scratch"):
        return "/scratch/data"
    return os.path.expanduser("~/data")


def _download_and_gunzip(filename: str, cache_dir: Optional[str] = None) -> str:
    """Download `filename` from the Zenke lab and gunzip it. Returns the `.h5` path.

    A no-op if the decompressed `.h5` is already present (so a pre-provisioned
    cache dir needs no network).
    """
    cache_dir = cache_dir or _default_cache_dir()
    cache_subdir = os.path.join(cache_dir, "hdspikes")
    os.makedirs(cache_subdir, exist_ok=True)

    gz_path = os.path.join(cache_subdir, filename)
    h5_path = gz_path[:-3] if gz_path.endswith(".gz") else gz_path + ".h5"

    if not os.path.isfile(gz_path) and not os.path.isfile(h5_path):
        url = f"https://zenkelab.org/datasets/{filename}"
        print(f"Downloading {url} -> {gz_path}", flush=True)
        urllib.request.urlretrieve(url, gz_path)

    if not os.path.isfile(h5_path) or (
        os.path.isfile(gz_path)
        and os.path.getctime(gz_path) > os.path.getctime(h5_path)
    ):
        print(f"Decompressing {gz_path}", flush=True)
        with gzip.open(gz_path, "rb") as fin, open(h5_path, "wb") as fout:
            shutil.copyfileobj(fin, fout)
    return h5_path


def _iter_hdf5_samples(
    hdf5_path: str,
    n_classes: int,
    time_scale_to_ms: float,
    target_classes: Optional[List[int]],
    max_samples_per_class: Optional[int],
) -> Iterator[Tuple[np.ndarray, np.ndarray, int]]:
    """Iterate an SHD/SSC HDF5 file yielding (units, times_ms, label)."""
    if target_classes is None:
        target_classes = list(range(n_classes))
    target = set(target_classes)

    fh, units_ds, times_ds, labels, use_h5py = _open_hdf5(hdf5_path)
    class_counts = {c: 0 for c in target_classes}
    try:
        for i in range(labels.shape[0]):
            label = int(labels[i])
            if label not in target:
                continue
            if (
                max_samples_per_class is not None
                and class_counts[label] >= max_samples_per_class
            ):
                continue
            u = np.asarray(units_ds[i][:] if use_h5py else units_ds[i], dtype=np.int64)
            t = np.asarray(times_ds[i][:] if use_h5py else times_ds[i], dtype=np.float64)
            class_counts[label] += 1
            yield u, t * time_scale_to_ms, label
    finally:
        fh.close()


# --------------------------------------------------------------------------- #
# SHD                                                                          #
# --------------------------------------------------------------------------- #
class SHDSource:
    """Spiking Heidelberg Digits — HDF5, times in seconds. No TF."""

    n_units_in = 700
    n_classes = 20
    default_max_duration_ms = 1400.0

    def __init__(self, split: str, data_path: Optional[str] = None, download: bool = True):
        if split not in ("train", "test"):
            raise ValueError(f"Unknown SHD split: {split!r} (expected train/test)")
        self.split = split
        self.data_path = data_path
        self.download = download

    def _h5_path(self) -> str:
        filename = f"shd_{self.split}.h5.gz"
        if self.download:
            return _download_and_gunzip(filename, cache_dir=self.data_path)
        # No download: expect a pre-decompressed .h5 under <data_path>/hdspikes/.
        base = self.data_path or _default_cache_dir()
        return os.path.join(base, "hdspikes", f"shd_{self.split}.h5")

    def iter_samples(self, target_classes=None, max_samples_per_class=None):
        yield from _iter_hdf5_samples(
            self._h5_path(),
            self.n_classes,
            time_scale_to_ms=1000.0,
            target_classes=target_classes,
            max_samples_per_class=max_samples_per_class,
        )


# --------------------------------------------------------------------------- #
# SSC                                                                          #
# --------------------------------------------------------------------------- #
class SSCSource:
    """Spiking Speech Commands — HDF5, times in seconds, local files. No TF."""

    n_units_in = 700
    n_classes = 35
    default_max_duration_ms = 1000.0
    _FILE_MAP = {
        "train": "ssc_train.h5.gz",
        "valid": "ssc_valid.h5.gz",
        "test": "ssc_test.h5.gz",
    }

    def __init__(self, split: str, data_path: Optional[str] = None):
        if split not in self._FILE_MAP:
            raise ValueError(
                f"Unknown SSC split: {split!r} (expected {list(self._FILE_MAP)})"
            )
        self.split = split
        self.data_path = data_path or _default_cache_dir()

    def _h5_path(self) -> str:
        gz_path = os.path.join(self.data_path, self._FILE_MAP[self.split])
        h5_path = gz_path[:-3] if gz_path.endswith(".gz") else gz_path
        # Serve the decompressed .h5 unless the .gz is newer (re-decompress on
        # staleness), matching SHDSource's _download_and_gunzip guard.
        if os.path.isfile(h5_path) and not (
            os.path.isfile(gz_path)
            and os.path.getctime(gz_path) > os.path.getctime(h5_path)
        ):
            return h5_path
        if not os.path.isfile(gz_path):
            raise FileNotFoundError(f"SSC file not found: {gz_path}")
        print(f"Decompressing {gz_path}", flush=True)
        with gzip.open(gz_path, "rb") as fin, open(h5_path, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        return h5_path

    def iter_samples(self, target_classes=None, max_samples_per_class=None):
        yield from _iter_hdf5_samples(
            self._h5_path(),
            self.n_classes,
            time_scale_to_ms=1000.0,
            target_classes=target_classes,
            max_samples_per_class=max_samples_per_class,
        )


# --------------------------------------------------------------------------- #
# NMNIST                                                                       #
# --------------------------------------------------------------------------- #
_NMNIST_N_X = 34
_NMNIST_N_Y = 34
_NMNIST_N_POL = 2
_NMNIST_TIME_MAX = 336040  # raw timestamp range in the AER files (23-bit field)


def decode_nmnist_events(
    raw: bytes, duration_ms: float = 300.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Decode one NMNIST `.bin` (5-byte-column AER) into flat (units, times_ms).

    Columns per event: [x, y, (pol<<7)|t[22:16], t[15:8], t[7:0]]. The 23-bit
    timestamp is rescaled from its native range into `[0, duration_ms]`. `units`
    is the flattened channel `pol*n_x*n_y + y*n_x + x` (matches legacy
    ``data/nmnist.py``).
    """
    byte_array = np.frombuffer(raw, dtype=np.uint8)
    if byte_array.size % 5 != 0:
        byte_array = byte_array[: byte_array.size - (byte_array.size % 5)]
    cols = [byte_array[c::5].astype(np.int64) for c in range(5)]
    x_coords, y_coords = cols[0], cols[1]
    polarities = cols[2] >> 7
    mask_23_bit = 0x7FFFFF  # 2**23 - 1: strips the polarity bit, keeps t[22:0]
    times = ((cols[2] << 16) | (cols[3] << 8) | cols[4]) & mask_23_bit
    times_ms = np.around(times * duration_ms / _NMNIST_TIME_MAX).astype(np.float64)
    units = (
        polarities * _NMNIST_N_X * _NMNIST_N_Y
        + y_coords * _NMNIST_N_X
        + x_coords
    )
    return units.astype(np.int64), times_ms


class NMNISTSource:
    """Neuromorphic-MNIST — per-file AER binary, `<data_path>/<Split>/<label>/*.bin`."""

    n_units_in = _NMNIST_N_X * _NMNIST_N_Y * _NMNIST_N_POL  # 2312
    n_classes = 10
    default_max_duration_ms = 300.0

    def __init__(self, split: str, data_path: str, duration_ms: float = 300.0):
        if data_path is None:
            raise ValueError("NMNISTSource requires an explicit data_path")
        self.split = split
        self.data_path = data_path
        self.duration_ms = float(duration_ms)

    def iter_samples(self, target_classes=None, max_samples_per_class=None):
        if target_classes is None:
            target_classes = list(range(self.n_classes))
        split_dir = os.path.join(self.data_path, self.split.capitalize())
        for label in target_classes:
            label_dir = os.path.join(split_dir, str(label))
            if not os.path.isdir(label_dir):
                print(f"Warning: NMNIST class {label} dir not found: {label_dir}")
                continue
            files = sorted(os.listdir(label_dir))
            if max_samples_per_class is not None:
                files = files[:max_samples_per_class]
            for fname in files:
                with open(os.path.join(label_dir, fname), "rb") as f:
                    raw = f.read()
                units, times_ms = decode_nmnist_events(raw, self.duration_ms)
                yield units, times_ms, int(label)


SOURCES = {"shd": SHDSource, "ssc": SSCSource, "nmnist": NMNISTSource}
