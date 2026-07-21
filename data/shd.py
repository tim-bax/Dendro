"""
SHD (Spiking Heidelberg Digits) data loading and input creation.
Extracted from 1layer/2comp_uniform.py for use by 1layer_version and 2layer_version.
No JAX dependency; uses NumPy only for create_shd_input_jax.
"""
import os
import time
import gzip
import shutil
import urllib.request
from typing import List, Tuple

import numpy as np
from tensorflow.keras.utils import get_file

# Try tables, fallback to h5py
USE_H5PY = False
try:
    import tables
except (ImportError, ValueError):
    USE_H5PY = True
try:
    import h5py
except ImportError:
    if USE_H5PY:
        raise ImportError("h5py is required when tables is not available. pip install h5py")
    h5py = None


def _alpha_kernel_np(t_vals: np.ndarray, tau: float) -> np.ndarray:
    """Alpha kernel (NumPy). Same as JAX version for create_*_input_jax."""
    k = (t_vals / tau) * np.exp(-t_vals / tau)
    return np.where(t_vals < 0, 0.0, k)


class SHDDataLoader:
    def __init__(self, data_path: str, duration_ms: int = 1400):
        self.data_path = data_path
        self.duration_ms = duration_ms
        self.n_units = 700

    def _get_and_gunzip(self, origin: str, filename: str, md5hash: str = None) -> str:
        if "SHD_CACHE_DIR" in os.environ:
            cache_dir = os.environ["SHD_CACHE_DIR"]
        elif os.path.exists("/share/neurocomputation/Tim/SHD_data"):
            cache_dir = "/share/neurocomputation/Tim/SHD_data"
        elif "SCRATCH" in os.environ:
            cache_dir = os.path.join(os.environ["SCRATCH"], "data")
        elif "TMPDIR" in os.environ and os.environ.get("TMPDIR") != "/tmp":
            cache_dir = os.path.join(os.environ["TMPDIR"], "data")
        elif os.path.exists("/scratch"):
            cache_dir = "/scratch/data"
        else:
            cache_dir = os.path.expanduser("~/data")

        os.makedirs(cache_dir, exist_ok=True)
        cache_subdir = "hdspikes"
        gz_file_path = get_file(filename, origin, md5_hash=md5hash, cache_dir=cache_dir, cache_subdir=cache_subdir)
        hdf5_file_path = gz_file_path[:-3]
        if not os.path.isfile(hdf5_file_path) or (
            os.path.exists(gz_file_path) and os.path.getctime(gz_file_path) > os.path.getctime(hdf5_file_path)
        ):
            print(f"Decompressing {gz_file_path}", flush=True)
            with gzip.open(gz_file_path, "rb") as f_in, open(hdf5_file_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        return hdf5_file_path

    def _load_sample_from_hdf5(self, units: np.ndarray, times: np.ndarray) -> List[np.ndarray]:
        times_ms = times * 1000.0
        times_clipped = np.clip(times_ms, 0, self.duration_ms - 1)
        times_int = np.around(times_clipped).astype(int)
        spike_data = [[] for _ in range(self.n_units)]
        for unit_id, spike_time in zip(units, times_int):
            if 0 <= unit_id < self.n_units:
                spike_data[unit_id].append(spike_time)
        return [np.array(spikes) for spikes in spike_data]

    def get_dataset(
        self,
        split: str = "train",
        max_samples_per_class: int = None,
        target_classes: List[int] = None,
    ) -> Tuple[List[List[np.ndarray]], List[int]]:
        if target_classes is None:
            target_classes = list(range(20))
        if split == "train":
            filename = "shd_train.h5.gz"
        elif split == "test":
            filename = "shd_test.h5.gz"
        else:
            raise ValueError(f"Unknown split: {split}. Must be 'train' or 'test'")
        base_url = "https://zenkelab.org/datasets"
        origin = f"{base_url}/{filename}"
        md5hash = None
        try:
            response = urllib.request.urlopen(f"{base_url}/md5sums.txt")
            lines = response.read().decode("utf-8").split("\n")
            file_hashes = {line.split()[1]: line.split()[0] for line in lines if len(line.split()) == 2}
            md5hash = file_hashes.get(filename)
        except Exception:
            pass
        hdf5_file_path = self._get_and_gunzip(origin, filename, md5hash)
        print(f"Loading {split} dataset from {hdf5_file_path}...", flush=True)
        print(f"Target classes: {target_classes}", flush=True)

        if USE_H5PY:
            fileh = h5py.File(hdf5_file_path, mode="r")
            units = fileh["spikes"]["units"]
            times = fileh["spikes"]["times"]
            labels = fileh["labels"]
        else:
            fileh = tables.open_file(hdf5_file_path, mode="r")
            units = fileh.root.spikes.units
            times = fileh.root.spikes.times
            labels = fileh.root.labels

        images = []
        labels_list = []
        class_counts = {label: 0 for label in target_classes}
        load_start_time = time.time()
        n_samples = len(labels) if not USE_H5PY else labels.shape[0]
        for idx in range(n_samples):
            label = int(labels[idx])
            if label not in target_classes:
                continue
            if max_samples_per_class is not None and class_counts[label] >= max_samples_per_class:
                continue
            if USE_H5PY:
                sample_units = units[idx][:]
                sample_times = times[idx][:]
            else:
                sample_units = units[idx]
                sample_times = times[idx]
            spike_data = self._load_sample_from_hdf5(sample_units, sample_times)
            images.append(spike_data)
            labels_list.append(label)
            class_counts[label] += 1
            if (len(images) % 100 == 0) and len(images) > 0:
                elapsed = time.time() - load_start_time
                rate = len(images) / elapsed if elapsed > 0 else 0
                remaining = (n_samples - idx) / rate if rate > 0 else 0
                print(f"  Loaded {len(images)} samples | Rate: {rate:.1f} samples/s | ETA: {remaining:.0f}s", flush=True)
        fileh.close()
        load_elapsed = time.time() - load_start_time
        print(
            f"Loaded {len(images)} total samples from {split} split in {load_elapsed:.1f}s ({len(images)/load_elapsed:.1f} samples/s)",
            flush=True,
        )
        print(f"Class distribution: {dict(class_counts)}", flush=True)
        return images, labels_list


def create_shd_input_jax(
    shd_data: List[np.ndarray],
    T: int = 1400,
    tau_alpha: float = 3.3,
    spike_amplitude: float = 5.0,
    use_kernel: bool = True,
) -> np.ndarray:
    """Create SHD input array (T, n_units). NumPy implementation, no JAX."""
    n_units = len(shd_data)
    x_input = np.zeros((T, n_units))
    if use_kernel:
        kernel_len = int(10 * tau_alpha)
        t_vals = np.arange(kernel_len, dtype=np.float64)
        k = _alpha_kernel_np(t_vals, tau_alpha)
        peak_value = np.exp(-1)
        k_normalized = k * (spike_amplitude / peak_value)
        for unit_idx, spike_times in enumerate(shd_data):
            for spike_time in spike_times:
                spike_time_int = int(spike_time)
                if 0 <= spike_time_int < T:
                    kernel_start = spike_time_int
                    kernel_end = min(kernel_start + kernel_len, T)
                    kernel_length_used = kernel_end - kernel_start
                    if kernel_length_used > 0:
                        x_input[kernel_start:kernel_end, unit_idx] += k_normalized[:kernel_length_used]
    else:
        for unit_idx, spike_times in enumerate(shd_data):
            for spike_time in spike_times:
                if 0 <= spike_time < T:
                    x_input[int(spike_time), unit_idx] = spike_amplitude
    return x_input


def apply_spike_dropout(x: np.ndarray, p_drop: float) -> np.ndarray:
    """Zero out a fraction of non-zero input bins (spike dropout). Use at train time only.
    x: (T, n_inputs) binned input. p_drop: probability of dropping each non-zero (0 = no dropout).
    Returns a copy with some non-zero entries set to 0. Uses np.random so each call is different."""
    if p_drop <= 0:
        return x.copy()
    out = np.asarray(x, dtype=np.float64).copy()
    nonzero = out != 0
    drop = np.random.random(out.shape) < p_drop
    out[nonzero & drop] = 0
    return out


def load_shd_data(
    data_path: str = None,
    train_samples_per_class: int = None,
    test_samples_per_class: int = None,
    target_classes: List[int] = None,
):
    """Load SHD train and test data. data_path None → auto-detect."""
    if target_classes is None:
        target_classes = list(range(20))
    if data_path is None:
        if "SHD_DATA_PATH" in os.environ:
            data_path = os.environ["SHD_DATA_PATH"]
        elif os.path.exists("/share/neurocomputation/Tim/SHD_data"):
            data_path = "/share/neurocomputation/Tim/SHD_data"
        else:
            data_path = os.path.expanduser("~/data/hdspikes")
    data_loader = SHDDataLoader(data_path)
    train_images, train_labels = data_loader.get_dataset(
        "train", max_samples_per_class=train_samples_per_class, target_classes=target_classes
    )
    test_images, test_labels = data_loader.get_dataset(
        "test", max_samples_per_class=test_samples_per_class, target_classes=target_classes
    )
    train_data = [(img, label) for img, label in zip(train_images, train_labels)]
    test_data = [(img, label) for img, label in zip(test_images, test_labels)]
    return train_data, test_data
