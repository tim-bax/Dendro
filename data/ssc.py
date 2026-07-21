"""
SSC (Spiking Speech Commands) data loading and input creation.
Uses the same input representation style as SHD.
"""
import gzip
import os
import shutil
from typing import List, Tuple

import numpy as np

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


class SSCDataLoader:
    def __init__(self, data_path: str, duration_ms: int = 1000):
        self.data_path = data_path
        self.duration_ms = duration_ms
        self.n_units = 700
        self.file_map = {
            "train": "ssc_train.h5.gz",
            "valid": "ssc_valid.h5.gz",
            "test": "ssc_test.h5.gz",
        }

    def _resolve_h5_path(self, split: str) -> str:
        if split not in self.file_map:
            raise ValueError(f"Unknown split: {split}. Must be one of {list(self.file_map)}")

        gz_name = self.file_map[split]
        gz_path = os.path.join(self.data_path, gz_name)
        if not os.path.exists(gz_path):
            raise FileNotFoundError(f"SSC file not found: {gz_path}")

        h5_path = gz_path[:-3] if gz_path.endswith(".gz") else gz_path
        if not os.path.isfile(h5_path) or os.path.getctime(gz_path) > os.path.getctime(h5_path):
            print(f"Decompressing {gz_path}", flush=True)
            with gzip.open(gz_path, "rb") as f_in, open(h5_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        return h5_path

    def _load_sample_from_hdf5(self, units: np.ndarray, times: np.ndarray) -> List[np.ndarray]:
        # SSC stores times in seconds in [0, 1). Convert to milliseconds.
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
            target_classes = list(range(35))

        hdf5_file_path = self._resolve_h5_path(split)
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

        fileh.close()
        print(f"Loaded {len(images)} total samples from {split} split", flush=True)
        print(f"Class distribution: {dict(class_counts)}", flush=True)
        return images, labels_list


def create_ssc_input_jax(
    ssc_data: List[np.ndarray],
    T: int = 1000,
    tau_alpha: float = 3.3,
    spike_amplitude: float = 5.0,
    use_kernel: bool = True,
) -> np.ndarray:
    """Create SSC input array (T, n_units). NumPy implementation, no JAX."""
    n_units = len(ssc_data)
    x_input = np.zeros((T, n_units))
    if use_kernel:
        kernel_len = int(10 * tau_alpha)
        t_vals = np.arange(kernel_len, dtype=np.float64)
        k = _alpha_kernel_np(t_vals, tau_alpha)
        peak_value = np.exp(-1)
        k_normalized = k * (spike_amplitude / peak_value)
        for unit_idx, spike_times in enumerate(ssc_data):
            for spike_time in spike_times:
                spike_time_int = int(spike_time)
                if 0 <= spike_time_int < T:
                    kernel_start = spike_time_int
                    kernel_end = min(kernel_start + kernel_len, T)
                    kernel_length_used = kernel_end - kernel_start
                    if kernel_length_used > 0:
                        x_input[kernel_start:kernel_end, unit_idx] += k_normalized[:kernel_length_used]
    else:
        for unit_idx, spike_times in enumerate(ssc_data):
            for spike_time in spike_times:
                if 0 <= spike_time < T:
                    x_input[int(spike_time), unit_idx] = spike_amplitude
    return x_input


def load_ssc_data(
    data_path: str = None,
    train_samples_per_class: int = None,
    eval_samples_per_class: int = None,
    target_classes: List[int] = None,
    eval_split: str = "test",
):
    """Load SSC train and eval data. eval_split can be 'valid' or 'test'."""
    if target_classes is None:
        target_classes = list(range(35))
    if eval_split not in ("valid", "test"):
        raise ValueError("--eval_split must be 'valid' or 'test'")
    if data_path is None:
        if "SSC_DATA_PATH" in os.environ:
            data_path = os.environ["SSC_DATA_PATH"]
        elif os.path.exists("/home/staff/t/tibax/Downloads/SSC_data"):
            data_path = "/home/staff/t/tibax/Downloads/SSC_data"
        elif os.path.exists("/share/neurocomputation/Tim/SSC_data"):
            data_path = "/share/neurocomputation/Tim/SSC_data"
        elif os.path.exists("/Users/tbax/Documents/SSC"):
            data_path = "/Users/tbax/Documents/SSC"
        else:
            data_path = os.path.expanduser("~/Downloads/SSC_data")

    data_loader = SSCDataLoader(data_path)
    train_images, train_labels = data_loader.get_dataset(
        "train",
        max_samples_per_class=train_samples_per_class,
        target_classes=target_classes,
    )
    eval_images, eval_labels = data_loader.get_dataset(
        eval_split,
        max_samples_per_class=eval_samples_per_class,
        target_classes=target_classes,
    )
    train_data = [(img, label) for img, label in zip(train_images, train_labels)]
    eval_data = [(img, label) for img, label in zip(eval_images, eval_labels)]
    return train_data, eval_data
