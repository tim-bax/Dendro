"""
NMNIST data loading and input creation.
Extracted from 1layer/2comp.py for use by 1layer_version and 2layer_version.
No JAX dependency; uses NumPy only for create_nmnist_input_jax.
"""
import os
from typing import List, Tuple

import numpy as np


def _alpha_kernel_np(t_vals: np.ndarray, tau: float) -> np.ndarray:
    """Alpha kernel (NumPy). Same as JAX version for create_*_input_jax."""
    k = (t_vals / tau) * np.exp(-t_vals / tau)
    return np.where(t_vals < 0, 0.0, k)


class NMNISTDataLoader:
    def __init__(self, data_path: str, duration_ms: int = 300):
        self.data_path = data_path
        self.duration_ms = duration_ms
        self.pixels_dict = {
            "n_x": 34,
            "n_y": 34,
            "n_polarity": 2,
        }
        self.pixels_dict["n_total"] = (
            self.pixels_dict["n_x"] * self.pixels_dict["n_y"] * self.pixels_dict["n_polarity"]
        )
        self.pixels_dict["active"] = list(range(self.pixels_dict["n_total"]))
        self.pixels_dict["n_active"] = len(self.pixels_dict["active"])

    def load_image(self, file_path: str) -> Tuple[List[np.ndarray], np.ndarray]:
        with open(file_path, "rb") as f:
            byte_array = np.asarray([x for x in f.read()])
        n_byte_columns = 5
        byte_columns = [byte_array[column::n_byte_columns] for column in range(n_byte_columns)]
        x_coords = byte_columns[0]
        y_coords = byte_columns[1]
        polarities = byte_columns[2] >> 7
        mask_22_bit = 0x7FFFFF
        times = (byte_columns[2] << 16 | byte_columns[3] << 8 | byte_columns[4]) & mask_22_bit
        time_max = 336040
        times = np.around(times * self.duration_ms / time_max)
        pixels = (
            polarities * self.pixels_dict["n_x"] * self.pixels_dict["n_y"]
            + y_coords * self.pixels_dict["n_x"]
            + x_coords
        )
        image = [times[pixels == pixel] for pixel in self.pixels_dict["active"]]
        return image, times

    def get_dataset(
        self,
        split: str = "train",
        max_samples_per_class: int = None,
        target_classes: List[int] = None,
    ) -> Tuple[List[List[np.ndarray]], List[int]]:
        if target_classes is None:
            target_classes = list(range(10))
        split_path = os.path.join(self.data_path, split.capitalize())
        images = []
        labels = []
        print(f"Loading {split} dataset from {split_path}...", flush=True)
        for label in target_classes:
            label_path = os.path.join(split_path, str(label))
            if not os.path.exists(label_path):
                print(f"Warning: Class {label} directory not found at {label_path}")
                continue
            files = sorted(os.listdir(label_path))
            if max_samples_per_class is not None:
                files = files[:max_samples_per_class]
            for file in files:
                file_path = os.path.join(label_path, file)
                image, _ = self.load_image(file_path)
                images.append(image)
                labels.append(label)
        print(f"Loaded {len(images)} total samples from {split} split", flush=True)
        return images, labels


def create_nmnist_input_jax(
    nmnist_data: List[np.ndarray],
    T: int = 300,
    tau_alpha: float = 3.3,
    spike_amplitude: float = 5.0,
    use_kernel: bool = True,
) -> np.ndarray:
    """Create NMNIST input array (T, n_pixels). NumPy implementation, no JAX."""
    n_pixels = len(nmnist_data)
    x_input = np.zeros((T, n_pixels))
    if use_kernel:
        kernel_len = int(10 * tau_alpha)
        t_vals = np.arange(kernel_len, dtype=np.float64)
        k = _alpha_kernel_np(t_vals, tau_alpha)
        peak_value = np.exp(-1)
        k_normalized = k * (spike_amplitude / peak_value)
        for pixel_idx, spike_times in enumerate(nmnist_data):
            for spike_time in spike_times:
                spike_time_int = int(spike_time)
                if 0 <= spike_time_int < T:
                    kernel_start = spike_time_int
                    kernel_end = min(kernel_start + kernel_len, T)
                    kernel_length_used = kernel_end - kernel_start
                    if kernel_length_used > 0:
                        x_input[kernel_start:kernel_end, pixel_idx] += k_normalized[:kernel_length_used]
    else:
        for pixel_idx, spike_times in enumerate(nmnist_data):
            for spike_time in spike_times:
                if 0 <= spike_time < T:
                    x_input[int(spike_time), pixel_idx] = spike_amplitude
    return x_input


def load_nmnist_data(
    data_path: str,
    train_samples_per_class: int = None,
    test_samples_per_class: int = None,
    target_classes: List[int] = None,
):
    """Load NMNIST train and test data."""
    if target_classes is None:
        target_classes = list(range(10))
    data_loader = NMNISTDataLoader(data_path)
    train_images, train_labels = data_loader.get_dataset(
        "train", max_samples_per_class=train_samples_per_class, target_classes=target_classes
    )
    test_images, test_labels = data_loader.get_dataset(
        "test", max_samples_per_class=test_samples_per_class, target_classes=target_classes
    )
    train_data = [(img, label) for img, label in zip(train_images, train_labels)]
    test_data = [(img, label) for img, label in zip(test_images, test_labels)]
    return train_data, test_data
