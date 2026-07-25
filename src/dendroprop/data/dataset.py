"""SpikeDataset interface + the memmap-backed cached dataset with augment-on-read.

The cache holds only CLEAN `uint8` count tensors. Augmentation (channel shift,
spike dropout) is applied per-read in ``__getitem__`` on a fresh float copy, so a
single cached array serves every epoch/seed without re-preprocessing and without
ever storing augmented data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Tuple, runtime_checkable

import numpy as np


@runtime_checkable
class SpikeDataset(Protocol):
    """A dataset of dense spike-count tensors the network can consume directly."""

    n_inputs: int
    n_classes: int
    T: int

    def __len__(self) -> int: ...
    def __getitem__(self, i) -> Tuple[np.ndarray, int]: ...  # (T, C) float, label
    def cache_key(self) -> str: ...


# --------------------------------------------------------------------------- #
# Augmentation (numpy, rng-driven) — applied on read, never baked into cache.  #
# --------------------------------------------------------------------------- #
def channel_shift(x: np.ndarray, shift_range: int, rng: np.random.Generator) -> np.ndarray:
    """Shift all channels of one `(T, C)` sample by a uniform integer offset.

    Drop + zero-fill, no wrap, no clamp (channel-axis analogue of temporal jitter):
    one shift per call, `shift = rng.integers(-shift_range, shift_range+1)` (inclusive,
    may be 0). Positive shift s moves channel c -> c+s; the top s channels fall off,
    the bottom s become 0. Returns a new array (input untouched).
    """
    if shift_range <= 0:
        return np.asarray(x).copy()
    x_np = np.asarray(x)
    C = x_np.shape[1]
    shift = int(rng.integers(-shift_range, shift_range + 1))
    if shift == 0:
        return x_np.copy()
    out = np.zeros_like(x_np)
    if shift > 0:
        out[:, shift:] = x_np[:, : C - shift]
    else:
        k = -shift
        out[:, : C - k] = x_np[:, k:]
    return out


def spike_dropout(x: np.ndarray, p_drop: float, rng: np.random.Generator) -> np.ndarray:
    """Zero out a fraction `p_drop` of the NON-zero bins (train-time only).

    Returns a copy; only already-nonzero entries can be dropped.
    """
    if p_drop <= 0:
        return np.asarray(x).copy()
    out = np.asarray(x).copy()
    nonzero = out != 0
    drop = rng.random(out.shape) < p_drop
    out[nonzero & drop] = 0
    return out


@dataclass(frozen=True)
class Augment:
    """Read-time augmentation config. Defaults are a no-op (clean tensors)."""

    channel_shift: int = 0
    spike_dropout: float = 0.0

    def active(self) -> bool:
        return self.channel_shift > 0 or self.spike_dropout > 0

    def __call__(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.channel_shift > 0:
            x = channel_shift(x, self.channel_shift, rng)
        if self.spike_dropout > 0:
            x = spike_dropout(x, self.spike_dropout, rng)
        return x


class CachedSpikeDataset:
    """A :class:`SpikeDataset` backed by memmapped `.npy` count arrays.

    ``X`` is the read-only `uint8` memmap `(N, T, C)`; ``__getitem__`` copies one
    sample out, casts to `float32`, and applies augmentation (a no-op unless
    configured). ``y``/``lengths`` are small int64 arrays.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lengths: np.ndarray,
        meta: dict,
        augment: Optional[Augment] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self._X = X
        self._y = np.asarray(y, dtype=np.int64)
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.n_inputs = int(meta["n_inputs"])
        self.n_classes = int(meta["n_classes"])
        self.T = int(meta["T"])
        self._key = str(meta["cache_key"])
        self._augment = augment or Augment()
        self._rng = rng if rng is not None else np.random.default_rng()

    def __len__(self) -> int:
        return int(self._X.shape[0])

    def __getitem__(self, i) -> Tuple[np.ndarray, int]:
        # np.array (not asarray) forces a fresh writable copy regardless of the
        # backing dtype, so a caller mutation can never reach the read-only memmap.
        x = np.array(self._X[i], dtype=np.float32)
        if self._augment.active():
            x = self._augment(x, self._rng).astype(np.float32, copy=False)
        return x, int(self._y[i])

    def cache_key(self) -> str:
        return self._key


def iterate_batches(
    dataset: SpikeDataset,
    batch_size: int,
    *,
    shuffle: bool = False,
    rng: Optional[np.random.Generator] = None,
    drop_last: bool = False,
):
    """Yield `(X (B, T, C) float32, y (B,) int64)` batches from a SpikeDataset."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    n = len(dataset)
    order = np.arange(n)
    if shuffle:
        (rng if rng is not None else np.random.default_rng()).shuffle(order)
    for start in range(0, n, batch_size):
        idx = order[start : start + batch_size]
        if drop_last and idx.shape[0] < batch_size:
            break
        xs, ys = [], []
        for i in idx:
            x, label = dataset[int(i)]
            xs.append(x)
            ys.append(label)
        yield (
            np.stack(xs, axis=0).astype(np.float32, copy=False),
            np.asarray(ys, dtype=np.int64),
        )
