"""Generalized count-bin preprocessor (pure NumPy, no I/O).

Every dataset routes its raw events through :func:`bin_events`, which turns a
single sample's `(units, times_ms)` event list into a dense count tensor
`(T_bins, n_channels_out)` — the representation the two-compartment network
consumes. Spatial pooling is expressed once, as a per-input-channel gather
(`channel_map`), so the SHD/SSC "sum every N consecutive channels" pool and the
NMNIST "2x2 spatial" pool are the same operation with different maps.

This is the generalization of the legacy, SHD-only ``data/shd_binned.py``
(``bin_shd_sample`` + ``collapse_factor``). For the consecutive-pool path it is
float-cast bit-parity with that reference (for per-bin counts <= 255).
"""
from __future__ import annotations

import numpy as np


def bin_events(
    units,
    times_ms,
    *,
    T_bins: int,
    bin_size_ms: float,
    channel_map: np.ndarray,
    n_channels_out: int,
    max_duration_ms: float | None = None,
    binarize: bool = False,
    count_dtype=np.uint8,
) -> np.ndarray:
    """Bin one sample's raw events into a dense `(T_bins, n_channels_out)` count matrix.

    Args:
        units:          int array of input channel indices.
        times_ms:       float array of spike timestamps, in milliseconds.
        T_bins:         number of output time bins (fixed window; zero-padded).
        bin_size_ms:    time bin width, ms.
        channel_map:    int array of length `n_units_in`; `channel_map[u]` is the
                        output channel for input channel `u` (encodes the spatial pool).
        n_channels_out: number of output channels.
        max_duration_ms: events at `t >= max_duration_ms` are dropped (not squashed);
                        if None, only `t >= 0` is required (still clamped to `T_bins-1`).
        binarize:       clip per-cell counts to {0, 1} after binning.
        count_dtype:    output integer dtype (uint8 for the cache; counts are clipped
                        to its max before casting).

    Returns:
        `(T_bins, n_channels_out)` array of dtype `count_dtype`.
    """
    units = np.asarray(units, dtype=np.int64)
    times_ms = np.asarray(times_ms, dtype=np.float64)
    channel_map = np.asarray(channel_map, dtype=np.int64)

    # Time window. Keep >= 0 always; drop past the fixed window if one is set.
    if max_duration_ms is not None:
        keep = (times_ms >= 0.0) & (times_ms < float(max_duration_ms))
    else:
        keep = times_ms >= 0.0
    # Units must be in range BEFORE the channel_map gather.
    keep &= (units >= 0) & (units < channel_map.shape[0])
    times_ms = times_ms[keep]
    units = units[keep]

    t_idx = (times_ms / bin_size_ms).astype(np.int64)
    np.minimum(t_idx, T_bins - 1, out=t_idx)

    c_out = channel_map[units]
    valid = (c_out >= 0) & (c_out < n_channels_out) & (t_idx >= 0)
    t_idx = t_idx[valid]
    c_out = c_out[valid]

    # Accumulate as an exact integer 2D histogram. bincount over the linear index
    # is order-independent and equals np.add.at, but never wraps (uint8 add.at would
    # wrap mod 256).
    lin = t_idx * n_channels_out + c_out
    counts = np.bincount(lin, minlength=T_bins * n_channels_out).reshape(
        T_bins, n_channels_out
    )
    if binarize:
        counts = np.minimum(counts, 1)
    counts = np.minimum(counts, np.iinfo(count_dtype).max)
    return counts.astype(count_dtype)


def consecutive_pool_map(n_units_in: int, collapse_factor: int) -> np.ndarray:
    """Sum-pool every `collapse_factor` consecutive input channels into one output.

    The SHD/SSC spatial pool (e.g. 700 -> 140 with collapse_factor=5). Returns an
    int64 map of length `n_units_in`; `n_channels_out = int(map.max()) + 1
    = ceil(n_units_in / collapse_factor)`.
    """
    if collapse_factor < 1:
        raise ValueError("collapse_factor must be >= 1")
    if n_units_in < 1:
        raise ValueError("n_units_in must be >= 1")
    return np.arange(n_units_in, dtype=np.int64) // collapse_factor


def nmnist_pool_map(
    n_x: int = 34, n_y: int = 34, n_pol: int = 2, pool: int = 2
) -> np.ndarray:
    """2x2 (or NxN) spatial pool for NMNIST's (x, y, polarity) grid.

    Input channel index `= pol*n_x*n_y + y*n_x + x` (matches the legacy
    ``data/nmnist.py`` flattening). Output channel `= pol*(nx'*ny') +
    (y//pool)*nx' + (x//pool)`, with `nx' = ceil(n_x/pool)`. For 34x34x2, pool=2
    this collapses 2312 -> 578 channels. Returns an int64 map of length
    `n_pol*n_y*n_x`; `n_channels_out = n_pol * nx' * ny'`.
    """
    if pool < 1:
        raise ValueError("pool must be >= 1")
    nx_out = (n_x + pool - 1) // pool
    ny_out = (n_y + pool - 1) // pool
    n_in = n_pol * n_y * n_x
    idx = np.arange(n_in, dtype=np.int64)
    in_pol = idx // (n_x * n_y)
    rem = idx % (n_x * n_y)
    in_y = rem // n_x
    in_x = rem % n_x
    return in_pol * (nx_out * ny_out) + (in_y // pool) * nx_out + (in_x // pool)


def nmnist_pool_n_channels(
    n_x: int = 34, n_y: int = 34, n_pol: int = 2, pool: int = 2
) -> int:
    """Number of output channels for :func:`nmnist_pool_map`."""
    nx_out = (n_x + pool - 1) // pool
    ny_out = (n_y + pool - 1) // pool
    return int(n_pol * nx_out * ny_out)
