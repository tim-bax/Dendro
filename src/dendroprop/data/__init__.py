"""Unified, cached, TensorFlow-free dataset layer.

One generalized count-bin preprocessor (:func:`bin_events`) + per-dataset source
adapters (SHD/SSC/NMNIST, NMNIST with a 2x2 spatial-pool path) + an uncompressed
`.npy` + memmap cache keyed by a params hash + a :class:`SpikeDataset` Protocol
with augment-on-read. This replaces the SHD-only, TF-carrying legacy loaders.

Typical use::

    from dendroprop.data import load_cached, default_params, Augment, iterate_batches

    params = default_params("shd")                 # paper defaults
    train = load_cached("shd", "train", params,    # builds the cache on first call
                        augment=Augment(channel_shift=8, spike_dropout=0.1))
    for X, y in iterate_batches(train, batch_size=128, shuffle=True):
        ...   # X: (B, T, C) float32, y: (B,) int64
"""
from .preprocess import (
    bin_events,
    consecutive_pool_map,
    nmnist_pool_map,
    nmnist_pool_n_channels,
)
from .dataset import (
    SpikeDataset,
    CachedSpikeDataset,
    Augment,
    channel_shift,
    spike_dropout,
    iterate_batches,
)
from .cache import (
    DEFAULT_PARAMS,
    default_params,
    default_cache_root,
    cache_key,
    build_cache,
    load_cached,
)
from .sources import (
    SOURCES,
    SHDSource,
    SSCSource,
    NMNISTSource,
    decode_nmnist_events,
)

__all__ = [
    # preprocessing
    "bin_events",
    "consecutive_pool_map",
    "nmnist_pool_map",
    "nmnist_pool_n_channels",
    # dataset / augmentation
    "SpikeDataset",
    "CachedSpikeDataset",
    "Augment",
    "channel_shift",
    "spike_dropout",
    "iterate_batches",
    # cache
    "DEFAULT_PARAMS",
    "default_params",
    "default_cache_root",
    "cache_key",
    "build_cache",
    "load_cached",
    # sources
    "SOURCES",
    "SHDSource",
    "SSCSource",
    "NMNISTSource",
    "decode_nmnist_events",
]
