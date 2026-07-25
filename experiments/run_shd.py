"""Thin SHD training runner -> one self-describing run dir.

    python experiments/run_shd.py --rule bptt --epochs 30 --run-dir runs/shd_bptt
    python experiments/run_shd.py --synthetic 64 --epochs 2 --run-dir runs/smoke  # no download

``--synthetic N`` fabricates N random count samples so the whole pipeline
(build -> train -> write run dir) runs without any dataset download; omit it to load
real SHD via the cached dataset layer. All the real work lives in the library
(``build_network`` / ``train`` / ``io.write_run``); this file only wires flags to it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jax import random  # noqa: E402

from dendroprop.config import NeuronConfig, RunConfig  # noqa: E402
from dendroprop.network import build_network  # noqa: E402
from dendroprop.train import train  # noqa: E402
from dendroprop import io  # noqa: E402
from dendroprop.data import Augment, default_params, load_cached  # noqa: E402
from dendroprop.data.dataset import CachedSpikeDataset  # noqa: E402

# --rule presets -> build_network(method, **kwargs). credit_method only routes
# bptt/dfa/rfa; the pruned-dendrite and STE-surrogate components are separate
# build_network kwargs, so the composite presets map onto (bptt, kwargs).
_RULE_PRESETS = {
    "bptt": ("bptt", {}),
    "dfa": ("dfa", {}),
    "rfa": ("rfa", {}),
    "bptt_pruned": ("bptt", {"dendritic_credit": False}),
    "ste": ("bptt", {"surrogate": "ste"}),
}


def _synthetic_dataset(n, T, C, n_classes, seed):
    """A tiny in-memory SpikeDataset of random counts (for a no-download smoke run)."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 3, size=(n, T, C), dtype=np.uint8)
    y = rng.integers(0, n_classes, size=n).astype(np.int64)
    lengths = np.full(n, T, dtype=np.int64)
    meta = {"n_inputs": C, "n_classes": n_classes, "T": T, "cache_key": f"synthetic-{seed}"}
    return CachedSpikeDataset(X, y, lengths, meta)


def build_datasets(args):
    if args.synthetic:
        C = (700 + args.collapse_factor - 1) // args.collapse_factor
        T = max(1, int(np.ceil(args.max_duration_ms / args.bin_size_ms)))
        train_ds = _synthetic_dataset(args.synthetic, T, C, 20, args.seed)
        test_ds = _synthetic_dataset(max(4, args.synthetic // 4), T, C, 20, args.seed + 1)
        return train_ds, test_ds

    params = default_params("shd")
    params.update(
        bin_size_ms=args.bin_size_ms,
        collapse_factor=args.collapse_factor,
        max_duration_ms=args.max_duration_ms,
        max_samples_per_class=args.train_per_class,
    )
    aug = Augment(channel_shift=args.channel_shift, spike_dropout=args.spike_dropout)
    # Seed the augment rng from --seed so augmented runs are reproducible.
    train_ds = load_cached("shd", "train", params, data_path=args.data_path,
                           augment=aug, rng=np.random.default_rng([args.seed]))
    test_params = dict(params, max_samples_per_class=args.test_per_class)
    test_ds = load_cached("shd", "test", test_params, data_path=args.data_path)
    return train_ds, test_ds


def main(argv=None):
    p = argparse.ArgumentParser(description="Train the two-compartment SNN on SHD.")
    p.add_argument("--rule", default="bptt", choices=list(_RULE_PRESETS),
                   help="preset: bptt | dfa | rfa | bptt_pruned (dendrite off) | ste (surrogate)")
    p.add_argument("--hidden", default="128",
                   help="comma-separated hidden dims, bottom-up (e.g. 128 or 256,128)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--optimizer", default="adam", choices=["adam", "sgd"])
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    # dataset options
    p.add_argument("--bin-size-ms", type=float, default=4.0)
    p.add_argument("--collapse-factor", type=int, default=5)
    p.add_argument("--max-duration-ms", type=float, default=1400.0)
    p.add_argument("--channel-shift", type=int, default=0)
    p.add_argument("--spike-dropout", type=float, default=0.0)
    p.add_argument("--train-per-class", type=int, default=None)
    p.add_argument("--test-per-class", type=int, default=None)
    p.add_argument("--data-path", default=None, help="dir holding hdspikes/*.h5(.gz)")
    p.add_argument("--synthetic", type=int, default=0,
                   help="N>0: fabricate N random samples (no download) for a smoke run")
    p.add_argument("--run-dir", default="runs/shd", help="output run directory")
    args = p.parse_args(argv)

    hidden_dims = [int(h) for h in str(args.hidden).split(",") if h != ""]
    config = NeuronConfig(dt=args.bin_size_ms)  # dt tracks the bin width
    run = RunConfig(
        optimizer=args.optimizer, lr=args.lr, clip_value=args.clip,
        weight_decay=args.weight_decay, dropout_rate=args.dropout,
    )

    train_ds, test_ds = build_datasets(args)
    dims = (train_ds.n_inputs, hidden_dims, train_ds.n_classes)
    method, build_kwargs = _RULE_PRESETS[args.rule]
    net = build_network(random.PRNGKey(args.seed), dims, config, method, **build_kwargs)

    net, history = train(
        net, train_ds, test_ds, run,
        epochs=args.epochs, batch_size=args.batch_size, seed=args.seed,
    )
    run_dir = io.write_run(args.run_dir, run, history, net)

    last = history.last
    print(f"[run_shd] wrote {run_dir} | epochs={args.epochs} rule={args.rule} "
          f"test_acc={last.test_acc:.3f} loss={last.loss:.4f}", flush=True)
    return run_dir


if __name__ == "__main__":
    main()
