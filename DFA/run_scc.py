#!/usr/bin/env python3
"""DFA network on SSC (count-bin preprocessing).

SSC counterpart of ``DFA/run_shd.py``: same DFA network and shared
``trainer.train_model`` loop, but loads SSC (35 classes, train/valid/test) via
``data.ssc_binned``. SSC ships a dedicated ``valid`` split, so there is no
speaker holdout — pass ``--eval_split valid`` for model selection during
development and ``--eval_split test`` (default) for the final held-out number.
"""
import argparse
import os
import sys
import time

import jax


def _precision_from_argv(argv):
    default = "64"
    for i, arg in enumerate(argv):
        if arg.startswith("--precision="):
            return arg.split("=", 1)[1]
        if arg == "--precision" and i + 1 < len(argv):
            return argv[i + 1]
    return default


_PRECISION = _precision_from_argv(sys.argv[1:])
if _PRECISION not in ("32", "64"):
    raise ValueError(f"Invalid --precision '{_PRECISION}'. Expected '32' or '64'.")
jax.config.update("jax_enable_x64", _PRECISION == "64")
import jax.numpy as jnp
from jax import random
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from data.ssc_binned import load_ssc_binned, apply_channel_shift
from config import NeuronConfig
from network import Network
from trainer import evaluate, train_model


def augment_sample(x, args):
    """Apply enabled training-time augmentations to one sample (training only)."""
    if args.augment_channel_shift:
        x = apply_channel_shift(x, args.channel_shift_range)
    return x


def parse_args():
    p = argparse.ArgumentParser(description="DFA model on SSC (count-bin preprocessing)")
    p.add_argument("--bin_size_ms", type=float, default=4.0,
                   help="Time bin width in ms (paper default 4.0; also try 10, 14).")
    p.add_argument("--collapse_factor", type=int, default=5,
                   help="Sum-pool every N consecutive input channels (paper default 5: 700 -> 140).")
    p.add_argument("--max_duration_ms", type=float, default=1000.0,
                   help="Fixed window length in ms (SSC clips are 1 s; default 1000).")
    p.add_argument("--eval_split", choices=["valid", "test"], default="test",
                   help="SSC evaluation split. 'valid' for tuning/model selection, "
                        "'test' for the final held-out number (default test).")
    p.add_argument("--data_path", type=str, default="",
                   help="Directory holding ssc_{train,valid,test}.h5.gz "
                        "(default: auto-detect / download).")
    p.add_argument("--train_samples_per_class", type=int, default=None)
    p.add_argument("--eval_samples_per_class", type=int, default=None)
    p.add_argument("--hidden", type=int, nargs="+", default=[64],
                   help="Hidden-layer sizes, one per layer.")
    p.add_argument("--feedback_scale", type=float, default=0.1,
                   help="Std of the fixed random DFA feedback matrices B_l "
                        "(J x N_l), sampled once at init. Tunable.")
    p.add_argument("--n_outputs", type=int, default=35)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_clip", type=float, default=5.0)
    p.add_argument("--loss_temperature", type=float, default=2.7)
    p.add_argument("--loss_count_bias", type=float, default=0.18)
    p.add_argument("--loss_label_smoothing", type=float, default=0.13)
    p.add_argument("--beta_s", type=float, default=1.0)
    p.add_argument("--beta_d", type=float, default=1.5)
    p.add_argument("--weight_scale", type=float, default=0.25)
    p.add_argument("--tau_soma", type=float, default=15.0)
    p.add_argument("--tau_dend", type=float, default=15.0)
    p.add_argument("--tau_m", type=float, default=20.0)
    p.add_argument("--tau_plat_min", type=float, default=100.0)
    p.add_argument("--tau_plat_max", type=float, default=350.0)
    p.add_argument("--mu_th", type=float, default=1.0)
    p.add_argument("--v_th", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--tau_w", type=float, default=100.0)
    p.add_argument("--a_adapt", type=float, default=0.0)
    p.add_argument("--b_adapt", type=float, default=0.0)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--augment_channel_shift", action="store_true",
                   help="Enable channel-shift augmentation on training inputs only.")
    p.add_argument("--channel_shift_range", type=int, default=5,
                   help="Channel-shift range in channels (uniform in [-range, +range]).")
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--adam_eps", type=float, default=1e-8)
    p.add_argument("--lr_patience", type=int, default=5)
    p.add_argument("--lr_factor", type=float, default=0.7)
    p.add_argument("--lr_min", type=float, default=1e-6)
    p.add_argument("--early_stop_patience", type=int, default=0)
    p.add_argument("--precision", choices=["32", "64"], default=_PRECISION)
    p.add_argument("--save_model", type=str, default="")
    p.add_argument("--no_save_model", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.precision != _PRECISION:
        raise ValueError(
            f"--precision mismatch during startup ({args.precision} vs {_PRECISION}). "
            "Pass --precision only once."
        )
    if args.channel_shift_range < 0:
        raise ValueError("--channel_shift_range must be >= 0")
    np.random.seed(args.seed)
    key = random.PRNGKey(args.seed)
    B = args.batch_size
    eval_name = args.eval_split

    print("Loading SSC data with count-bin preprocessing...", flush=True)
    dtype = np.float64 if args.precision == "64" else np.float32
    X_tr, y_tr, _, X_ev, y_ev, _ = load_ssc_binned(
        bin_size_ms=args.bin_size_ms,
        collapse_factor=args.collapse_factor,
        max_duration_ms=args.max_duration_ms,
        train_samples_per_class=args.train_samples_per_class,
        eval_samples_per_class=args.eval_samples_per_class,
        data_path=args.data_path or None,
        eval_split=eval_name,
        binarize=False,
        dtype=dtype,
    )
    train_data = [(X_tr[i], int(y_tr[i])) for i in range(len(y_tr))]
    eval_data = [(X_ev[i], int(y_ev[i])) for i in range(len(y_ev))]
    T = train_data[0][0].shape[0]
    n_inputs = train_data[0][0].shape[1]
    print(
        f"Train: {len(train_data)}  {eval_name.capitalize()}: {len(eval_data)}  "
        f"n_inputs: {n_inputs}  T: {T}  batch_size: {B}  "
        f"precision=float{args.precision}  "
        f"bin={args.bin_size_ms}ms  collapse={args.collapse_factor}",
        flush=True,
    )

    config = NeuronConfig(
        dt=args.bin_size_ms,
        tau_soma=args.tau_soma, tau_dend=args.tau_dend, tau_m=args.tau_m,
        tau_plat_min=args.tau_plat_min, tau_plat_max=args.tau_plat_max,
        tau_w=args.tau_w, a_adapt=args.a_adapt, b_adapt=args.b_adapt,
        mu_th=args.mu_th, v_th=args.v_th, gamma=args.gamma,
        beta_s=args.beta_s, beta_d=args.beta_d, weight_scale=args.weight_scale,
        loss_temperature=args.loss_temperature, loss_count_bias=args.loss_count_bias,
        loss_label_smoothing=args.loss_label_smoothing,
    )
    net = Network(
        key, n_inputs, args.hidden, args.n_outputs, config,
        optimizer=args.optimizer, beta1=args.beta1, beta2=args.beta2, adam_eps=args.adam_eps,
        dropout_rate=args.dropout, weight_decay=args.weight_decay,
        feedback_scale=args.feedback_scale,
    )
    opt_str = f"adam(β1={args.beta1},β2={args.beta2})" if args.optimizer == "adam" else "sgd"
    hidden_str = " -> ".join(f"{n} (2-comp)" for n in args.hidden)
    arch_str = f"{n_inputs} -> {hidden_str} -> {args.n_outputs} (LI readout)"
    print(
        f"Network: {arch_str}  DFA(feedback_scale={args.feedback_scale})  "
        f"optimizer={opt_str}  lr={args.lr}",
        flush=True,
    )

    pre_acc = evaluate(net, eval_data, B)
    print(f"Pre-training {eval_name} accuracy: {pre_acc:.2f}%", flush=True)

    dev = jax.local_devices()[0]
    if hasattr(dev, "memory_stats") and dev.memory_stats() is not None:
        ms = dev.memory_stats()
        print(
            f"GPU memory: {ms['bytes_in_use']/1e6:.1f} MB in use, "
            f"{ms['peak_bytes_in_use']/1e6:.1f} MB peak, "
            f"{ms['bytes_limit']/1e6:.1f} MB pool",
            flush=True,
        )
    else:
        print(f"Device: {dev} (no memory stats available)", flush=True)

    result = train_model(
        net, train_data, eval_data,
        val_data=None,
        epochs=args.epochs, lr=args.lr, batch_size=B,
        gradient_clip=args.gradient_clip,
        lr_patience=args.lr_patience, lr_factor=args.lr_factor, lr_min=args.lr_min,
        early_stop_patience=args.early_stop_patience,
        augment_fn=(lambda x: augment_sample(x, args)),
        log=True,
    )
    print(
        f"\nBest {eval_name} accuracy: {result['best_metric']:.2f}% "
        f"(epoch {result['best_epoch']})  |  final {eval_name}={result['final_test']:.2f}%",
        flush=True,
    )

    if not args.no_save_model:
        if args.save_model:
            model_path = args.save_model
        else:
            ts = time.strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(_SCRIPT_DIR, "models", f"ssc_seed{args.seed}_{ts}.npz")
        os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
        net.save(model_path, extra={
            "best_metric": float(result["best_metric"]),
            "final_test": float(result["final_test"]),
            "best_epoch": int(result["best_epoch"]),
            "seed": int(args.seed),
            "eval_split": eval_name,
            "args": vars(args),
        })
        print(f"Saved best-checkpoint model -> {model_path}", flush=True)


if __name__ == "__main__":
    main()
