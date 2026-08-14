#!/usr/bin/env python3
"""Optuna hyperparameter search for the DFA network on SSC.

SSC counterpart of ``DFA/tune_shd.py``. SSC ships a dedicated ``valid`` split, so
selection uses it directly (no speaker holdout); the SSC ``test`` split is never
touched during the search. Optimizes the SAME twelve knobs as the SHD search:

  Shared 11:   lr, weight_scale, loss_temperature, loss_count_bias,
               loss_label_smoothing, gamma, beta_s, beta_d,
               channel_shift_range, tau_plat_min, tau_plat_max
  DFA-only:    feedback_scale (random-feedback magnitude, log-uniform)

    python DFA/tune_ssc.py --precision 32 --n_trials 40 --hidden 128 128

Retrain the winner with DFA/run_scc.py (which reports test) via the command
printed at the end.
"""
import argparse
import json
import os
import sys
import time

import jax


def _precision_from_argv(argv):
    for i, arg in enumerate(argv):
        if arg.startswith("--precision="):
            return arg.split("=", 1)[1]
        if arg == "--precision" and i + 1 < len(argv):
            return argv[i + 1]
    return "64"


_PRECISION = _precision_from_argv(sys.argv[1:])
if _PRECISION not in ("32", "64"):
    raise ValueError(f"Invalid --precision '{_PRECISION}'. Expected '32' or '64'.")
jax.config.update("jax_enable_x64", _PRECISION == "64")

import numpy as np
from jax import random
import optuna

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from data.ssc_binned import load_ssc_binned, apply_channel_shift
from config import NeuronConfig
from network import Network
from trainer import train_model


# ── Search space (shared 11 + DFA feedback_scale) ─────────────────────────────
LR_RANGE = (1e-4, 3e-3)              # log-uniform
WEIGHT_SCALE_RANGE = (0.1, 1.5)      # log-uniform (Xavier-std multiplier)
LOSS_TEMPERATURE_RANGE = (0.1, 3.0)  # uniform
LOSS_COUNT_BIAS_RANGE = (0.0, 0.5)   # uniform
LABEL_SMOOTHING_RANGE = (0.0, 0.3)   # uniform
GAMMA_RANGE = (0.5, 0.9)             # uniform
BETA_S_RANGE = (0.3, 4.0)            # uniform
BETA_D_RANGE = (0.3, 4.0)            # uniform
CHANNEL_SHIFT_RANGE = (0, 12)        # int; 0 = no augmentation
TAU_PLAT_MIN_RANGE = (50.0, 900.0)
TAU_PLAT_MAX_RANGE = (200.0, 900.0)
MIN_PLAT_GAP = 10.0
FEEDBACK_SCALE_RANGE = (0.02, 0.5)   # log-uniform (DFA-specific)


def build_pruner(args):
    """Light pruner: MedianPruner with a generous warmup, or NopPruner."""
    if args.pruner == "none":
        return optuna.pruners.NopPruner()
    return optuna.pruners.MedianPruner(
        n_startup_trials=args.prune_startup_trials,
        n_warmup_steps=args.prune_warmup_steps,
        interval_steps=args.prune_interval_steps,
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Optuna tuning for DFA on SSC (val-driven, light pruning)."
    )
    # ── Search control ──
    p.add_argument("--n_trials", type=int, default=40)
    p.add_argument("--epochs", type=int, default=60,
                   help="Per-trial epoch budget. Trials run to the full count "
                        "unless truncated by the (light) MedianPruner.")
    p.add_argument("--val_smooth_window", type=int, default=3)
    p.add_argument("--early_stop_patience", type=int, default=0,
                   help="Per-trial early stopping (0 = disabled; rely on pruning).")
    p.add_argument("--progress_every", type=int, default=5)
    p.add_argument("--storage", type=str, default="",
                   help="Optuna storage URL (e.g. sqlite:///dfa_ssc.db). "
                        "Empty = in-memory.")
    p.add_argument("--study_name", type=str, default="dfa_ssc_128")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="")
    # ── Pruning (light by default) ──
    p.add_argument("--pruner", choices=["median", "none"], default="median")
    p.add_argument("--prune_startup_trials", type=int, default=8)
    p.add_argument("--prune_warmup_steps", type=int, default=20)
    p.add_argument("--prune_interval_steps", type=int, default=1)
    # ── Architecture (fixed across the search) ──
    p.add_argument("--hidden", type=int, nargs="+", default=[128, 128])
    p.add_argument("--n_outputs", type=int, default=35)
    # ── Preprocessing (unified across variants: collapse 5 -> 140 inputs) ──
    p.add_argument("--bin_size_ms", type=float, default=4.0)
    p.add_argument("--collapse_factor", type=int, default=5)
    p.add_argument("--max_duration_ms", type=float, default=1000.0)
    p.add_argument("--data_path", type=str, default="",
                   help="Directory holding ssc_{train,valid,test}.h5.gz.")
    p.add_argument("--train_samples_per_class", type=int, default=None)
    p.add_argument("--eval_samples_per_class", type=int, default=None)
    # ── Frozen biophysics / recipe (fixed) ──
    p.add_argument("--tau_soma", type=float, default=15.0)
    p.add_argument("--tau_dend", type=float, default=15.0)
    p.add_argument("--tau_m", type=float, default=20.0)
    p.add_argument("--tau_w", type=float, default=100.0)
    p.add_argument("--a_adapt", type=float, default=0.0)
    p.add_argument("--b_adapt", type=float, default=0.0)
    p.add_argument("--mu_th", type=float, default=1.0)
    p.add_argument("--v_th", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.0,
                   help="Dropout rate (fixed across the search).")
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr_patience", type=int, default=10)
    # ── Training (fixed) ──
    p.add_argument("--gradient_clip", type=float, default=5.0)
    p.add_argument("--lr_factor", type=float, default=0.7)
    p.add_argument("--lr_min", type=float, default=1e-6)
    p.add_argument("--precision", choices=["32", "64"], default=_PRECISION)
    return p.parse_args()


def build_config(args, *, loss_temperature, loss_count_bias, loss_label_smoothing,
                 weight_scale, gamma, beta_s, beta_d, tau_plat_min, tau_plat_max):
    return NeuronConfig(
        dt=args.bin_size_ms,
        tau_soma=args.tau_soma, tau_dend=args.tau_dend, tau_m=args.tau_m,
        tau_plat_min=tau_plat_min, tau_plat_max=tau_plat_max,
        tau_w=args.tau_w, a_adapt=args.a_adapt, b_adapt=args.b_adapt,
        mu_th=args.mu_th, v_th=args.v_th, gamma=gamma,
        beta_s=beta_s, beta_d=beta_d, weight_scale=weight_scale,
        loss_temperature=loss_temperature, loss_count_bias=loss_count_bias,
        loss_label_smoothing=loss_label_smoothing,
    )


def main():
    args = parse_args()
    if args.precision != _PRECISION:
        raise ValueError(
            f"--precision mismatch ({args.precision} vs {_PRECISION}). Pass it once."
        )

    dtype = np.float64 if args.precision == "64" else np.float32
    print(f"Loading SSC once (precision=float{args.precision})...", flush=True)
    X_tr, y_tr, _, X_va, y_va, _ = load_ssc_binned(
        bin_size_ms=args.bin_size_ms,
        collapse_factor=args.collapse_factor,
        max_duration_ms=args.max_duration_ms,
        train_samples_per_class=args.train_samples_per_class,
        eval_samples_per_class=args.eval_samples_per_class,
        data_path=args.data_path or None,
        eval_split="valid",  # selection on the dedicated valid split; never test
        binarize=False, dtype=dtype,
    )
    train_data = [(X_tr[i], int(y_tr[i])) for i in range(len(y_tr))]
    val_data = [(X_va[i], int(y_va[i])) for i in range(len(y_va))]
    n_inputs = train_data[0][0].shape[1]
    T = train_data[0][0].shape[0]
    base_key = random.PRNGKey(args.seed)

    print(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}", flush=True)
    hidden_str = "-".join(str(h) for h in args.hidden)
    print(
        f"Search: {n_inputs} -> {hidden_str} -> {args.n_outputs}  T={T}  "
        f"(valid N={len(val_data)}, train N={len(train_data)})  "
        f"trials={args.n_trials}  epochs/trial={args.epochs}  "
        f"bs={args.batch_size}  pruner={args.pruner}",
        flush=True,
    )
    print(
        "Tuned: lr weight_scale loss_temperature loss_count_bias "
        "loss_label_smoothing gamma beta_s beta_d channel_shift_range "
        "tau_plat_min tau_plat_max feedback_scale",
        flush=True,
    )

    def objective(trial):
        np.random.seed(args.seed + trial.number)

        lr = trial.suggest_float("lr", *LR_RANGE, log=True)
        weight_scale = trial.suggest_float("weight_scale", *WEIGHT_SCALE_RANGE, log=True)
        loss_temperature = trial.suggest_float("loss_temperature", *LOSS_TEMPERATURE_RANGE)
        loss_count_bias = trial.suggest_float("loss_count_bias", *LOSS_COUNT_BIAS_RANGE)
        loss_label_smoothing = trial.suggest_float(
            "loss_label_smoothing", *LABEL_SMOOTHING_RANGE)
        gamma = trial.suggest_float("gamma", *GAMMA_RANGE)
        beta_s = trial.suggest_float("beta_s", *BETA_S_RANGE)
        beta_d = trial.suggest_float("beta_d", *BETA_D_RANGE)
        channel_shift_range = trial.suggest_int("channel_shift_range", *CHANNEL_SHIFT_RANGE)
        tau_plat_min = trial.suggest_float(
            "tau_plat_min", TAU_PLAT_MIN_RANGE[0], TAU_PLAT_MIN_RANGE[1] - MIN_PLAT_GAP)
        lo_max = min(max(TAU_PLAT_MAX_RANGE[0], tau_plat_min + MIN_PLAT_GAP),
                     TAU_PLAT_MAX_RANGE[1])
        tau_plat_max = trial.suggest_float("tau_plat_max", lo_max, TAU_PLAT_MAX_RANGE[1])
        feedback_scale = trial.suggest_float("feedback_scale", *FEEDBACK_SCALE_RANGE, log=True)

        print(
            f"[trial {trial.number:3d}] start: lr={lr:.2e} "
            f"weight_scale={weight_scale:.3f} temp={loss_temperature:.3f} "
            f"count_bias={loss_count_bias:.3f} smooth={loss_label_smoothing:.3f} "
            f"gamma={gamma:.3f} beta_s={beta_s:.3f} beta_d={beta_d:.3f} "
            f"chan_shift={channel_shift_range} "
            f"tau_plat=[{tau_plat_min:.1f}, {tau_plat_max:.1f}]ms "
            f"fb={feedback_scale:.3f}",
            flush=True,
        )

        config = build_config(
            args, loss_temperature=loss_temperature, loss_count_bias=loss_count_bias,
            loss_label_smoothing=loss_label_smoothing, weight_scale=weight_scale,
            gamma=gamma, beta_s=beta_s, beta_d=beta_d,
            tau_plat_min=tau_plat_min, tau_plat_max=tau_plat_max,
        )
        key = random.fold_in(base_key, trial.number)
        net = Network(
            key, n_inputs, args.hidden, args.n_outputs, config,
            optimizer="adam", dropout_rate=args.dropout,
            weight_decay=args.weight_decay, feedback_scale=feedback_scale,
        )

        def augment_fn(x):
            return apply_channel_shift(x, channel_shift_range)

        result = train_model(
            net, train_data, test_data=None, val_data=val_data,
            epochs=args.epochs, lr=lr, batch_size=args.batch_size,
            gradient_clip=args.gradient_clip,
            lr_patience=args.lr_patience, lr_factor=args.lr_factor, lr_min=args.lr_min,
            early_stop_patience=args.early_stop_patience,
            val_smooth_window=args.val_smooth_window,
            augment_fn=augment_fn, trial=trial, log=False,
            progress_every=args.progress_every,
        )
        return result["best_metric"]

    def _log_trial(study, trial):
        state = trial.state.name
        best = study.best_value if study.best_trial else float("nan")
        val = trial.value if trial.value is not None else float("nan")
        print(
            f"[trial {trial.number:3d}] {state:8s} val={val:6.2f}  "
            f"best={best:6.2f}  params={trial.params}",
            flush=True,
        )

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        direction="maximize", sampler=sampler, pruner=build_pruner(args),
        study_name=args.study_name,
        storage=(args.storage or None),
        load_if_exists=bool(args.storage),
    )
    study.optimize(objective, n_trials=args.n_trials, callbacks=[_log_trial])

    print("\n=== Best trial ===", flush=True)
    print(f"  smoothed val acc: {study.best_value:.2f}%", flush=True)
    print(f"  params: {json.dumps(study.best_params, indent=2)}", flush=True)

    out = args.out
    if not out:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out = os.path.join(_SCRIPT_DIR, "tune_results", f"best_{args.study_name}_{ts}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "best_value": study.best_value,
            "best_params": study.best_params,
            "n_trials": len(study.trials),
            "eval_split": "valid",
            "hidden": list(args.hidden),
            "fixed_args": vars(args),
        }, f, indent=2)
    print(f"Wrote best params -> {out}", flush=True)

    # Ready-to-run retrain command with the winner (reports test).
    bp = study.best_params
    cs = int(bp["channel_shift_range"])
    chan_shift_flag = (
        f"--augment_channel_shift --channel_shift_range {cs} " if cs > 0 else ""
    )
    print(
        "\nRetrain the winner, then report test:\n"
        f"  python DFA/run_scc.py --hidden {hidden_str.replace('-', ' ')} "
        f"--n_outputs {args.n_outputs} --optimizer adam --precision {args.precision} "
        f"--batch_size {args.batch_size} --epochs {args.epochs} --eval_split test "
        f"--bin_size_ms {args.bin_size_ms} --collapse_factor {args.collapse_factor} "
        f"--max_duration_ms {args.max_duration_ms} "
        f"--lr {bp['lr']:.6g} --weight_scale {bp['weight_scale']:.4g} "
        f"--loss_temperature {bp['loss_temperature']:.4g} "
        f"--loss_count_bias {bp['loss_count_bias']:.4g} "
        f"--loss_label_smoothing {bp['loss_label_smoothing']:.4g} "
        f"--gamma {bp['gamma']:.4g} --beta_s {bp['beta_s']:.4g} "
        f"--beta_d {bp['beta_d']:.4g} "
        f"--tau_plat_min {bp['tau_plat_min']:.4g} --tau_plat_max {bp['tau_plat_max']:.4g} "
        f"--feedback_scale {bp['feedback_scale']:.4g} "
        f"--mu_th {args.mu_th} --v_th {args.v_th} --tau_w {args.tau_w} "
        f"--a_adapt {args.a_adapt} --b_adapt {args.b_adapt} "
        f"--tau_soma {args.tau_soma} --tau_dend {args.tau_dend} --tau_m {args.tau_m} "
        f"--dropout {args.dropout} {chan_shift_flag}"
        f"--lr_patience {args.lr_patience} --lr_factor {args.lr_factor}",
        flush=True,
    )


if __name__ == "__main__":
    main()
