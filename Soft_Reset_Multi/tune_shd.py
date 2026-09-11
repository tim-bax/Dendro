#!/usr/bin/env python3
"""Optuna hyperparameter search for the Soft_Reset network on SHD.

Refit around a strong 512-hidden ROOF-surrogate recipe: the whole biophysics /
regularization recipe is FROZEN to that run's values (passed on the CLI) and only
SIX knobs are optimized on the macro-averaged (per-class) held-out-SPEAKER
VALIDATION accuracy:

  lr, loss_temperature, loss_label_smoothing, gamma,
  channel_shift_range, tau_plat_max

Everything else is frozen, notably: weight_scale, mu_th, beta_s, beta_s_dend,
beta_d, dropout, weight_decay, tau_plat_min, and ``--dend_surrogate_roof`` (the
flat-roof dendritic gradient — a DIFFERENT gradient path than the default bump, so
it must be enabled to match the target run). ``loss_count_bias`` is FROZEN to 0.0
(a scalar added uniformly to all logits, hence a softmax no-op), so it is not
searched.

SHD ships no validation split, so some train SPEAKERS are held out as val (the SHD
*test* set is never touched during the search). The default holdout is ``[9, 10]``
— a male (spk9, age 32) + female (spk10, age 25) pair chosen to mirror the SHD
*test* set's two novel speakers (spk4 male/31, spk5 female/27), so the val
speaker-generalization gap tracks test. Retrain the winner on ALL train speakers
with run_shd.py (no held-out speakers) before reporting test.

    python Soft_Reset/tune_shd.py --precision 32 --n_trials 200 --n_hidden 512 \
      --dend_surrogate_roof --storage sqlite:///Soft_Reset/soft_reset_shd_512.db

Notes (Soft_Reset is a distinct model):
  * single hidden layer -> ``--n_hidden`` is a scalar int (not a ``--hidden`` list);
  * no gradient clipping -> ``batch_train_step`` takes no ``clip_value``;
  * channel-shift augmentation is always applied (its range is a tuned dimension);
  * tau_plat_max is tuned, tau_plat_min is frozen.

Protocol note: TPE sampler + a light MedianPruner (generous warmup; only clearly
hopeless trials are stopped).
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

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

import jax.numpy as jnp
import numpy as np
from jax import random
import optuna

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from data.shd_binned import load_shd_binned, apply_channel_shift
from config import NeuronConfig
from network import Network


# ── Search space for the six tuned knobs ──────────────────────────────────────
LR_RANGE = (1e-4, 3e-3)              # log-uniform
LOSS_TEMPERATURE_RANGE = (0.1, 3.0)  # uniform
LABEL_SMOOTHING_RANGE = (0.0, 0.3)   # uniform
GAMMA_RANGE = (0.3, 0.9)             # uniform (plateau-induced threshold reduction)
CHANNEL_SHIFT_RANGE = (3, 15)        # int-uniform (channel-shift augmentation range)
TAU_PLAT_MAX_RANGE = (200.0, 700.0)  # uniform (ms; must stay above frozen tau_plat_min)


def split_by_speakers(data, speakers, val_ids):
    """Partition a train list into (train, val) by held-out speaker id."""
    speakers = np.asarray(speakers)
    val_set = set(int(s) for s in val_ids)
    present = set(int(s) for s in np.unique(speakers))
    missing = val_set - present
    if missing:
        raise ValueError(
            f"--val_speakers {sorted(missing)} not present in the train split "
            f"(train speakers: {sorted(present)})."
        )
    train_data, val_data = [], []
    for sample, spk in zip(data, speakers):
        (val_data if int(spk) in val_set else train_data).append(sample)
    train_classes = {int(y) for _, y in train_data}
    all_classes = {int(y) for _, y in data}
    dropped = all_classes - train_classes
    if dropped:
        print(
            f"WARNING: holding out speakers {sorted(val_set)} removes classes "
            f"{sorted(dropped)} from the train split.",
            flush=True,
        )
    return train_data, val_data


def build_pruner(args):
    """Light pruner: MedianPruner with a generous warmup so only clearly-hopeless
    trials are stopped, or NopPruner when --pruner none."""
    if args.pruner == "none":
        return optuna.pruners.NopPruner()
    return optuna.pruners.MedianPruner(
        n_startup_trials=args.prune_startup_trials,
        n_warmup_steps=args.prune_warmup_steps,
        interval_steps=args.prune_interval_steps,
    )


def evaluate(net, dataset, batch_size=64):
    """Macro-averaged (mean per-class) top-1 accuracy (%) over a dataset.

    Averages per-class recall over the classes present, so an imbalanced val set
    (the speaker holdout leans English) weights every digit equally and a weak
    class such as ``neun`` can't be masked by the majority. Batched with
    last-batch padding.
    """
    n = len(dataset)
    if n == 0:
        return 0.0
    correct = defaultdict(int)
    total = defaultdict(int)
    if batch_size <= 1:
        for x, y in dataset:
            yc = int(y)
            total[yc] += 1
            correct[yc] += int(net.predict(x) == yc)
    else:
        for start in range(0, n, batch_size):
            batch = dataset[start:start + batch_size]
            actual = len(batch)
            xs = [x for x, y in batch]
            ys = [int(y) for x, y in batch]
            if actual < batch_size:
                xs += [xs[0]] * (batch_size - actual)
            preds = np.asarray(net.batch_predict(jnp.stack(xs)))
            for i in range(actual):
                total[ys[i]] += 1
                correct[ys[i]] += int(int(preds[i]) == ys[i])
    return 100.0 * sum(correct[c] / total[c] for c in total) / len(total)


def parse_args():
    p = argparse.ArgumentParser(
        description="Optuna tuning for Soft_Reset on SHD (val-driven, light pruning)."
    )
    # ── Search control ──
    p.add_argument("--n_trials", type=int, default=40)
    p.add_argument("--epochs", type=int, default=60,
                   help="Per-trial epoch budget. Trials run to the full count "
                        "unless truncated by the (light) MedianPruner.")
    p.add_argument("--val_speakers", type=int, nargs="+", default=[9, 10],
                   help="Speaker ids held out of train as the val set (>=1). Default "
                        "[9, 10] (male spk9/32 + female spk10/25) mirrors the SHD test "
                        "set's novel speakers (spk4 male/31, spk5 female/27).")
    p.add_argument("--progress_every", type=int, default=5,
                   help="Print a per-trial heartbeat every N epochs (0 = off).")
    p.add_argument("--storage", type=str, default="",
                   help="Optuna storage URL (e.g. sqlite:///soft_reset_shd.db) "
                        "for a resumable study. Empty = in-memory.")
    p.add_argument("--study_name", type=str, default="soft_reset_shd_512")
    p.add_argument("--seed", type=int, default=12)
    p.add_argument("--out", type=str, default="",
                   help="Where to write best-params JSON. Empty = auto under "
                        "Soft_Reset/tune_results/.")
    # ── Pruning (light by default) ──
    p.add_argument("--pruner", choices=["median", "none"], default="median",
                   help="Light MedianPruner (default) or no pruning.")
    p.add_argument("--prune_startup_trials", type=int, default=8,
                   help="MedianPruner: no pruning until this many trials finish.")
    p.add_argument("--prune_warmup_steps", type=int, default=20,
                   help="MedianPruner: no pruning before this epoch within a trial.")
    p.add_argument("--prune_interval_steps", type=int, default=1)
    # ── Architecture (fixed across the search) ──
    p.add_argument("--n_hidden", type=int, default=512,
                   help="Single hidden-layer width (Soft_Reset is one layer).")
    p.add_argument("--n_outputs", type=int, default=20)
    # ── Preprocessing (unified across variants: collapse 5 -> 140 inputs) ──
    p.add_argument("--bin_size_ms", type=float, default=4.0)
    p.add_argument("--collapse_factor", type=int, default=5)
    p.add_argument("--max_duration_ms", type=float, default=700.0)
    # ── Frozen biophysics / recipe (fixed; pinned to the 512-hidden roof run) ──
    p.add_argument("--tau_soma", type=float, default=15.0)
    p.add_argument("--tau_dend", type=float, default=15.0)
    p.add_argument("--tau_m", type=float, default=20.0)
    p.add_argument("--tau_w", type=float, default=100.0)
    p.add_argument("--a_adapt", type=float, default=0.0,
                   help="Subthreshold adaptation coupling (fixed; 0.0 = off).")
    p.add_argument("--b_adapt", type=float, default=0.0,
                   help="Spike-triggered adaptation jump (fixed; 0.0 = off).")
    p.add_argument("--tau_plat_min", type=float, default=150.92269413835055,
                   help="Plateau duration min (fixed; tau_plat_max is searched).")
    p.add_argument("--v_th", type=float, default=1.0)
    p.add_argument("--mu_th", type=float, default=1.0,
                   help="Dendritic plateau threshold (fixed; not searched).")
    p.add_argument("--weight_scale", type=float, default=0.1,
                   help="Xavier-std multiplier at init (fixed; not searched).")
    p.add_argument("--beta_s", type=float, default=10.0,
                   help="Soma surrogate sharpness, soma path (fixed; not searched).")
    p.add_argument("--beta_s_dend", type=float, default=10.0,
                   help="Soma surrogate sharpness, dend path (fixed; not searched). "
                        "With --dend_surrogate_roof this is the roof's out-of-band decay.")
    p.add_argument("--beta_d", type=float, default=10.0,
                   help="Dend surrogate sharpness (fixed; not searched).")
    p.add_argument("--dend_surrogate_roof", action="store_true",
                   help="Use the flat-roof dendritic surrogate over [v_th-gamma, v_th] "
                        "(a distinct gradient path). Enable to match the target run.")
    p.add_argument("--dropout", type=float, default=0.32,
                   help="Dropout rate (fixed across the search; SHD recipe).")
    # NOTE: lr, loss_temperature, loss_label_smoothing, gamma, channel_shift_range,
    # tau_plat_max are TUNED (see the objective). Channel-shift augmentation is
    # always applied (its range is a tuned dimension). loss_count_bias is FROZEN
    # to 0.0 in build_config.
    # ── Training (fixed) ──
    p.add_argument("--optimizer", choices=["sgd", "adam"], default="adam")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--weight_decay", type=float, default=2.6e-6)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--adam_eps", type=float, default=1e-8)
    # ── Fixed LR-schedule shape ──
    p.add_argument("--lr_patience", type=int, default=9)
    p.add_argument("--lr_factor", type=float, default=0.7)
    p.add_argument("--lr_min", type=float, default=1e-6)
    p.add_argument("--precision", choices=["32", "64"], default=_PRECISION)
    return p.parse_args()


def build_config(args, *, loss_temperature, loss_label_smoothing, gamma,
                 tau_plat_max):
    return NeuronConfig(
        dt=args.bin_size_ms,
        tau_soma=args.tau_soma, tau_dend=args.tau_dend, tau_m=args.tau_m,
        tau_plat_min=args.tau_plat_min, tau_plat_max=tau_plat_max,  # max tuned
        tau_w=args.tau_w, a_adapt=args.a_adapt, b_adapt=args.b_adapt,
        mu_th=args.mu_th, v_th=args.v_th, gamma=gamma,
        beta_s=args.beta_s, beta_s_dend=args.beta_s_dend, beta_d=args.beta_d,
        dend_surrogate_roof=args.dend_surrogate_roof,  # roof path (frozen recipe)
        weight_scale=args.weight_scale,
        loss_temperature=loss_temperature, loss_count_bias=0.0,  # frozen no-op
        loss_label_smoothing=loss_label_smoothing,
    )


def train_one_trial(net, train_data, val_data, args, augment_fn, trial):
    """Batched training loop; returns the BEST val accuracy over all epochs."""
    B = args.batch_size
    n_train = len(train_data)
    n_batches = max(1, n_train // B)

    current_lr = args.lr_current
    best_val_acc = 0.0
    epochs_since_lr_drop = 0

    for epoch in range(1, args.epochs + 1):
        idx = np.random.permutation(n_train)
        t0 = time.time()
        for bi in range(n_batches):
            batch_idx = idx[bi * B: bi * B + B]
            x_batch = jnp.stack([augment_fn(train_data[int(i)][0]) for i in batch_idx])
            y_batch = jnp.array([int(train_data[int(i)][1]) for i in batch_idx])
            net.batch_train_step(x_batch, y_batch, lr=current_lr)

        val_acc = evaluate(net, val_data, B)
        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            epochs_since_lr_drop = 0
        else:
            epochs_since_lr_drop += 1

        # Record intermediate value and let the (light) pruner decide.
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if (args.progress_every > 0
                and (epoch % args.progress_every == 0 or epoch == args.epochs)):
            print(
                f"[trial {trial.number:3d}]   epoch {epoch:03d}/{args.epochs} "
                f"val={val_acc:6.2f}%  best={best_val_acc:6.2f}%  "
                f"lr={current_lr:.2e}  ({time.time() - t0:.1f}s)",
                flush=True,
            )

        # ReduceLROnPlateau on the val metric (fixed schedule shape).
        if (args.lr_factor < 1.0 and args.lr_patience > 0
                and current_lr > args.lr_min
                and epochs_since_lr_drop >= args.lr_patience):
            new_lr = max(current_lr * args.lr_factor, args.lr_min)
            if new_lr < current_lr:
                current_lr = new_lr
                epochs_since_lr_drop = 0

    return best_val_acc


def main():
    args = parse_args()
    if args.precision != _PRECISION:
        raise ValueError(
            f"--precision mismatch ({args.precision} vs {_PRECISION}). Pass it once."
        )

    dtype = np.float64 if args.precision == "64" else np.float32
    print(f"Loading SHD once (precision=float{args.precision})...", flush=True)
    X_tr, y_tr, _, X_te, y_te, _, spk_tr, _ = load_shd_binned(
        bin_size_ms=args.bin_size_ms,
        collapse_factor=args.collapse_factor,
        max_duration_ms=args.max_duration_ms,
        binarize=False, dtype=dtype, return_speakers=True,
    )
    all_train = [(X_tr[i], int(y_tr[i])) for i in range(len(y_tr))]
    train_data, val_data = split_by_speakers(all_train, spk_tr, args.val_speakers)
    n_inputs = train_data[0][0].shape[1]
    T = train_data[0][0].shape[0]
    base_key = random.PRNGKey(args.seed)

    # Report the JAX backend/device up front so it's unambiguous whether the
    # search is on GPU.
    dev = jax.local_devices()[0]
    print(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}", flush=True)
    if hasattr(dev, "memory_stats") and dev.memory_stats() is not None:
        ms = dev.memory_stats()
        print(
            f"Device: {dev}  |  GPU memory: {ms['bytes_in_use']/1e6:.1f} MB in use, "
            f"{ms['peak_bytes_in_use']/1e6:.1f} MB peak, "
            f"{ms['bytes_limit']/1e6:.1f} MB pool",
            flush=True,
        )
    else:
        print(f"Device: {dev} (no memory stats available)", flush=True)

    print(
        f"Search: {n_inputs} -> {args.n_hidden} -> {args.n_outputs}  T={T}  "
        f"val_speakers={sorted(args.val_speakers)} "
        f"(val N={len(val_data)}, train N={len(train_data)})  "
        f"trials={args.n_trials}  epochs/trial={args.epochs}  "
        f"opt={args.optimizer}  bs={args.batch_size}  pruner={args.pruner}",
        flush=True,
    )
    print(
        "Objective: macro-averaged (mean per-class) val top-1 accuracy "
        "(equal weight per digit).",
        flush=True,
    )
    print(
        "Tuned: lr loss_temperature loss_label_smoothing gamma "
        "channel_shift_range tau_plat_max",
        flush=True,
    )
    print(
        "Frozen recipe: "
        f"loss_count_bias=0.0 (no-op) weight_scale={args.weight_scale} "
        f"mu_th={args.mu_th} beta_s={args.beta_s} beta_s_dend={args.beta_s_dend} "
        f"beta_d={args.beta_d} dend_surrogate_roof={args.dend_surrogate_roof} "
        f"tau_soma={args.tau_soma} tau_dend={args.tau_dend} tau_m={args.tau_m} "
        f"tau_w={args.tau_w} a_adapt={args.a_adapt} b_adapt={args.b_adapt} "
        f"tau_plat_min={args.tau_plat_min} (max in {TAU_PLAT_MAX_RANGE}) v_th={args.v_th} "
        f"dropout={args.dropout} weight_decay={args.weight_decay} "
        f"channel_shift=tuned{CHANNEL_SHIFT_RANGE} "
        f"lr_patience={args.lr_patience} lr_factor={args.lr_factor}",
        flush=True,
    )

    def objective(trial):
        # Reproducible-per-trial data order / augmentation.
        np.random.seed(args.seed + trial.number)

        lr = trial.suggest_float("lr", *LR_RANGE, log=True)
        loss_temperature = trial.suggest_float("loss_temperature", *LOSS_TEMPERATURE_RANGE)
        loss_label_smoothing = trial.suggest_float(
            "loss_label_smoothing", *LABEL_SMOOTHING_RANGE)
        gamma = trial.suggest_float("gamma", *GAMMA_RANGE)
        channel_shift = trial.suggest_int("channel_shift_range", *CHANNEL_SHIFT_RANGE)
        tau_plat_max = trial.suggest_float("tau_plat_max", *TAU_PLAT_MAX_RANGE)

        print(
            f"[trial {trial.number:3d}] start: lr={lr:.2e} "
            f"temp={loss_temperature:.3f} smooth={loss_label_smoothing:.3f} "
            f"gamma={gamma:.3f} channel_shift={channel_shift:d} "
            f"tau_plat_max={tau_plat_max:.1f}",
            flush=True,
        )

        config = build_config(
            args, loss_temperature=loss_temperature,
            loss_label_smoothing=loss_label_smoothing,
            gamma=gamma, tau_plat_max=tau_plat_max,
        )
        key = random.fold_in(base_key, trial.number)
        net = Network(
            key, n_inputs, args.n_hidden, args.n_outputs, config,
            optimizer=args.optimizer, beta1=args.beta1, beta2=args.beta2,
            adam_eps=args.adam_eps, dropout_rate=args.dropout,
            weight_decay=args.weight_decay,
        )

        def augment_fn(x):
            # Channel-shift range is a tuned dimension, so always apply it (only to
            # train inputs; eval stays un-augmented).
            return apply_channel_shift(x, channel_shift)

        args.lr_current = lr  # starting lr for this trial's schedule
        return train_one_trial(net, train_data, val_data, args, augment_fn, trial)

    def _log_trial(study, trial):
        state = trial.state.name
        best = study.best_value if study.best_trial else float("nan")
        val = trial.value if trial.value is not None else float("nan")
        print(
            f"[trial {trial.number:3d}] {state:8s} best_val={val:6.2f}  "
            f"study_best={best:6.2f}  params={trial.params}",
            flush=True,
        )

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        direction="maximize", sampler=sampler,
        pruner=build_pruner(args),
        study_name=args.study_name,
        storage=(args.storage or None),
        load_if_exists=bool(args.storage),
    )
    study.optimize(objective, n_trials=args.n_trials, callbacks=[_log_trial])

    print("\n=== Best trial ===", flush=True)
    print(f"  best val acc (max over epochs): {study.best_value:.2f}%", flush=True)
    print(f"  params: {json.dumps(study.best_params, indent=2)}", flush=True)

    out = args.out
    if not out:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out = os.path.join(_SCRIPT_DIR, "tune_results",
                           f"best_{args.study_name}_{ts}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "best_value": study.best_value,
            "best_params": study.best_params,
            "n_trials": len(study.trials),
            "val_speakers": sorted(args.val_speakers),
            "n_hidden": args.n_hidden,
            "fixed_args": {k: v for k, v in vars(args).items() if k != "lr_current"},
        }, f, indent=2)
    print(f"Wrote best params -> {out}", flush=True)

    # Ready-to-run retrain command (on ALL train speakers) with the winner.
    bp = study.best_params
    roof_flag = "--dend_surrogate_roof " if args.dend_surrogate_roof else ""
    print(
        "\nRetrain the winner on ALL speakers, then report test:\n"
        f"  python Soft_Reset/run_shd.py --n_hidden {args.n_hidden} "
        f"--n_outputs {args.n_outputs} "
        f"--optimizer {args.optimizer} --precision {args.precision} "
        f"--batch_size {args.batch_size} --epochs {args.epochs} "
        f"--bin_size_ms {args.bin_size_ms} --collapse_factor {args.collapse_factor} "
        f"--max_duration_ms {args.max_duration_ms} "
        f"--lr {bp['lr']:.6g} --weight_scale {args.weight_scale:.4g} "
        f"--loss_temperature {bp['loss_temperature']:.4g} "
        f"--loss_count_bias 0 "
        f"--loss_label_smoothing {bp['loss_label_smoothing']:.4g} "
        f"--gamma {bp['gamma']:.4g} --mu_th {args.mu_th:.4g} "
        f"--beta_s {args.beta_s:.4g} --beta_s_dend {args.beta_s_dend:.4g} "
        f"--beta_d {args.beta_d:.4g} {roof_flag}"
        f"--tau_plat_min {args.tau_plat_min} --tau_plat_max {bp['tau_plat_max']:.6g} "
        f"--v_th {args.v_th} --tau_w {args.tau_w} "
        f"--a_adapt {args.a_adapt} --b_adapt {args.b_adapt} "
        f"--tau_soma {args.tau_soma} --tau_dend {args.tau_dend} --tau_m {args.tau_m} "
        f"--dropout {args.dropout} --weight_decay {args.weight_decay} "
        f"--augment_channel_shift --channel_shift_range {bp['channel_shift_range']} "
        f"--lr_patience {args.lr_patience} --lr_factor {args.lr_factor} "
        f"--lr_min {args.lr_min} --seed {args.seed}",
        flush=True,
    )


if __name__ == "__main__":
    main()
