#!/usr/bin/env python3
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

from data.shd_binned import load_shd_binned, apply_channel_shift
from config import NeuronConfig
from network import Network


def augment_sample(x, args):
    """Apply enabled training-time augmentations to one sample (training only)."""
    if args.augment_channel_shift:
        x = apply_channel_shift(x, args.channel_shift_range)
    return x


def parse_args():
    p = argparse.ArgumentParser(description="No-history model on SHD (count-bin preprocessing)")
    p.add_argument("--bin_size_ms", type=float, default=4.0,
                   help="Time bin width in ms (paper default 4.0; also try 10, 14).")
    p.add_argument("--collapse_factor", type=int, default=5,
                   help="Sum-pool every N consecutive input channels (paper default 5: 700 -> 140).")
    p.add_argument("--max_duration_ms", type=float, default=1400.0,
                   help="Fixed window length in ms (paper default 1400). Sets T = ceil(max_duration_ms / bin_size_ms).")
    p.add_argument("--hidden", type=int, nargs="+", default=[64],
                   help="Hidden-layer sizes, one per layer. The number of values "
                        "sets the depth, e.g. --hidden 128 64 32 builds three "
                        "hidden layers of 128, 64 and 32 two-comp neurons.")
    p.add_argument("--feedback_scale", type=float, default=0.1,
                   help="Std of the fixed random DFA feedback matrices B_l "
                        "(J x N_l), sampled once at init. Tunable.")
    p.add_argument("--n_outputs", type=int, default=20)
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
    p.add_argument("--tau_soma", type=float, default=15.0,
                   help="Soma membrane time constant (physical ms; default 15.0).")
    p.add_argument("--tau_dend", type=float, default=15.0,
                   help="Dendritic membrane time constant (physical ms; default 15.0).")
    p.add_argument("--tau_m", type=float, default=20.0,
                   help="Readout LIF membrane time constant (physical ms; default 20.0).")
    p.add_argument("--tau_plat_min", type=float, default=100.0,
                   help="Plateau duration min (physical ms; default 100).")
    p.add_argument("--tau_plat_max", type=float, default=350.0,
                   help="Plateau duration max (physical ms; default 350).")
    p.add_argument("--mu_th", type=float, default=1.0,
                   help="Dendritic plateau threshold (default 1.0). Lower if plateaus rarely fire.")
    p.add_argument("--v_th", type=float, default=1.0,
                   help="Somatic spike threshold (default 1.0). Lower if neurons rarely spike.")
    p.add_argument("--gamma", type=float, default=0.5,
                   help="Plateau-induced threshold reduction (default 0.5). Effective v_th = v_th - gamma*h.")
    p.add_argument("--tau_w", type=float, default=100.0,
                   help="Adaptation current time constant (ms; default 100.0).")
    p.add_argument("--a_adapt", type=float, default=0.0,
                   help="Subthreshold adaptation coupling (default 0.0 = disabled).")
    p.add_argument("--b_adapt", type=float, default=0.0,
                   help="Spike-triggered adaptation jump size (default 0.0 = disabled).")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument(
        "--augment_channel_shift",
        action="store_true",
        help="Enable channel-shift augmentation on training inputs only.",
    )
    p.add_argument(
        "--channel_shift_range",
        type=int,
        default=5,
        help="Channel-shift range in channels (uniform in [-range, +range]); "
             "operates on the collapsed channel axis.",
    )
    p.add_argument("--weight_decay", type=float, default=0.0,
                   help="Decoupled weight decay (AdamW-style for Adam, "
                        "L2-equivalent for SGD). Typical: 1e-5 to 1e-3.")
    p.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--adam_eps", type=float, default=1e-8)
    p.add_argument("--lr_patience", type=int, default=5,
                   help="ReduceLROnPlateau patience (epochs without test-acc improvement). "
                        "0 disables scheduling.")
    p.add_argument("--lr_factor", type=float, default=0.7,
                   help="ReduceLROnPlateau multiplier (lr := lr * factor). "
                        "Set to 1.0 to disable.")
    p.add_argument("--lr_min", type=float, default=1e-6,
                   help="LR floor; scheduler will not reduce below this.")
    p.add_argument("--early_stop_patience", type=int, default=0,
                   help="Stop training if no test-acc improvement for this many epochs. "
                        "0 disables.")
    p.add_argument(
        "--precision",
        choices=["32", "64"],
        default=_PRECISION,
        help="Floating-point precision for JAX computations.",
    )
    p.add_argument("--save_model", type=str, default="",
                   help="Path to write the final trained model (.npz). "
                        "If empty (default), auto-generate "
                        "no_history/models/shd_seed{seed}_{timestamp}.npz.")
    p.add_argument("--no_save_model", action="store_true",
                   help="Disable saving the trained model entirely.")
    return p.parse_args()


def evaluate(net, dataset, batch_size=1):
    n = len(dataset)
    if batch_size <= 1:
        correct = sum(1 for x, y in dataset if net.predict(x) == int(y))
        return 100.0 * correct / max(n, 1)

    correct = 0
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = dataset[start:end]
        actual = len(batch)

        xs = [x for x, y in batch]
        ys = jnp.array([int(y) for x, y in batch])

        if actual < batch_size:
            xs += [xs[0]] * (batch_size - actual)

        preds = net.batch_predict(jnp.stack(xs))
        correct += int(jnp.sum(preds[:actual] == ys))

    return 100.0 * correct / max(n, 1)


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

    print("Loading SHD data with count-bin preprocessing...", flush=True)
    dtype = np.float64 if args.precision == "64" else np.float32
    X_tr, y_tr, _, X_te, y_te, _ = load_shd_binned(
        bin_size_ms=args.bin_size_ms,
        collapse_factor=args.collapse_factor,
        max_duration_ms=args.max_duration_ms,
        binarize=False,
        dtype=dtype,
    )
    train_data = [(X_tr[i], int(y_tr[i])) for i in range(len(y_tr))]
    test_data = [(X_te[i], int(y_te[i])) for i in range(len(y_te))]
    T = train_data[0][0].shape[0]
    n_inputs = train_data[0][0].shape[1]
    print(
        f"Train: {len(train_data)}  Test: {len(test_data)}  "
        f"n_inputs: {n_inputs}  T: {T}  batch_size: {B}  "
        f"precision=float{args.precision}  "
        f"bin={args.bin_size_ms}ms  collapse={args.collapse_factor}",
        flush=True,
    )

    config = NeuronConfig(
        dt=args.bin_size_ms,
        tau_soma=args.tau_soma,
        tau_dend=args.tau_dend,
        tau_m=args.tau_m,
        tau_plat_min=args.tau_plat_min,
        tau_plat_max=args.tau_plat_max,
        tau_w=args.tau_w,
        a_adapt=args.a_adapt,
        b_adapt=args.b_adapt,
        mu_th=args.mu_th,
        v_th=args.v_th,
        gamma=args.gamma,
        beta_s=args.beta_s,
        beta_d=args.beta_d,
        weight_scale=args.weight_scale,
        loss_temperature=args.loss_temperature,
        loss_count_bias=args.loss_count_bias,
        loss_label_smoothing=args.loss_label_smoothing,
    )
    alpha_s = float(np.exp(-config.dt / config.tau_soma))
    alpha_m = float(np.exp(-config.dt / config.tau_m))
    alpha_d = float(np.exp(-config.dt / config.tau_dend))
    dend_str = f"tau_dend={config.tau_dend}ms -> alpha_d={alpha_d:.4f}"
    print(
        f"Neuron dynamics: dt={config.dt}ms (=bin_size_ms), "
        f"tau_soma={config.tau_soma}ms -> alpha_s={alpha_s:.4f}, "
        f"{dend_str}, "
        f"tau_m={config.tau_m}ms -> alpha_m={alpha_m:.4f}, "
        f"plateau in [{config.tau_plat_min}, {config.tau_plat_max}]ms "
        f"= [{int(config.tau_plat_min/config.dt)}, {int(config.tau_plat_max/config.dt)}] steps",
        flush=True,
    )
    net = Network(
        key, n_inputs, args.hidden, args.n_outputs, config,
        optimizer=args.optimizer, beta1=args.beta1, beta2=args.beta2, adam_eps=args.adam_eps,
        dropout_rate=args.dropout, weight_decay=args.weight_decay,
        feedback_scale=args.feedback_scale,
    )
    opt_str = f"adam(β1={args.beta1},β2={args.beta2})" if args.optimizer == "adam" else "sgd"
    drop_str = f"  dropout={args.dropout}" if args.dropout > 0 else ""
    chan_shift_str = ""
    if args.augment_channel_shift:
        chan_shift_str = f"  augment_channel_shift=True(range=±{args.channel_shift_range})"
    wd_str = f"  weight_decay={args.weight_decay}" if args.weight_decay > 0 else ""
    hidden_str = " -> ".join(f"{n} (2-comp)" for n in args.hidden)
    arch_str = f"{n_inputs} -> {hidden_str} -> {args.n_outputs} (LI readout)"
    print(
        f"Network: {arch_str}  DFA(feedback_scale={args.feedback_scale})  "
        f"optimizer={opt_str}  lr={args.lr}{drop_str}{chan_shift_str}{wd_str}",
        flush=True,
    )

    pre_acc = evaluate(net, test_data, B)
    print(f"Pre-training test accuracy: {pre_acc:.2f}%", flush=True)

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

    n_train = len(train_data)
    n_batches = n_train // B
    samples_per_epoch = n_batches * B
    log_interval = 1000
    log_every = max(1, log_interval // B)

    # Fixed diagnostic batch (test samples, no augmentation) for per-epoch
    # firing-rate readout.
    diag_n = min(len(test_data), 128)
    diag_x = jnp.stack([test_data[i][0] for i in range(diag_n)]) if diag_n else None

    current_lr = args.lr
    best_test_acc = 0.0
    best_epoch = 0
    epochs_since_lr_drop = 0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        idx = np.random.permutation(n_train)
        losses = []
        correct = 0
        gnorm_sums = {}
        gnorm_count = 0
        epoch_t0 = time.time()
        batch_t0 = time.time()

        for bi in range(n_batches):
            start = bi * B
            batch_idx = idx[start : start + B]

            if B == 1:
                x, y = train_data[int(batch_idx[0])]
                x = augment_sample(x, args)
                loss, pred, gnorms = net.train_step(
                    jnp.array(x), int(y), lr=current_lr, clip_value=args.gradient_clip,
                )
                batch_correct = int(pred == int(y))
            else:
                x_batch_np = [
                    augment_sample(train_data[int(i)][0], args)
                    for i in batch_idx
                ]
                x_batch = jnp.stack(x_batch_np)
                y_batch = jnp.array([int(train_data[int(i)][1]) for i in batch_idx])
                loss, preds, gnorms = net.batch_train_step(
                    x_batch, y_batch, lr=current_lr, clip_value=args.gradient_clip,
                )
                batch_correct = int(jnp.sum(preds == y_batch))

            losses.append(loss)
            correct += batch_correct
            for k, v in gnorms.items():
                gnorm_sums[k] = gnorm_sums.get(k, 0.0) + v
            gnorm_count += 1

            if bi == 0 and hasattr(dev, "memory_stats") and dev.memory_stats() is not None:
                ms = dev.memory_stats()
                print(
                    f"GPU after 1st train batch: {ms['bytes_in_use']/1e6:.1f} MB in use, "
                    f"{ms['peak_bytes_in_use']/1e6:.1f} MB peak",
                    flush=True,
                )

            if (bi + 1) % log_every == 0:
                elapsed = time.time() - batch_t0
                samples_done = (bi + 1) * B
                sps = (log_every * B) / max(elapsed, 1e-6)
                avg_loss = float(np.mean(losses[-log_every:]))
                acc_so_far = 100.0 * correct / samples_done
                remaining = (samples_per_epoch - samples_done) / max(sps, 1e-6)
                print(
                    f"  [{samples_done:5d}/{samples_per_epoch}] loss={avg_loss:.4f} "
                    f"acc={acc_so_far:.1f}% | {sps:.1f} samples/s, "
                    f"~{remaining:.0f}s remaining",
                    flush=True,
                )
                batch_t0 = time.time()

        epoch_elapsed = time.time() - epoch_t0
        train_acc = 100.0 * correct / max(samples_per_epoch, 1)
        test_acc = evaluate(net, test_data, B)
        avg_loss = float(np.mean(losses)) if losses else 0.0

        improved = test_acc > best_test_acc
        if improved:
            best_test_acc = test_acc
            best_epoch = epoch
            epochs_since_lr_drop = 0
            epochs_without_improvement = 0
            marker = "  *best"
        else:
            epochs_since_lr_drop += 1
            epochs_without_improvement += 1
            marker = ""

        print(
            f"Epoch {epoch:03d} | loss={avg_loss:.4f} "
            f"train_acc={train_acc:.2f}% test_acc={test_acc:.2f}% "
            f"lr={current_lr:.2e} ({epoch_elapsed:.1f}s){marker}",
            flush=True,
        )

        # Per-epoch gradient magnitudes (mean over batches) and firing rates.
        if gnorm_count > 0:
            # Per-layer keys dend0/soma0/dend1/... then readout last.
            layer_keys = sorted(
                (k for k in gnorm_sums if k != "readout"),
                key=lambda k: (int(k[4:]), k[:4]),
            )
            key_order = layer_keys + (["readout"] if "readout" in gnorm_sums else [])
            gn_str = "  ".join(
                f"{k}={gnorm_sums[k] / gnorm_count:.4g}"
                for k in key_order
            )
            rate_str = ""
            if diag_x is not None:
                rates = net.activity(diag_x)
                rate_str = "  | firing: " + "  ".join(
                    f"{k}={v:.4f}" for k, v in rates.items()
                )
            print(f"         gnorms: {gn_str}{rate_str}", flush=True)

        if (args.lr_factor < 1.0
                and args.lr_patience > 0
                and current_lr > args.lr_min
                and epochs_since_lr_drop >= args.lr_patience):
            new_lr = max(current_lr * args.lr_factor, args.lr_min)
            if new_lr < current_lr:
                print(f"  LR scheduler: {current_lr:.2e} -> {new_lr:.2e} "
                      f"(no improvement for {epochs_since_lr_drop} epochs)", flush=True)
                current_lr = new_lr
                epochs_since_lr_drop = 0

        if (args.early_stop_patience > 0
                and epochs_without_improvement >= args.early_stop_patience):
            print(f"  Early stopping: no improvement for "
                  f"{epochs_without_improvement} epochs.", flush=True)
            break

    final_acc = evaluate(net, test_data, B)
    print(
        f"\nFinal test accuracy: {final_acc:.2f}%  |  "
        f"Best test accuracy: {best_test_acc:.2f}% (epoch {best_epoch})",
        flush=True,
    )

    if not args.no_save_model:
        if args.save_model:
            model_path = args.save_model
        else:
            ts = time.strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(
                _SCRIPT_DIR, "models", f"shd_seed{args.seed}_{ts}.npz"
            )
        os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
        net.save(model_path, extra={
            "final_acc": float(final_acc),
            "best_acc": float(best_test_acc),
            "best_epoch": int(best_epoch),
            "seed": int(args.seed),
            "args": vars(args),
        })
        print(f"Saved trained model -> {model_path}", flush=True)


if __name__ == "__main__":
    main()
