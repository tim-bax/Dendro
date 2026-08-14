"""Shared training loop for the DFA network.

Both the CLI entry point (``run_shd.py``) and the Optuna tuner (``tune_shd.py``)
call ``train_model`` so the epoch loop, model selection and checkpointing live in
one place.

Model selection: when a validation set is supplied, the LR scheduler, early
stopping and best-checkpoint are driven by a **smoothed** (k-epoch moving
average) validation accuracy — SHD's test accuracy is too noisy to select on, and
selecting on test is peeking anyway. With no validation set the logic falls back
to test accuracy, reproducing the original single-metric behaviour.

This module imports ``jax.numpy`` (via ``network``), so callers MUST set
``jax_enable_x64`` before importing it (see the precision handling in the
entry-point scripts).
"""
from collections import deque

import time

import numpy as np
import jax.numpy as jnp


# ── Data ────────────────────────────────────────────────────────────────────

def split_by_speakers(data, speakers, val_ids):
    """Partition a train list into (train, val) by held-out speaker id.

    data:     list of (x, y) samples.
    speakers: array-like of per-sample speaker ids, aligned with ``data``.
    val_ids:  iterable of speaker ids to route into the validation set.

    Returns (train_data, val_data). Validates that every requested val speaker
    is actually present, and warns if the remaining train split loses a class.
    """
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
            f"{sorted(dropped)} from the training split.",
            flush=True,
        )
    return train_data, val_data


def evaluate(net, dataset, batch_size=1):
    """Classification accuracy (%) over ``dataset`` (no dropout)."""
    n = len(dataset)
    if n == 0:
        return 0.0
    if batch_size <= 1:
        correct = sum(1 for x, y in dataset if net.predict(x) == int(y))
        return 100.0 * correct / n

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
    return 100.0 * correct / n


# ── Training loop ───────────────────────────────────────────────────────────

def train_model(
    net,
    train_data,
    test_data,
    *,
    val_data=None,
    epochs=10,
    lr=1e-3,
    batch_size=1,
    gradient_clip=5.0,
    lr_patience=5,
    lr_factor=0.7,
    lr_min=1e-6,
    early_stop_patience=0,
    val_smooth_window=3,
    augment_fn=None,
    trial=None,
    log=True,
    progress_every=0,
):
    """Train ``net`` and return a summary dict.

    Selection metric = smoothed val accuracy if ``val_data`` is given, else test
    accuracy. The best-on-metric weights are snapshotted and restored into ``net``
    before returning, so the caller's model is the selected checkpoint (not the
    final epoch).

    ``trial`` (an Optuna trial) enables intermediate reporting + pruning.

    ``log`` prints the full per-epoch + intra-epoch batch logs (the CLI path).
    ``progress_every`` (used when ``log`` is False, e.g. during tuning) prints a
    compact one-line heartbeat every N epochs so a quiet run stays assessable.

    Returns dict with: best_metric, best_val, test_at_best, best_epoch,
    final_test, final_val, history.
    """
    use_val = val_data is not None and len(val_data) > 0
    B = max(1, int(batch_size))
    n_train = len(train_data)
    n_batches = n_train // B
    samples_per_epoch = n_batches * B
    log_every = max(1, 1000 // B)

    # Fixed diagnostic batch for per-epoch firing rates.
    diag_src = test_data if test_data else (val_data or [])
    diag_n = min(len(diag_src), 128)
    diag_x = jnp.stack([diag_src[i][0] for i in range(diag_n)]) if diag_n else None

    current_lr = float(lr)
    best_metric = -1.0
    best_val = 0.0
    test_at_best = 0.0
    best_epoch = 0
    best_weights = None
    epochs_since_lr_drop = 0
    epochs_without_improvement = 0
    val_window = deque(maxlen=max(1, int(val_smooth_window)))
    history = []

    for epoch in range(1, epochs + 1):
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
                if augment_fn is not None:
                    x = augment_fn(x)
                loss, pred, gnorms = net.train_step(
                    jnp.array(x), int(y), lr=current_lr, clip_value=gradient_clip,
                )
                batch_correct = int(pred == int(y))
            else:
                x_batch_np = [
                    augment_fn(train_data[int(i)][0]) if augment_fn is not None
                    else train_data[int(i)][0]
                    for i in batch_idx
                ]
                x_batch = jnp.stack(x_batch_np)
                y_batch = jnp.array([int(train_data[int(i)][1]) for i in batch_idx])
                loss, preds, gnorms = net.batch_train_step(
                    x_batch, y_batch, lr=current_lr, clip_value=gradient_clip,
                )
                batch_correct = int(jnp.sum(preds == y_batch))

            losses.append(loss)
            correct += batch_correct
            for k, v in gnorms.items():
                gnorm_sums[k] = gnorm_sums.get(k, 0.0) + v
            gnorm_count += 1

            if log and (bi + 1) % log_every == 0:
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
        avg_loss = float(np.mean(losses)) if losses else 0.0
        test_acc = evaluate(net, test_data, B) if test_data else 0.0
        val_acc = evaluate(net, val_data, B) if use_val else 0.0

        # Selection metric: smoothed val (if val) else raw test.
        if use_val:
            val_window.append(val_acc)
            metric = float(np.mean(val_window))
        else:
            metric = test_acc

        improved = metric > best_metric
        if improved:
            best_metric = metric
            best_val = val_acc
            test_at_best = test_acc
            best_epoch = epoch
            best_weights = list(net._weights_flat())
            epochs_since_lr_drop = 0
            epochs_without_improvement = 0
            marker = "  *best"
        else:
            epochs_since_lr_drop += 1
            epochs_without_improvement += 1
            marker = ""

        history.append({
            "epoch": epoch, "loss": avg_loss, "train_acc": train_acc,
            "val_acc": val_acc, "test_acc": test_acc, "metric": metric,
            "lr": current_lr,
        })

        if log:
            val_str = f"val_acc={val_acc:.2f}%(smooth={metric:.2f}) " if use_val else ""
            print(
                f"Epoch {epoch:03d} | loss={avg_loss:.4f} "
                f"train_acc={train_acc:.2f}% {val_str}test_acc={test_acc:.2f}% "
                f"lr={current_lr:.2e} ({epoch_elapsed:.1f}s){marker}",
                flush=True,
            )
            if gnorm_count > 0:
                layer_keys = sorted(
                    (k for k in gnorm_sums if k != "readout"),
                    key=lambda k: (int(k[4:]), k[:4]),
                )
                key_order = layer_keys + (["readout"] if "readout" in gnorm_sums else [])
                gn_str = "  ".join(
                    f"{k}={gnorm_sums[k] / gnorm_count:.4g}" for k in key_order
                )
                rate_str = ""
                if diag_x is not None:
                    rates = net.activity(diag_x)
                    rate_str = "  | firing: " + "  ".join(
                        f"{k}={v:.4f}" for k, v in rates.items()
                    )
                print(f"         gnorms: {gn_str}{rate_str}", flush=True)

        elif progress_every and (epoch % progress_every == 0 or epoch == epochs):
            vtxt = (f"val={val_acc:.1f}% smooth={metric:.1f}% " if use_val
                    else f"test={test_acc:.1f}% ")
            print(
                f"      [epoch {epoch:3d}/{epochs}] {vtxt}"
                f"best={best_metric:.1f}% lr={current_lr:.1e} "
                f"({epoch_elapsed:.1f}s/ep)",
                flush=True,
            )

        # Optuna intermediate report + pruning (on the selection metric).
        if trial is not None:
            trial.report(metric, epoch)
            if trial.should_prune():
                import optuna
                if best_weights is not None:
                    net._set_weights_flat(best_weights)
                raise optuna.TrialPruned()

        # ReduceLROnPlateau on the selection metric.
        if (lr_factor < 1.0 and lr_patience > 0 and current_lr > lr_min
                and epochs_since_lr_drop >= lr_patience):
            new_lr = max(current_lr * lr_factor, lr_min)
            if new_lr < current_lr:
                if log:
                    print(f"  LR scheduler: {current_lr:.2e} -> {new_lr:.2e} "
                          f"(no improvement for {epochs_since_lr_drop} epochs)", flush=True)
                current_lr = new_lr
                epochs_since_lr_drop = 0

        if (early_stop_patience > 0
                and epochs_without_improvement >= early_stop_patience):
            if log:
                print(f"  Early stopping: no improvement for "
                      f"{epochs_without_improvement} epochs.", flush=True)
            break

    final_test = evaluate(net, test_data, B) if test_data else 0.0
    final_val = evaluate(net, val_data, B) if use_val else 0.0

    # Restore the selected checkpoint into the caller's model.
    if best_weights is not None:
        net._set_weights_flat(best_weights)

    return {
        "best_metric": best_metric,
        "best_val": best_val,
        "test_at_best": test_at_best,
        "best_epoch": best_epoch,
        "final_test": final_test,
        "final_val": final_val,
        "history": history,
    }
