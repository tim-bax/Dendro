"""Training loop — returns data instead of printing.

``train`` runs the per-step sequence ``forward_accumulate_batch`` ->
``loss_and_grads_batch`` -> batch-mean the grads -> ``optim`` step ->
``replace_trainable``. It threads the optimizer state outside the (pure-pytree)
``Network`` and records per-epoch metrics into a plain-Python :class:`MetricsHistory`
that ``io.py`` serializes and ``figures/`` ingest — no JAX.

Key protocol (so a run is reproducible and the loop is bit-for-bit testable):
epoch ``e`` uses ``fold_in(PRNGKey(seed), e)``; batch ``b`` within it uses
``fold_in(epoch_key, b)`` split into per-sample dropout keys; shuffling uses
``np.random.default_rng([seed, e])``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
import jax.numpy as jnp
from jax import random, tree_util

from .network import (
    smooth_targets,
    forward_accumulate_batch,
    loss_and_grads_batch,
    predict_batch,
    activity_batch,
)
from .optim import adam_init, adam_step, sgd_step, hypers_from_run
from .data.dataset import iterate_batches

# CSV column order for a run's metrics.csv (io.py + figures rely on this).
METRIC_FIELDS = (
    "epoch", "loss", "train_acc", "test_acc",
    "lr", "gnorm_r", "gnorm_s", "gnorm_d", "firing",
)


@dataclass
class EpochMetrics:
    epoch: int
    loss: float
    train_acc: float
    test_acc: float
    lr: float
    gnorm_r: float
    gnorm_s: float
    gnorm_d: float
    firing: float

    def as_row(self) -> dict:
        return {k: getattr(self, k) for k in METRIC_FIELDS}


@dataclass
class MetricsHistory:
    """Per-epoch training metrics as plain data (no JAX arrays)."""

    epochs: List[EpochMetrics] = field(default_factory=list)

    def append(self, m: EpochMetrics) -> None:
        self.epochs.append(m)

    def rows(self) -> List[dict]:
        return [m.as_row() for m in self.epochs]

    @property
    def last(self) -> Optional[EpochMetrics]:
        return self.epochs[-1] if self.epochs else None

    def column(self, name: str) -> List[float]:
        return [getattr(m, name) for m in self.epochs]


def _grad_norms(grads) -> Tuple[float, float, float]:
    """(gnorm_r, gnorm_s, gnorm_d) — L2 norm of the readout, all soma, all dend grads."""
    def l2(leaves):
        if not leaves:
            return 0.0
        return float(jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in leaves)))

    return (
        l2([grads.w_readout]),
        l2(list(grads.w_soma)),
        l2(list(grads.w_dend)),
    )


def evaluate(net, dataset, batch_size: int, want_firing: bool = True) -> Tuple[float, float]:
    """(accuracy, mean firing rate) over ``dataset`` — inference forward, no network dropout.

    ``want_firing=False`` skips the separate ``activity_batch`` scan (returns firing 0.0),
    for callers that only need accuracy. NB: if ``dataset`` carries read-time augmentation,
    these metrics are measured on the augmented stream (test sets are loaded clean).
    """
    correct = 0
    total = 0
    fire_weighted = 0.0
    for X, y in iterate_batches(dataset, batch_size):
        xb = jnp.asarray(X)
        mv = predict_batch(net, xb)                     # (B, J) mean readout voltage
        pred = np.asarray(jnp.argmax(mv, axis=1))
        correct += int(np.sum(pred == np.asarray(y)))
        if want_firing:
            fire_weighted += float(jnp.sum(activity_batch(net, xb)))
        total += len(y)
    total = max(total, 1)
    return correct / total, (fire_weighted / total if want_firing else 0.0)


def _apply_grads(net, grads, run_config, opt_state, hypers):
    """One optimizer step (adam threads state; sgd is stateless). Returns (net, opt_state)."""
    params = net.trainable()
    if run_config.optimizer == "adam":
        new_params, opt_state = adam_step(params, grads, opt_state, hypers)
    elif run_config.optimizer == "sgd":
        new_params = sgd_step(params, grads, hypers)
    else:
        raise ValueError(f"Unknown optimizer {run_config.optimizer!r} (expected adam/sgd)")
    return net.replace_trainable(new_params), opt_state


def train(
    net,
    train_ds,
    test_ds,
    run_config,
    *,
    epochs: int,
    batch_size: int,
    seed: int = 0,
    eval_batch_size: Optional[int] = None,
    shuffle: bool = True,
    lr_schedule: Optional[Callable[[int], float]] = None,
) -> Tuple[object, MetricsHistory]:
    """Train ``net`` on ``train_ds`` for ``epochs``; return ``(trained_net, history)``.

    The ``Network`` is immutable (a pytree), so the trained model is RETURNED, not
    mutated in place. Optimizer state lives here, never on the model. ``lr_schedule``
    (epoch -> lr) drives the per-epoch learning rate; omitted → ``run_config.lr``.
    """
    if run_config.optimizer not in ("adam", "sgd"):
        raise ValueError(
            f"Unknown optimizer {run_config.optimizer!r} (expected adam/sgd)"
        )
    eval_bs = eval_batch_size or batch_size
    n_classes = train_ds.n_classes
    label_smoothing = net.config.loss_label_smoothing
    base_key = random.PRNGKey(seed)

    opt_state = adam_init(net.trainable()) if run_config.optimizer == "adam" else None
    history = MetricsHistory()

    for epoch in range(epochs):
        lr = lr_schedule(epoch) if lr_schedule is not None else run_config.lr
        hypers = hypers_from_run(run_config, lr=lr)
        epoch_key = random.fold_in(base_key, epoch)
        rng = np.random.default_rng([seed, epoch]) if shuffle else None

        losses, gr, gs, gd = [], [], [], []
        for b, (X, y) in enumerate(
            iterate_batches(train_ds, batch_size, shuffle=shuffle, rng=rng)
        ):
            xb = jnp.asarray(X)
            T = xb.shape[1]
            sm = smooth_targets(jnp.asarray(y), n_classes, label_smoothing)
            keys = random.split(random.fold_in(epoch_key, b), xb.shape[0])

            mv, accum = forward_accumulate_batch(net, xb, keys, run_config.dropout_rate)
            loss_b, _, grads_b = loss_and_grads_batch(net, mv, accum, sm, T)
            grads = tree_util.tree_map(lambda g: jnp.mean(g, axis=0), grads_b)

            net, opt_state = _apply_grads(net, grads, run_config, opt_state, hypers)

            losses.append(float(jnp.mean(loss_b)))
            n_r, n_s, n_d = _grad_norms(grads)
            gr.append(n_r); gs.append(n_s); gd.append(n_d)

        train_acc, firing = evaluate(net, train_ds, eval_bs)
        test_acc, _ = evaluate(net, test_ds, eval_bs, want_firing=False)
        history.append(EpochMetrics(
            epoch=epoch,
            loss=float(np.mean(losses)) if losses else float("nan"),
            train_acc=train_acc,
            test_acc=test_acc,
            lr=float(lr),
            gnorm_r=float(np.mean(gr)) if gr else 0.0,
            gnorm_s=float(np.mean(gs)) if gs else 0.0,
            gnorm_d=float(np.mean(gd)) if gd else 0.0,
            firing=firing,
        ))

    return net, history
