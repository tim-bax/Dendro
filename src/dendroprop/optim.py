"""Optimizers — hand-rolled, outside the credit/gradient computation.

``credit.py`` / ``Network.loss_and_grads`` compute the gradients (a ``TrainableParams``
tree); this module applies them. The ``Network`` stays a pure pytree — optimizer state
(``AdamState``) lives outside it and is threaded by the caller (the train loop), never
stored on the model.

AdamW / SGD:
  - **clip-first** — gradients are clipped before the moment/decay update, so the
    weight-decay term is never capped by the gradient clip;
  - **ascent** — the analytic gradients point uphill on the objective, so the update is
    ``w + lr·…`` (not ``w − lr·…``);
  - **decoupled AdamW weight decay** — the ``λ·w`` term bypasses the ``1/√v`` rescaling
    (Loshchilov & Hutter), subtracted directly from ``w``.

The moment trees ``AdamState.m``/``v`` mirror ``Network.trainable()`` (same treedef), so
every update is a leaf-wise ``jax.tree.map`` over ``(param, grad, m, v)`` — the fixed
constants (``T_p`` / ``alpha_*`` / ``B`` / ``config``) are structurally absent from the
trainable tree and therefore never touched. ``step`` increments before the bias
correction, so the first update uses ``step == 1``. See docs/architecture.md.
"""
import jax
import jax.numpy as jnp
from flax import struct

from .state import TrainableParams


@struct.dataclass
class AdamState:
    """Adam optimizer state, threaded by the caller (never stored on the Network).

    ``m``/``v`` are ``TrainableParams`` trees congruent with ``Network.trainable()``;
    ``step`` is an int32 scalar (the update counter, bias correction reads step+1)."""
    m: TrainableParams
    v: TrainableParams
    step: jnp.ndarray  # int32 scalar


@struct.dataclass
class AdamHypers:
    """Per-step AdamW hyperparameters (all traced leaves, so ``lr`` can follow a schedule
    without retracing). Derive from a ``RunConfig`` via ``hypers_from_run``."""
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    clip_value: float = 1.0


def hypers_from_run(run, lr=None):
    """Build ``AdamHypers`` from a ``RunConfig`` for one step. ``lr`` overridable so a
    learning-rate schedule can drive it per step without mutating the ``RunConfig``."""
    return AdamHypers(
        lr=run.lr if lr is None else lr,
        beta1=run.beta1,
        beta2=run.beta2,
        eps=run.adam_eps,
        weight_decay=run.weight_decay,
        clip_value=run.clip_value,
    )


def adam_init(trainable):
    """Zero first/second moments mirroring the ``trainable`` tree; step counter at 0."""
    return AdamState(
        m=jax.tree.map(jnp.zeros_like, trainable),
        v=jax.tree.map(jnp.zeros_like, trainable),
        step=jnp.array(0, dtype=jnp.int32),
    )


def adam_step(params, grads, state, h):
    """One AdamW ascent update. Returns ``(new_params, new_state)``.

    Leaf-wise over the trainable tree: clip → moments → bias-corrected ascent with
    decoupled decay. ``step`` increments before the bias correction (first update uses
    ``step == 1``)."""
    step = state.step + 1
    g = jax.tree.map(lambda gi: jnp.clip(gi, -h.clip_value, h.clip_value), grads)
    m = jax.tree.map(lambda mi, gi: h.beta1 * mi + (1 - h.beta1) * gi, state.m, g)
    v = jax.tree.map(lambda vi, gi: h.beta2 * vi + (1 - h.beta2) * gi ** 2, state.v, g)
    bc1 = 1 - h.beta1 ** step
    bc2 = 1 - h.beta2 ** step
    new_params = jax.tree.map(
        lambda w, mi, vi: w + h.lr * (mi / bc1) / (jnp.sqrt(vi / bc2) + h.eps)
        - h.lr * h.weight_decay * w,
        params, m, v,
    )
    return new_params, AdamState(m=m, v=v, step=step)


def sgd_step(params, grads, h):
    """One SGD ascent update with decoupled weight decay (clip-first):
    ``w + lr·clip(g) − lr·wd·w`` leaf-wise."""
    return jax.tree.map(
        lambda w, gi: w + h.lr * jnp.clip(gi, -h.clip_value, h.clip_value)
        - h.lr * h.weight_decay * w,
        params, grads,
    )
