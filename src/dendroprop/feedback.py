"""Top-down feedback for the credit routing.

The hidden error is ``e_n = delta @ M`` where ``delta = target - probs`` (J,) and
``M`` is (J, N). ``M = w_readout`` gives **backprop / BPTT**; ``M = B`` (a fixed
random matrix) gives **feedback alignment (DFA/RFA)** — identical for a single hidden
layer, where both reduce to one random projection of the output error. Feedback
alignment closes over a fixed random ``B``; it must not fall back to ``w_readout``,
which would degrade it to BPTT. See docs/architecture.md.
"""
import jax.numpy as jnp
from jax import random


def error_signal(delta, M):
    """Top-down error projected onto hidden units: ``e_n = delta @ M``.

    delta (J,), M (J, N) -> (N,). ``M = w_readout`` (backprop) or fixed random B
    (feedback alignment). This is the sole site the feedback matrix enters the
    gradient; grad_r never uses M, so grad_r is invariant to the routing choice.
    """
    return delta @ M


def init_feedback_weights(key, n_outputs, n_hidden, config):
    """Fixed random feedback matrix B, shape (J, N), mirroring w_readout's init.

    Xavier std ``sqrt(2/n_hidden) * config.weight_scale`` (the readout's own init,
    n_inputs = n_hidden there), so B is statistically like a fresh w_readout and
    grad_s/grad_d land at BPTT magnitude. Non-trained: created once at model init
    and stored in the run artifact.
    """
    xavier_std = jnp.sqrt(2.0 / n_hidden)
    return random.normal(key, (n_outputs, n_hidden)) * xavier_std * config.weight_scale
