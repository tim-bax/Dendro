"""The forward state-update equations, shared by inference and training.

``dynamics_step`` is the two-compartment neuron's per-step update (eqs 17-23),
operating on a ``DynamicsState`` struct. Training, inference, and the activity
diagnostic all route through it, so their dynamics cannot diverge.

The LI readout's leaky integration (``readout_integrate``) lives here too: there is
no separate readout module, and inference needs the readout to produce
``mean_voltage``.

``predict`` / ``activity`` are the forward-only scans built on the above, with
single- and batched (vmap) jitted variants. They are also the single-layer reference
the golden fixtures replay against, so the frozen inputs feed straight in. See
docs/architecture.md.
"""

import jax.numpy as jnp
from jax import jit, lax, vmap

from .state import DynamicsState


def dynamics_step(state, dend_in, soma_in, t, alpha_s, alpha_d, T_p, config, alpha_w):
    """One timestep of the two-compartment dynamics. JIT-friendly, no side effects.

    Returns (new_state, o, v_pre_reset, h_new, h_prev, mu_at_tprime_new): the
    updated ``DynamicsState`` plus the observables the training path's surrogate
    gradients need — the spike, the pre-reset voltage, the new/previous plateau
    flags, and the dendritic potential latched at plateau onset.
    """
    mu_prev, v_prev, h_prev = state.mu, state.v, state.h
    t_prime_prev, mu_at_tprime_prev, w_prev = state.t_prime, state.mu_at_tprime, state.w

    t_prime = jnp.where(t == 0, 0, jnp.where(h_prev == 1, t_prime_prev, t))
    mu = jnp.where(t > 0, alpha_d * mu_prev + (1 - h_prev) * dend_in, dend_in)
    mu_at_tprime = jnp.where(h_prev == 0, mu, mu_at_tprime_prev)

    plateau_duration = t - t_prime
    h = jnp.where(
        (mu_at_tprime >= config.mu_th)
        & (plateau_duration <= T_p)
        & (plateau_duration >= 0),
        1,
        0,
    ).astype(jnp.int32)

    v_pre_reset = jnp.where(t > 0, alpha_s * v_prev + soma_in - w_prev, soma_in)
    o = jnp.where(v_pre_reset >= config.v_th - config.gamma * h, 1, 0).astype(jnp.int32)
    v = v_pre_reset * (1 - o)
    w = (
        alpha_w * w_prev
        + (1 - alpha_w) * config.a_adapt * v_pre_reset
        + config.b_adapt * o
    )

    new_state = DynamicsState(
        mu=mu, v=v, h=h, t_prime=t_prime, mu_at_tprime=mu_at_tprime, w=w
    )
    return new_state, o, v_pre_reset, h, h_prev, mu_at_tprime


def readout_integrate(v_prev, sum_v_prev, spike_input, w, alpha_m):
    """LI readout leaky integration for one timestep — no threshold, no reset.

    Returns (v, sum_v).
    """
    v = alpha_m * v_prev + spike_input @ w.T
    return v, sum_v_prev + v


# ══════════════════════════════════════════════════════════════════════
#  Forward-only inference scans (one sample). vmap wraps them for batches.
#  The signature order lets the frozen golden inputs feed straight in
#  (w_readout/alpha_m are unused by _activity but kept in-signature to share
#  the vmap in_axes spec).
# ══════════════════════════════════════════════════════════════════════


def _predict(
    x_input, w_dend, w_soma, w_readout, alpha_s, alpha_d, alpha_m, T_p, config, alpha_w
):
    """Forward pass only -> mean_voltage (J,). No eligibility, no dropout."""
    dend_inputs = x_input @ w_dend.T
    soma_inputs = x_input @ w_soma.T
    T = x_input.shape[0]
    n_hidden = w_dend.shape[0]
    n_outputs = w_readout.shape[0]
    time_indices = jnp.arange(T, dtype=jnp.int32)

    def step(carry, inputs):
        h_state, r_v, r_sum_v = carry
        dend_in, soma_in, t = inputs
        h_state, o, *_ = dynamics_step(
            h_state,
            dend_in,
            soma_in,
            t,
            alpha_s,
            alpha_d,
            T_p,
            config,
            alpha_w,
        )
        r_v, r_sum_v = readout_integrate(
            r_v,
            r_sum_v,
            o.astype(jnp.float64),
            w_readout,
            alpha_m,
        )
        return (h_state, r_v, r_sum_v), None

    init = (DynamicsState.zeros(n_hidden), jnp.zeros(n_outputs), jnp.zeros(n_outputs))
    (_, _, sum_v), _ = lax.scan(step, init, (dend_inputs, soma_inputs, time_indices))
    return sum_v / T  # mean voltage


def _activity(
    x_input, w_dend, w_soma, w_readout, alpha_s, alpha_d, alpha_m, T_p, config, alpha_w
):
    """Mean hidden firing rate = spikes / (T * n_neurons). Forward only."""
    dend_in = x_input @ w_dend.T
    soma_in = x_input @ w_soma.T
    T = x_input.shape[0]
    n = w_dend.shape[0]
    time_indices = jnp.arange(T, dtype=jnp.int32)

    # w_readout, alpha_m unused — present so this shares _PRED_AXES with _predict.

    def step(carry, inputs):
        h_state, s = carry
        dend_in_t, soma_in_t, t = inputs
        h_state, o, *_ = dynamics_step(
            h_state,
            dend_in_t,
            soma_in_t,
            t,
            alpha_s,
            alpha_d,
            T_p,
            config,
            alpha_w,
        )
        return (h_state, s + jnp.sum(o.astype(jnp.float64))), None

    (_, s), _ = lax.scan(
        step, (DynamicsState.zeros(n), 0.0), (dend_in, soma_in, time_indices)
    )
    return s / (T * n)


# vmap in_axes: x_input batched (0), everything else shared (None).
_PRED_AXES = (0, None, None, None, None, None, None, None, None, None)

predict = jit(_predict)
activity = jit(_activity)
predict_batch = jit(vmap(_predict, in_axes=_PRED_AXES))
activity_batch = jit(vmap(_activity, in_axes=_PRED_AXES))
