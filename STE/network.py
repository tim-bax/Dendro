import dataclasses
import json

import numpy as np
import jax.numpy as jnp
from jax import random, jit, lax, vmap

from config import NeuronConfig, surrogate_sigma
from two_comp_neuron import TwoCompNeuron
from lif_neuron import LINeuron


# Extra divisor on the somatic gradient only, balancing its magnitude against the
# dendritic gradient (C_soma runs ~8x larger). Empirical, not derived — carried
# over from the single-layer rule and applied per hidden layer.
_SOMA_GRAD_SCALE = 8.0

# Guards log(0) in the cross-entropy. Small enough not to bias the loss.
_LOG_EPS = 1e-8


# ══════════════════════════════════════════════════════════════════════
#  Core functions — each processes ONE sample.
#
#  DFA (Direct Feedback Alignment) with an arbitrary number of hidden
#  layers. Every hidden layer is a TwoCompNeuron; layer 0 reads the input
#  spikes, layer l>0 reads the (dropped) spikes of layer l-1, and the last
#  hidden layer feeds the leaky-integrator readout.
#
#  The learning rule is a per-layer three-factor rule. Each hidden layer l
#  accumulates two eligibility tensors of shape (N_l, N_{l-1}) during the
#  forward scan; these are contracted AFTER the scan with a per-layer
#  learning signal L_l = B_l^T · e, where e is the single end-of-sequence
#  output error and B_l is the layer's feedback matrix (J, N_l). Because B_l
#  is time-independent and e is time-independent, L_l factors out of the time
#  sum, so the accumulators never carry the output index J.
#
#  Feedback matrices (weight-transported symmetric / STE): B_l is built each
#  step from the CURRENT forward weights as a reverse cumulative product,
#      B_l = W_readout · Π_{k>l} (w_soma_k + c·w_dend_k),
#  so the LAST hidden layer uses the live readout weights W_readout verbatim
#  (exact ∂E/∂z, symmetric e-prop, Bellec et al. 2020, Eq. 4) and every deeper
#  layer transports the error down through the two-compartment forward weights
#  above it. The dendrite branch is collapsed to a single straight-through gain
#  c (self.fb_dend_c, default gamma·0.5) instead of the exact per-timestep
#  plateau sensitivity — hence "STE". No fixed random feedback is used.
#
#  weights pytree:  {"dend": [w_dend_0..], "soma": [w_soma_0..],
#                    "readout": w_readout}
#  tp_list:         [T_p_0, .., T_p_{L-1}]  (per-layer, per-neuron plateau)
# ══════════════════════════════════════════════════════════════════════


def _forward_and_accum(
    x_input, weights, alpha_s, alpha_d, alpha_m, tp_list, config, alpha_w,
    h_carry_init, r_carry_init, C_soma_init, A_dend_init, sum_Er_init,
    rng_key, dropout_rate,
):
    """Forward pass + per-layer eligibility accumulation for one sample.

    x_input:      (T, K) input spike train
    weights:      dict with "dend"/"soma" lists (len L) and "readout"
    tp_list:      list of L per-neuron plateau-duration arrays
    h_carry_init: list of L 9-tuples of hidden-neuron state zeros
    r_carry_init: 3-tuple of readout-neuron state zeros
    C_soma_init:  list of L (N_l, N_{l-1}) somatic accumulator zeros
    A_dend_init:  list of L (N_l, N_{l-1}) dendritic accumulator zeros
    sum_Er_init:  (N_{L-1},) readout-eligibility sum zeros
    rng_key:      PRNG key for dropout masks
    dropout_rate: fraction of hidden spikes to drop (0.0 = no dropout)

    Returns: mean_voltage (J,), sum_Er (N_{L-1},),
             C_soma_list (each N_l,N_{l-1}), A_dend_list (each N_l,N_{l-1})
    """
    w_dend = weights["dend"]
    w_soma = weights["soma"]
    w_readout = weights["readout"]
    L = len(w_dend)

    T = x_input.shape[0]
    time_indices = jnp.arange(T, dtype=jnp.int32)
    # One dropout subkey per (timestep, layer): drop each hidden layer's output.
    dropout_keys = random.split(rng_key, T * L).reshape(T, L, 2)
    dropout_scale = 1.0 / (1.0 - dropout_rate)

    # Only layer 0's synaptic drive can be precomputed (its presynaptic input is
    # the fixed x_input); deeper layers depend on spikes produced inside the scan.
    dend_in_0 = x_input @ w_dend[0].T
    soma_in_0 = x_input @ w_soma[0].T

    def step(carry, inputs):
        h_states, r_carry, C_soma, A_dend, sum_Er = carry
        dend_in0_t, soma_in0_t, x_t, t, drop_keys_t = inputs
        x_t = x_t.astype(jnp.float64)

        new_h_states = list(h_states)
        new_C_soma = list(C_soma)
        new_A_dend = list(A_dend)

        o_prev = None
        for l in range(L):
            if l == 0:
                dend_in_l, soma_in_l = dend_in0_t, soma_in0_t
                presyn = x_t
            else:
                dend_in_l = o_prev @ w_dend[l].T
                soma_in_l = o_prev @ w_soma[l].T
                presyn = o_prev

            h_carry, o_l, v_pre_l, h_l, h_prev_l, mu_at_tp_l = TwoCompNeuron.forward_step(
                h_states[l], dend_in_l, soma_in_l, t,
                alpha_s, alpha_d, tp_list[l], config, alpha_w,
            )

            # Local surrogate factors for this layer.
            sp_l = surrogate_sigma(
                v_pre_l + config.gamma * h_l - config.v_th, config.beta_s,
            )
            hp_l = surrogate_sigma(mu_at_tp_l - config.mu_th, config.beta_d)

            # Update this layer's eligibility traces from its presynaptic signal.
            (mu_c, v_c, h_c, tp_c, matp_c, E_soma_c,
             dmu_c, dmu_atp_c, w_c) = h_carry
            E_soma_new = TwoCompNeuron.update_somatic_eligibility(
                E_soma_c, presyn, alpha_s,
            )
            dmu_new, dmu_atp_new = TwoCompNeuron.update_dendritic_eligibility(
                dmu_c, dmu_atp_c, presyn, h_prev_l, alpha_d,
            )
            new_h_states[l] = (
                mu_c, v_c, h_c, tp_c, matp_c, E_soma_new, dmu_new, dmu_atp_new, w_c,
            )

            # Accumulate (N_l, N_{l-1}) tensors; the J-index learning signal is
            # applied after the scan.
            new_C_soma[l] = new_C_soma[l] + sp_l[:, None] * E_soma_new[None, :]
            new_A_dend[l] = new_A_dend[l] + (sp_l * hp_l * config.gamma)[:, None] * dmu_atp_new

            # Dropout this layer's output before it drives the next layer / readout.
            o_l_float = o_l.astype(jnp.float64)
            mask = random.bernoulli(
                drop_keys_t[l], 1.0 - dropout_rate, o_l.shape,
            ).astype(jnp.float64)
            o_prev = o_l_float * mask * dropout_scale

        # Readout sees the (dropped) output of the last hidden layer.
        r_carry, r_v, r_E = LINeuron.forward_step(
            r_carry, o_prev, w_readout, alpha_m,
        )
        sum_Er = sum_Er + r_E

        new_carry = (new_h_states, r_carry, new_C_soma, new_A_dend, sum_Er)
        return new_carry, None

    init_carry = (h_carry_init, r_carry_init, C_soma_init, A_dend_init, sum_Er_init)
    scan_inputs = (dend_in_0, soma_in_0, x_input, time_indices, dropout_keys)
    final_carry, _ = lax.scan(step, init_carry, scan_inputs)

    _, r_carry_f, C_soma_f, A_dend_f, sum_Er_f = final_carry
    mean_voltage = r_carry_f[1] / T  # sum_v / T

    return mean_voltage, sum_Er_f, C_soma_f, A_dend_f


def _predict_only(
    x_input, weights, alpha_s, alpha_d, alpha_m, tp_list, config, alpha_w,
):
    """Forward pass only — no eligibility bookkeeping. Returns mean_voltage (J,).

    Reuses TwoCompNeuron.forward_step (the single definition of the dynamics),
    leaving the three eligibility slots of each carry untouched at zero. No
    dropout.
    """
    w_dend = weights["dend"]
    w_soma = weights["soma"]
    w_readout = weights["readout"]
    L = len(w_dend)

    T = x_input.shape[0]
    time_indices = jnp.arange(T, dtype=jnp.int32)
    dend_in_0 = x_input @ w_dend[0].T
    soma_in_0 = x_input @ w_soma[0].T

    def _zero_h(n, k):
        return (
            jnp.zeros(n), jnp.zeros(n),
            jnp.zeros(n, dtype=jnp.int32), jnp.zeros(n, dtype=jnp.int32),
            jnp.zeros(n), jnp.zeros(k),
            jnp.zeros((n, k)), jnp.zeros((n, k)), jnp.zeros(n),
        )

    n_outputs = w_readout.shape[0]

    def step(carry, inputs):
        h_states, r_carry = carry
        dend_in0_t, soma_in0_t, x_t, t = inputs
        x_t = x_t.astype(jnp.float64)

        new_h_states = list(h_states)
        o_prev = None
        for l in range(L):
            if l == 0:
                dend_in_l, soma_in_l = dend_in0_t, soma_in0_t
            else:
                dend_in_l = o_prev @ w_dend[l].T
                soma_in_l = o_prev @ w_soma[l].T
            h_carry, o_l, *_ = TwoCompNeuron.forward_step(
                h_states[l], dend_in_l, soma_in_l, t,
                alpha_s, alpha_d, tp_list[l], config, alpha_w,
            )
            new_h_states[l] = h_carry
            o_prev = o_l.astype(jnp.float64)

        r_carry, _, _ = LINeuron.forward_step(r_carry, o_prev, w_readout, alpha_m)
        return (new_h_states, r_carry), None

    h_zeros = [
        _zero_h(w_dend[l].shape[0], w_dend[l].shape[1]) for l in range(L)
    ]
    r_zeros = (jnp.zeros(n_outputs), jnp.zeros(n_outputs), jnp.zeros(w_dend[-1].shape[0]))
    (_, r_carry_f), _ = lax.scan(
        step, (h_zeros, r_zeros), (dend_in_0, soma_in_0, x_input, time_indices),
    )
    return r_carry_f[1] / T  # mean voltage


def _loss_and_grads(
    mean_voltage, sum_Er, C_soma_list, A_dend_list, B_list,
    target_smoothed, T, loss_temperature, loss_count_bias,
):
    """Compute loss and per-layer DFA weight gradients for one sample."""
    scaled_logits = mean_voltage / loss_temperature + loss_count_bias
    probs = jnp.exp(scaled_logits - jnp.max(scaled_logits))
    probs = probs / jnp.sum(probs)

    prediction = jnp.argmax(mean_voltage)
    loss = -jnp.sum(target_smoothed * jnp.log(probs + _LOG_EPS))
    global_error = target_smoothed - probs  # e, shape (J,)

    # Readout: exact gradient (this layer sits directly at the loss).
    grad_readout = (global_error[:, None] * sum_Er[None, :]) / T

    # Hidden layers: broadcast the error through each layer's fixed random B_l.
    grad_soma_list = []
    grad_dend_list = []
    for C_soma, A_dend, B in zip(C_soma_list, A_dend_list, B_list):
        L_signal = B.T @ global_error  # (N_l,)
        grad_soma_list.append(
            (L_signal[:, None] * C_soma) / (T * _SOMA_GRAD_SCALE)
        )
        grad_dend_list.append((L_signal[:, None] * A_dend) / T)

    return loss, prediction, grad_readout, grad_soma_list, grad_dend_list


def _apply_grads(weights_flat, grads_flat, lr, clip_value, weight_decay):
    """SGD with decoupled weight decay over a flat list of weight tensors.

    Gradients are clipped first, then w ← w + lr·g − lr·λ·w. For plain SGD the
    decay term is equivalent to an L2 penalty (λ/2)·||w||² but stays outside the
    clip so it can't be capped.
    """
    new = []
    for w, g in zip(weights_flat, grads_flat):
        g = jnp.clip(g, -clip_value, clip_value)
        new.append(w + lr * g - lr * weight_decay * w)
    return new


def _adam_apply(
    weights_flat, grads_flat, m_flat, v_flat,
    step, lr, beta1, beta2, eps, clip_value, weight_decay,
):
    """AdamW-style decoupled weight decay over a flat list of weight tensors."""
    new_w, new_m, new_v = [], [], []
    for w, g, m, v in zip(weights_flat, grads_flat, m_flat, v_flat):
        g = jnp.clip(g, -clip_value, clip_value)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2
        m_hat = m / (1 - beta1 ** step)
        v_hat = v / (1 - beta2 ** step)
        w = w + lr * m_hat / (jnp.sqrt(v_hat) + eps) - lr * weight_decay * w
        new_w.append(w)
        new_m.append(m)
        new_v.append(v)
    return new_w, new_m, new_v


# ══════════════════════════════════════════════════════════════════════
#  Activity diagnostic — mean firing rate per hidden layer (forward only).
# ══════════════════════════════════════════════════════════════════════

def _activity(
    x_input, weights, alpha_s, alpha_d, alpha_m, tp_list, config, alpha_w,
):
    w_dend = weights["dend"]
    w_soma = weights["soma"]
    L = len(w_dend)
    T = x_input.shape[0]
    time_indices = jnp.arange(T, dtype=jnp.int32)
    dend_in_0 = x_input @ w_dend[0].T
    soma_in_0 = x_input @ w_soma[0].T

    def _zero_h(n, k):
        return (
            jnp.zeros(n), jnp.zeros(n),
            jnp.zeros(n, dtype=jnp.int32), jnp.zeros(n, dtype=jnp.int32),
            jnp.zeros(n), jnp.zeros(k),
            jnp.zeros((n, k)), jnp.zeros((n, k)), jnp.zeros(n),
        )

    def step(carry, inputs):
        h_states, spike_sums = carry
        dend_in0_t, soma_in0_t, x_t, t = inputs
        x_t = x_t.astype(jnp.float64)

        new_h_states = list(h_states)
        new_sums = list(spike_sums)
        o_prev = None
        for l in range(L):
            if l == 0:
                dend_in_l, soma_in_l = dend_in0_t, soma_in0_t
            else:
                dend_in_l = o_prev @ w_dend[l].T
                soma_in_l = o_prev @ w_soma[l].T
            h_carry, o_l, *_ = TwoCompNeuron.forward_step(
                h_states[l], dend_in_l, soma_in_l, t,
                alpha_s, alpha_d, tp_list[l], config, alpha_w,
            )
            new_h_states[l] = h_carry
            new_sums[l] = new_sums[l] + jnp.sum(o_l.astype(jnp.float64))
            o_prev = o_l.astype(jnp.float64)
        return (new_h_states, new_sums), None

    h_zeros = [_zero_h(w_dend[l].shape[0], w_dend[l].shape[1]) for l in range(L)]
    (_, sums), _ = lax.scan(
        step, (h_zeros, [jnp.array(0.0)] * L),
        (dend_in_0, soma_in_0, x_input, time_indices),
    )
    # Rate per layer = spikes / (T * n_neurons).
    return [sums[l] / (T * w_dend[l].shape[0]) for l in range(L)]


# ══════════════════════════════════════════════════════════════════════
#  Pre-compiled versions.
#
#  vmap in_axes: x, carry/accumulator inits and the PRNG key are batched
#  (axis 0); weights, params (alphas/T_p/config) and dropout_rate are shared
#  (None). Passing weights/params as pytrees lets a single None cover each
#  whole subtree.
# ══════════════════════════════════════════════════════════════════════

_FWD_AXES = (
    0,                                  # x_input
    None,                               # weights (pytree)
    None, None, None,                   # alpha_s, alpha_d, alpha_m
    None, None, None,                   # tp_list, config, alpha_w
    0, 0, 0, 0, 0,                      # h/r carry inits, C_soma, A_dend, sum_Er
    0,                                  # rng_key
    None,                               # dropout_rate
)

_PRED_AXES = (
    0,                                  # x_input
    None,                               # weights
    None, None, None, None, None, None, # alphas, tp_list, config, alpha_w
)

_LOSS_AXES = (
    0, 0, 0, 0,                         # mean_voltage, sum_Er, C_soma, A_dend
    None,                               # B_list (shared, fixed)
    0,                                  # target_smoothed
    None, None, None,                   # T, loss_temperature, loss_count_bias
)

_fwd_single = jit(_forward_and_accum)
_pred_single = jit(_predict_only)
_loss_single = jit(_loss_and_grads)
_apply = jit(_apply_grads)
_adam = jit(_adam_apply)

_fwd_batch = jit(vmap(_forward_and_accum, in_axes=_FWD_AXES))
_pred_batch = jit(vmap(_predict_only, in_axes=_PRED_AXES))
_loss_batch = jit(vmap(_loss_and_grads, in_axes=_LOSS_AXES))
_act_batch = jit(vmap(_activity, in_axes=_PRED_AXES))


# ══════════════════════════════════════════════════════════════════════
#  Network class — ties everything together
# ══════════════════════════════════════════════════════════════════════

class Network:
    def __init__(
        self,
        key: jnp.ndarray,
        n_inputs: int,
        hidden_sizes,
        n_outputs: int,
        config: NeuronConfig,
        optimizer: str = "sgd",
        beta1: float = 0.9,
        beta2: float = 0.999,
        adam_eps: float = 1e-8,
        dropout_rate: float = 0.0,
        weight_decay: float = 0.0,
        fb_dend_c: float = None,
    ):
        self.n_inputs = n_inputs
        self.hidden_sizes = list(hidden_sizes)
        self.n_outputs = n_outputs
        self.config = config
        self.optimizer = optimizer
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        # Straight-through gain on the dendrite route in the transported feedback.
        self.fb_dend_c = config.gamma * 0.5 if fb_dend_c is None else fb_dend_c
        self.n_layers = len(self.hidden_sizes)

        # Chain of hidden two-compartment layers: dims [n_inputs, *hidden_sizes].
        dims = [n_inputs] + self.hidden_sizes
        keys = random.split(key, self.n_layers + 2)
        self.hidden = [
            TwoCompNeuron(keys[l], self.hidden_sizes[l], dims[l], config)
            for l in range(self.n_layers)
        ]
        self.readout = LINeuron(keys[self.n_layers], n_outputs, self.hidden_sizes[-1], config)
        self.rng_key = keys[self.n_layers + 1]

        if optimizer == "adam":
            self.beta1 = beta1
            self.beta2 = beta2
            self.adam_eps = adam_eps
            self.adam_step = jnp.array(0, dtype=jnp.int32)
            self.m = [jnp.zeros_like(w) for w in self._weights_flat()]
            self.v = [jnp.zeros_like(w) for w in self._weights_flat()]

    # ── Weight views ──
    #
    # Two shapes: a pytree (dict of lists) for the scan/forward core, and a flat
    # ordered list [dend_0, soma_0, .., dend_{L-1}, soma_{L-1}, readout] for the
    # optimizer. Both are rebuilt from the layer objects each call so they always
    # reflect the current weights.

    def _weights(self):
        return {
            "dend": [h.w_dend for h in self.hidden],
            "soma": [h.w_soma for h in self.hidden],
            "readout": self.readout.w,
        }

    def _weights_flat(self):
        flat = []
        for h in self.hidden:
            flat.append(h.w_dend)
            flat.append(h.w_soma)
        flat.append(self.readout.w)
        return flat

    def _set_weights_flat(self, flat):
        i = 0
        for h in self.hidden:
            h.w_dend = flat[i]; i += 1
            h.w_soma = flat[i]; i += 1
        self.readout.w = flat[i]

    def _grads_flat(self, g_dend_list, g_soma_list, g_readout):
        flat = []
        for l in range(self.n_layers):
            flat.append(g_dend_list[l])
            flat.append(g_soma_list[l])
        flat.append(g_readout)
        return flat

    def _feedback_list(self):
        # Weight-transported symmetric feedback (STE). Each hidden layer receives
        # the output error through the static chain
        #     B_l = W_out · Π_{k>l} (W_soma_k + c·W_dend_k),
        # a reverse cumulative product of the forward weights above it. The last
        # hidden layer uses the live readout weights W_out verbatim; the dendrite
        # route is collapsed to a single straight-through gain c = self.fb_dend_c.
        # For L=1 this returns [W_out] (all-symmetric).
        c = self.fb_dend_c
        B = [None] * self.n_layers
        B[self.n_layers - 1] = self.readout.w                  # (J, N_{L-1})
        for l in range(self.n_layers - 2, -1, -1):
            nxt = self.hidden[l + 1]
            M = nxt.w_soma + c * nxt.w_dend                    # (N_{l+1}, N_l)
            B[l] = B[l + 1] @ M                                # (J, N_l)
        return B

    def _tp_list(self):
        return [h.T_p for h in self.hidden]

    def _params(self):
        # alpha_s/alpha_d/alpha_w are identical across layers (global taus).
        h0 = self.hidden[0]
        return (h0.alpha_s, h0.alpha_d, self.readout.alpha_m,
                self._tp_list(), self.config, h0.alpha_w)

    # ── Zero-initialized carries ──

    def _h_carry(self, B=None):
        dims = [self.n_inputs] + self.hidden_sizes
        carries = []
        for l in range(self.n_layers):
            n, k = self.hidden_sizes[l], dims[l]
            s = (B, n) if B else (n,)
            sk = (B, k) if B else (k,)
            snk = (B, n, k) if B else (n, k)
            carries.append((
                jnp.zeros(s), jnp.zeros(s),
                jnp.zeros(s, dtype=jnp.int32), jnp.zeros(s, dtype=jnp.int32),
                jnp.zeros(s), jnp.zeros(sk),
                jnp.zeros(snk), jnp.zeros(snk),
                jnp.zeros(s),
            ))
        return carries

    def _r_carry(self, B=None):
        j, n = self.n_outputs, self.hidden_sizes[-1]
        sj = (B, j) if B else (j,)
        sn = (B, n) if B else (n,)
        return (jnp.zeros(sj), jnp.zeros(sj), jnp.zeros(sn))

    def _acc_zeros(self, B=None):
        """Per-layer (N_l, N_{l-1}) accumulator lists + readout-elig sum vector."""
        dims = [self.n_inputs] + self.hidden_sizes
        C_soma, A_dend = [], []
        for l in range(self.n_layers):
            n, k = self.hidden_sizes[l], dims[l]
            shape = (B, n, k) if B else (n, k)
            C_soma.append(jnp.zeros(shape))
            A_dend.append(jnp.zeros(shape))
        er_shape = (B, self.hidden_sizes[-1]) if B else (self.hidden_sizes[-1],)
        sum_Er = jnp.zeros(er_shape)
        return C_soma, A_dend, sum_Er

    def _smooth_targets(self, targets):
        cfg = self.config
        one_hot = jnp.eye(self.n_outputs)[targets]
        return one_hot * (1 - cfg.loss_label_smoothing) + cfg.loss_label_smoothing / self.n_outputs

    def _update_weights(self, g_dend_list, g_soma_list, g_readout, lr, clip_value):
        grads_flat = self._grads_flat(g_dend_list, g_soma_list, g_readout)
        if self.optimizer == "adam":
            self.adam_step = self.adam_step + 1
            new_w, self.m, self.v = _adam(
                self._weights_flat(), grads_flat, self.m, self.v,
                self.adam_step, lr, self.beta1, self.beta2, self.adam_eps,
                clip_value, self.weight_decay,
            )
        else:
            new_w = _apply(
                self._weights_flat(), grads_flat, lr, clip_value, self.weight_decay,
            )
        self._set_weights_flat(new_w)

    def _next_key(self):
        self.rng_key, subkey = random.split(self.rng_key)
        return subkey

    def _grad_norms(self, g_dend_list, g_soma_list, g_readout):
        norms = {"readout": float(jnp.linalg.norm(g_readout))}
        for l in range(self.n_layers):
            norms[f"dend{l}"] = float(jnp.linalg.norm(g_dend_list[l]))
            norms[f"soma{l}"] = float(jnp.linalg.norm(g_soma_list[l]))
        return norms

    # ── Single-sample API ──

    def train_step(self, x_input, target, lr=1e-3, clip_value=1.0):
        T = x_input.shape[0]
        C0, A0, Er0 = self._acc_zeros()

        mean_v, sum_Er, C_soma, A_dend = _fwd_single(
            x_input, self._weights(), *self._params(),
            self._h_carry(), self._r_carry(), C0, A0, Er0,
            self._next_key(), self.dropout_rate,
        )

        loss, pred, g_r, g_s_list, g_d_list = _loss_single(
            mean_v, sum_Er, C_soma, A_dend, self._feedback_list(),
            self._smooth_targets(target), T,
            self.config.loss_temperature, self.config.loss_count_bias,
        )

        gnorms = self._grad_norms(g_d_list, g_s_list, g_r)
        self._update_weights(g_d_list, g_s_list, g_r, lr, clip_value)
        return float(loss), int(pred), gnorms

    def predict(self, x_input):
        mean_v = _pred_single(x_input, self._weights(), *self._params())
        return int(jnp.argmax(mean_v))

    # ── Batched API ──

    def batch_train_step(self, x_batch, targets, lr=1e-3, clip_value=1.0):
        B = x_batch.shape[0]
        T = x_batch.shape[1]
        batch_keys = random.split(self._next_key(), B)
        C0, A0, Er0 = self._acc_zeros(B)

        mean_v, sum_Er, C_soma, A_dend = _fwd_batch(
            x_batch, self._weights(), *self._params(),
            self._h_carry(B), self._r_carry(B), C0, A0, Er0,
            batch_keys, self.dropout_rate,
        )

        losses, preds, g_r, g_s_list, g_d_list = _loss_batch(
            mean_v, sum_Er, C_soma, A_dend, self._feedback_list(),
            self._smooth_targets(targets), T,
            self.config.loss_temperature, self.config.loss_count_bias,
        )

        g_r_avg = jnp.mean(g_r, axis=0)
        g_s_avg = [jnp.mean(g, axis=0) for g in g_s_list]
        g_d_avg = [jnp.mean(g, axis=0) for g in g_d_list]

        gnorms = self._grad_norms(g_d_avg, g_s_avg, g_r_avg)
        self._update_weights(g_d_avg, g_s_avg, g_r_avg, lr, clip_value)
        return float(jnp.mean(losses)), preds, gnorms

    def batch_predict(self, x_batch):
        mean_v = _pred_batch(x_batch, self._weights(), *self._params())
        return jnp.argmax(mean_v, axis=1)

    def activity(self, x_batch):
        """Mean firing rate per hidden layer over a batch (no dropout)."""
        rates = _act_batch(x_batch, self._weights(), *self._params())
        return {f"hidden{l}": float(jnp.mean(rates[l])) for l in range(self.n_layers)}

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, path, extra=None):
        meta = {
            "arch": {
                "n_inputs": int(self.n_inputs),
                "hidden_sizes": [int(n) for n in self.hidden_sizes],
                "n_outputs": int(self.n_outputs),
            },
            "build": {
                "optimizer": self.optimizer,
                "dropout_rate": float(self.dropout_rate),
                "weight_decay": float(self.weight_decay),
                "fb_dend_c": float(self.fb_dend_c),
            },
            "config": dataclasses.asdict(self.config),
            "extra": extra or {},
        }
        if self.optimizer == "adam":
            meta["build"].update(
                beta1=float(self.beta1),
                beta2=float(self.beta2),
                adam_eps=float(self.adam_eps),
            )

        arrays = {"w_readout": np.asarray(self.readout.w)}
        for l, h in enumerate(self.hidden):
            arrays[f"w_dend_{l}"] = np.asarray(h.w_dend)
            arrays[f"w_soma_{l}"] = np.asarray(h.w_soma)
            arrays[f"T_p_{l}"] = np.asarray(h.T_p)
        # Shared scalar dynamics (identical across layers).
        h0 = self.hidden[0]
        arrays["alpha_d"] = np.asarray(h0.alpha_d)
        arrays["alpha_s"] = np.asarray(h0.alpha_s)
        arrays["alpha_w"] = np.asarray(h0.alpha_w)
        arrays["alpha_m"] = np.asarray(self.readout.alpha_m)

        if self.optimizer == "adam":
            arrays["adam_step"] = np.asarray(self.adam_step)
            for i, (m, v) in enumerate(zip(self.m, self.v)):
                arrays[f"m_{i}"] = np.asarray(m)
                arrays[f"v_{i}"] = np.asarray(v)

        arrays["meta"] = np.array(json.dumps(meta))
        np.savez_compressed(path, **arrays)
        return path

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        arch = meta["arch"]
        build = meta["build"]
        config = NeuronConfig(**meta["config"])

        net = cls(
            random.PRNGKey(0),
            arch["n_inputs"], arch["hidden_sizes"], arch["n_outputs"], config,
            optimizer=build["optimizer"],
            beta1=build.get("beta1", 0.9),
            beta2=build.get("beta2", 0.999),
            adam_eps=build.get("adam_eps", 1e-8),
            dropout_rate=build["dropout_rate"],
            weight_decay=build["weight_decay"],
            fb_dend_c=build.get("fb_dend_c", None),
        )

        def arr(k):
            return jnp.asarray(data[k])

        net.readout.w = arr("w_readout")
        for l, h in enumerate(net.hidden):
            h.w_dend = arr(f"w_dend_{l}")
            h.w_soma = arr(f"w_soma_{l}")
            h.T_p = arr(f"T_p_{l}").astype(jnp.int32)
            h.alpha_d = arr("alpha_d")
            h.alpha_s = arr("alpha_s")
            h.alpha_w = arr("alpha_w")
        net.readout.alpha_m = arr("alpha_m")

        if build["optimizer"] == "adam" and "adam_step" in data.files:
            net.adam_step = arr("adam_step").astype(jnp.int32)
            n_tensors = 2 * len(net.hidden) + 1
            net.m = [arr(f"m_{i}") for i in range(n_tensors)]
            net.v = [arr(f"v_{i}") for i in range(n_tensors)]

        net.meta = meta
        return net
