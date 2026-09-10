import dataclasses
import json

import numpy as np
import jax.numpy as jnp
from jax import random, jit, lax, vmap

from config import NeuronConfig, surrogate_sigma
from two_comp_neuron import TwoCompNeuron
from lif_neuron import LINeuron


# Extra divisor on the somatic gradient only, balancing its magnitude against the
# dendritic gradient (the somatic term runs ~8x larger). Empirical, not derived —
# carried over from the single-layer rule and applied per hidden layer. Set to 1.0
# for the un-scaled gradient.
_SOMA_GRAD_SCALE = 8.0

# Guards log(0) in the cross-entropy. Small enough not to bias the loss.
_LOG_EPS = 1e-8


# ══════════════════════════════════════════════════════════════════════
#  Core functions — each processes ONE sample.
#
#  t'-PRUNED SOMA-BACKBONE BPTT with an arbitrary number of hidden layers.
#  Every hidden layer is a TwoCompNeuron; layer 0 reads the input spikes,
#  layer l>0 reads the (dropped) spikes of layer l-1, and the last hidden
#  layer feeds the leaky-integrator readout.
#
#  Exact BPTT would credit a spike o^(l)[t] through TWO edges into the
#  previous layer: a same-time SOMA edge (o^(l-1)[t] -> I_s -> v~ -> o^(l)[t])
#  and a DENDRITE edge that jumps to the plateau-onset time t' (o^(l-1)[t']
#  -> mu -> h -> o^(l)[t]). Following both across L layers yields a binary
#  tree of 2^L root-to-leaf paths, each with an independent t' time-shift.
#
#  We PRUNE the dendrite edge from the *inter-layer* Jacobian only. Error is
#  transported between layers through the real transposed soma weights
#  W_s^(l)^T (same time index, no branching, no t'-scatter). The dendrite is
#  descended into LOCALLY, at the layer whose W_d is being trained, keeping
#  that layer's own t' plateau path. The result is a single somatic backbone
#  readout -> o^(L) -> .. -> o^(1) with one local dendrite tap per rung.
#
#  Time credit is handled e-prop style: the per-layer eligibility traces
#  (E_soma, dmu_dw_at_tprime) carry the within-layer temporal credit; there is
#  no backprop through the alpha_s membrane recurrence or the reset. Because
#  pruning removes the cross-layer time coupling and e-prop removes the
#  within-layer time adjoint, the learning signal is INSTANTANEOUS in t — the
#  only quantity that comes "from the future" is the end-of-sequence error e
#  (the pure integrator gives a constant, time-independent readout weight; see
#  below). So the whole rule is two
#  forward passes: pass 1 gets e, pass 2 re-runs the forward and accumulates
#  the hand-derived gradients. No autodiff. DFA-level memory (no per-time
#  tensors are stored).
#
#  Readout is a PURE INTEGRATOR (alpha_m = 1, no leak) and the logit is its
#  FINAL membrane potential U[T] = sum_k W_r o^(L)[k]. Every spike lands in U[T]
#  with the same weight W_r, so the readout credit is constant in time:
#  rho[t] ≡ 1 — no geometric-series weighting, no 1/T.
#
#  Per-time backbone (ascent-direction, matching w <- w + lr*g):
#    e           = target - softmax(U/temp + bias)
#    do^(L)[t]   = (W_r^T e) * mask^L[t]                       # rho[t] ≡ 1; dropped o^L
#    Lv^(l)[t]   = do^(l)[t] * s'_s^(l)[t]                     # = d(v~)
#    do^(l-1)[t] = (W_s^(l)^T Lv^(l)[t]) * mask^(l-1)[t]
#  Grads:
#    grad W_s^(l) = sum_t  Lv^(l)[t]                ⊗ E_s^(l)[t]   / _SOMA_GRAD_SCALE
#    grad W_d^(l) = sum_t (Lv^(l)[t]·γ·s'_d^(l)[t]) ⊗ P̂^(l)[t]
#    grad W_r     = e ⊗ E_r[T]                                     (E_r[T] = sum_k o^(L)[k])
#
#  weights pytree:  {"dend": [w_dend_0..], "soma": [w_soma_0..],
#                    "readout": w_readout}
#  tp_list:         [T_p_0, .., T_p_{L-1}]  (per-layer, per-neuron plateau)
# ══════════════════════════════════════════════════════════════════════


def _forward_backward(
    x_input, weights, alpha_s, alpha_d, alpha_m, tp_list, config, alpha_w,
    h_carry_init, r_carry_init, grad_soma_init, grad_dend_init,
    target_smoothed, loss_temperature, loss_count_bias,
    rng_key, dropout_rate,
):
    """Two-pass soma-backbone BPTT for one sample.

    Pass 1: forward (with dropout) -> final readout voltage U[T], elig E_r[T].
    Between: e = target - softmax(...), loss, prediction, exact readout grad.
    Pass 2: re-run the forward with the SAME dropout masks; at each step run the
            top-down soma backbone and accumulate the hidden-layer gradients.

    Returns: logits U[T] (J,), loss, prediction,
             grad_readout, grad_soma_list (each N_l,N_{l-1}), grad_dend_list.
    """
    w_dend = weights["dend"]
    w_soma = weights["soma"]
    w_readout = weights["readout"]
    L = len(w_dend)

    T = x_input.shape[0]
    time_indices = jnp.arange(T, dtype=jnp.int32)
    # One dropout subkey per (timestep, layer). Shared between both passes so the
    # spikes/eligibilities of pass 2 match the mean-voltage of pass 1.
    dropout_keys = random.split(rng_key, T * L).reshape(T, L, 2)
    dropout_scale = 1.0 / (1.0 - dropout_rate)

    # Only layer 0's synaptic drive can be precomputed (fixed x_input); deeper
    # layers depend on spikes produced inside the scan.
    dend_in_0 = x_input @ w_dend[0].T
    soma_in_0 = x_input @ w_soma[0].T

    def _layer_forward(h_state, dend_in_l, soma_in_l, presyn, t, tp_l):
        """One layer's forward step + eligibility update; returns everything the
        backbone needs at this (t, l)."""
        h_carry, o_l, v_pre_l, h_l, h_prev_l, mu_at_tp_l = TwoCompNeuron.forward_step(
            h_state, dend_in_l, soma_in_l, t,
            alpha_s, alpha_d, tp_l, config, alpha_w,
        )
        # Somatic surrogate s'_s at two betas: beta_s drives the soma backbone
        # (grad W_soma + inter-layer transport); beta_s_dend is the s'_s factor in
        # the dendrite tap (∂o/∂h = γ·s'_s). Equal betas => identical to before.
        soma_arg = v_pre_l + config.gamma * h_l - config.v_th
        sp_l = surrogate_sigma(soma_arg, config.beta_s)
        sp_dend_l = surrogate_sigma(soma_arg, config.beta_s_dend)
        hp_l = surrogate_sigma(mu_at_tp_l - config.mu_th, config.beta_d)

        (mu_c, v_c, h_c, tp_c, matp_c, E_soma_c,
         dmu_c, dmu_atp_c, w_c) = h_carry
        E_soma_new = TwoCompNeuron.update_somatic_eligibility(E_soma_c, presyn, alpha_s)
        dmu_new, dmu_atp_new = TwoCompNeuron.update_dendritic_eligibility(
            dmu_c, dmu_atp_c, presyn, h_prev_l, alpha_d,
        )
        new_state = (mu_c, v_c, h_c, tp_c, matp_c, E_soma_new, dmu_new, dmu_atp_new, w_c)
        return new_state, o_l, sp_l, sp_dend_l, hp_l, E_soma_new, dmu_atp_new

    # ── Pass 1: forward with dropout → final readout voltage + eligibility ─
    def step1(carry, inputs):
        h_states, r_carry = carry
        dend_in0_t, soma_in0_t, x_t, t, drop_keys_t = inputs
        x_t = x_t.astype(jnp.float64)

        new_h_states = list(h_states)
        o_prev = None
        for l in range(L):
            if l == 0:
                dend_in_l, soma_in_l, presyn = dend_in0_t, soma_in0_t, x_t
            else:
                dend_in_l = o_prev @ w_dend[l].T
                soma_in_l = o_prev @ w_soma[l].T
                presyn = o_prev
            new_h_states[l], o_l, *_ = _layer_forward(
                h_states[l], dend_in_l, soma_in_l, presyn, t, tp_list[l],
            )
            mask = random.bernoulli(
                drop_keys_t[l], 1.0 - dropout_rate, o_l.shape,
            ).astype(jnp.float64)
            o_prev = o_l.astype(jnp.float64) * mask * dropout_scale

        r_carry, _, _ = LINeuron.forward_step(r_carry, o_prev, w_readout, alpha_m)
        return (new_h_states, r_carry), None

    scan_inputs = (dend_in_0, soma_in_0, x_input, time_indices, dropout_keys)
    (_, r_carry_f), _ = lax.scan(
        step1, (h_carry_init, r_carry_init), scan_inputs,
    )
    logits_v = r_carry_f[0]   # final integrator voltage U[T]
    E_r_final = r_carry_f[1]  # E_r[T] = sum_k o^(L)[k], the readout eligibility

    # ── Loss, error signal, exact readout gradient ────────────────────────
    scaled_logits = logits_v / loss_temperature + loss_count_bias
    probs = jnp.exp(scaled_logits - jnp.max(scaled_logits))
    probs = probs / jnp.sum(probs)
    prediction = jnp.argmax(logits_v)
    loss = -jnp.sum(target_smoothed * jnp.log(probs + _LOG_EPS))
    e = target_smoothed - probs  # (J,), ascent-direction error

    grad_readout = e[:, None] * E_r_final[None, :]  # exact: e ⊗ E_r[T]
    readout_seed = w_readout.T @ e  # (N_L,), the top of the soma backbone

    # ── Pass 2: re-run forward (same dropout) → accumulate hidden grads ────
    def step2(carry, inputs):
        h_states, grad_soma, grad_dend = carry
        dend_in0_t, soma_in0_t, x_t, t, drop_keys_t = inputs
        x_t = x_t.astype(jnp.float64)

        new_h_states = list(h_states)
        sps, sps_dend, hps, Es, Phat, masks = [], [], [], [], [], []
        o_prev = None
        for l in range(L):
            if l == 0:
                dend_in_l, soma_in_l, presyn = dend_in0_t, soma_in0_t, x_t
            else:
                dend_in_l = o_prev @ w_dend[l].T
                soma_in_l = o_prev @ w_soma[l].T
                presyn = o_prev
            (new_h_states[l], o_l, sp_l, sp_dend_l, hp_l,
             E_soma_new, dmu_atp_new) = _layer_forward(
                h_states[l], dend_in_l, soma_in_l, presyn, t, tp_list[l],
            )
            mask = random.bernoulli(
                drop_keys_t[l], 1.0 - dropout_rate, o_l.shape,
            ).astype(jnp.float64) * dropout_scale
            sps.append(sp_l); sps_dend.append(sp_dend_l); hps.append(hp_l)
            Es.append(E_soma_new); Phat.append(dmu_atp_new); masks.append(mask)
            o_prev = o_l.astype(jnp.float64) * mask

        # Top-down soma backbone (instantaneous in t).
        new_grad_soma = list(grad_soma)
        new_grad_dend = list(grad_dend)
        Lv = None
        for l in reversed(range(L)):
            if l == L - 1:
                delta_o = readout_seed          # rho[t] ≡ 1 (pure integrator)
            else:
                delta_o = w_soma[l + 1].T @ Lv  # W_s^(l+1)^T · d(v~)^(l+1)
            delta_o = delta_o * masks[l]        # downstream sees the dropped spike
            Lv = delta_o * sps[l]               # d(v~)^(l), soma pathway (beta_s)
            new_grad_soma[l] = new_grad_soma[l] + (Lv[:, None] * Es[l][None, :])
            # The dendrite reaches the output ONLY through the soma spike
            # (o = Θ(v_pre + γ·h − v_th), so ∂o/∂h = s'_s·γ). Its gradient must
            # therefore carry the somatic pseudo-derivative — here evaluated at
            # beta_s_dend, branching off delta_o (not the beta_s Lv above).
            Lv_dend = delta_o * sps_dend[l]     # dendritic pathway s'_s (beta_s_dend)
            new_grad_dend[l] = new_grad_dend[l] + (
                (Lv_dend * config.gamma * hps[l])[:, None] * Phat[l]
            )
        return (new_h_states, new_grad_soma, new_grad_dend), None

    scan_inputs2 = (dend_in_0, soma_in_0, x_input, time_indices, dropout_keys)
    (_, grad_soma_f, grad_dend_f), _ = lax.scan(
        step2, (h_carry_init, grad_soma_init, grad_dend_init), scan_inputs2,
    )
    grad_soma_f = [g / _SOMA_GRAD_SCALE for g in grad_soma_f]

    return logits_v, loss, prediction, grad_readout, grad_soma_f, grad_dend_f


def _predict_only(
    x_input, weights, alpha_s, alpha_d, alpha_m, tp_list, config, alpha_w,
):
    """Forward pass only — no eligibility bookkeeping. Returns logits U[T] (J,).

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
    r_zeros = (jnp.zeros(n_outputs), jnp.zeros(w_dend[-1].shape[0]))
    (_, r_carry_f), _ = lax.scan(
        step, (h_zeros, r_zeros), (dend_in_0, soma_in_0, x_input, time_indices),
    )
    return r_carry_f[0]  # final integrator voltage U[T]


def _apply_grads(weights_flat, grads_flat, lr, weight_decay):
    """SGD with decoupled weight decay over a flat list of weight tensors.

    w ← w + lr·g − lr·λ·w. For plain SGD the decay term is equivalent to an L2
    penalty (λ/2)·||w||².
    """
    new = []
    for w, g in zip(weights_flat, grads_flat):
        new.append(w + lr * g - lr * weight_decay * w)
    return new


def _adam_apply(
    weights_flat, grads_flat, m_flat, v_flat,
    step, lr, beta1, beta2, eps, weight_decay,
):
    """AdamW-style decoupled weight decay over a flat list of weight tensors."""
    new_w, new_m, new_v = [], [], []
    for w, g, m, v in zip(weights_flat, grads_flat, m_flat, v_flat):
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
#  vmap in_axes: x, carry/accumulator inits, targets and the PRNG key are
#  batched (axis 0); weights, params (alphas/T_p/config), loss scalars and
#  dropout_rate are shared (None). Passing weights/params as pytrees lets a
#  single None cover each whole subtree.
# ══════════════════════════════════════════════════════════════════════

_FB_AXES = (
    0,                                  # x_input
    None,                               # weights (pytree)
    None, None, None,                   # alpha_s, alpha_d, alpha_m
    None, None, None,                   # tp_list, config, alpha_w
    0, 0, 0, 0,                         # h/r carry inits, grad_soma, grad_dend
    0,                                  # target_smoothed
    None, None,                         # loss_temperature, loss_count_bias
    0,                                  # rng_key
    None,                               # dropout_rate
)

_PRED_AXES = (
    0,                                  # x_input
    None,                               # weights
    None, None, None, None, None, None, # alphas, tp_list, config, alpha_w
)

_fb_single = jit(_forward_backward)
_pred_single = jit(_predict_only)
_apply = jit(_apply_grads)
_adam = jit(_adam_apply)

_fb_batch = jit(vmap(_forward_backward, in_axes=_FB_AXES))
_pred_batch = jit(vmap(_predict_only, in_axes=_PRED_AXES))
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
    ):
        self.n_inputs = n_inputs
        self.hidden_sizes = list(hidden_sizes)
        self.n_outputs = n_outputs
        self.config = config
        self.optimizer = optimizer
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
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
        return (jnp.zeros(sj), jnp.zeros(sn))

    def _acc_zeros(self, B=None):
        """Per-layer (N_l, N_{l-1}) gradient accumulators."""
        dims = [self.n_inputs] + self.hidden_sizes
        grad_soma, grad_dend = [], []
        for l in range(self.n_layers):
            n, k = self.hidden_sizes[l], dims[l]
            shape = (B, n, k) if B else (n, k)
            grad_soma.append(jnp.zeros(shape))
            grad_dend.append(jnp.zeros(shape))
        return grad_soma, grad_dend

    def _smooth_targets(self, targets):
        cfg = self.config
        one_hot = jnp.eye(self.n_outputs)[targets]
        return one_hot * (1 - cfg.loss_label_smoothing) + cfg.loss_label_smoothing / self.n_outputs

    def _update_weights(self, g_dend_list, g_soma_list, g_readout, lr):
        grads_flat = self._grads_flat(g_dend_list, g_soma_list, g_readout)
        if self.optimizer == "adam":
            self.adam_step = self.adam_step + 1
            new_w, self.m, self.v = _adam(
                self._weights_flat(), grads_flat, self.m, self.v,
                self.adam_step, lr, self.beta1, self.beta2, self.adam_eps,
                self.weight_decay,
            )
        else:
            new_w = _apply(
                self._weights_flat(), grads_flat, lr, self.weight_decay,
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

    def train_step(self, x_input, target, lr=1e-3):
        gs0, gd0 = self._acc_zeros()

        logits, loss, pred, g_r, g_s_list, g_d_list = _fb_single(
            x_input, self._weights(), *self._params(),
            self._h_carry(), self._r_carry(), gs0, gd0,
            self._smooth_targets(target),
            self.config.loss_temperature, self.config.loss_count_bias,
            self._next_key(), self.dropout_rate,
        )

        gnorms = self._grad_norms(g_d_list, g_s_list, g_r)
        self._update_weights(g_d_list, g_s_list, g_r, lr)
        return float(loss), int(pred), gnorms

    def predict(self, x_input):
        logits = _pred_single(x_input, self._weights(), *self._params())
        return int(jnp.argmax(logits))

    # ── Batched API ──

    def batch_train_step(self, x_batch, targets, lr=1e-3):
        B = x_batch.shape[0]
        batch_keys = random.split(self._next_key(), B)
        gs0, gd0 = self._acc_zeros(B)

        logits, losses, preds, g_r, g_s_list, g_d_list = _fb_batch(
            x_batch, self._weights(), *self._params(),
            self._h_carry(B), self._r_carry(B), gs0, gd0,
            self._smooth_targets(targets),
            self.config.loss_temperature, self.config.loss_count_bias,
            batch_keys, self.dropout_rate,
        )

        g_r_avg = jnp.mean(g_r, axis=0)
        g_s_avg = [jnp.mean(g, axis=0) for g in g_s_list]
        g_d_avg = [jnp.mean(g, axis=0) for g in g_d_list]

        gnorms = self._grad_norms(g_d_avg, g_s_avg, g_r_avg)
        self._update_weights(g_d_avg, g_s_avg, g_r_avg, lr)
        return float(jnp.mean(losses)), preds, gnorms

    def batch_predict(self, x_batch):
        logits = _pred_batch(x_batch, self._weights(), *self._params())
        return jnp.argmax(logits, axis=1)

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
