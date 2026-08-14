import dataclasses
import json

import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit, lax, vmap, value_and_grad, custom_vjp

from config import NeuronConfig, surrogate_sigma
from two_comp_neuron import TwoCompNeuron
from lif_neuron import LINeuron


# Guards log(0) in the cross-entropy. Small enough not to bias the loss.
_LOG_EPS = 1e-8


# ══════════════════════════════════════════════════════════════════════
#  Core functions — each processes ONE sample.
#
#  FULL SURROGATE BPTT via jax.grad — the "correct-gradient" counterpart to
#  the hand-derived, pruned rule in ../BPTT_fix. Same network, same loss; the
#  ONLY difference is how the hidden-layer gradients are obtained.
#
#  Here we write the whole forward (spike dynamics -> readout -> softmax CE)
#  as one differentiable scalar loss(weights) and let reverse-mode autodiff
#  compute the EXACT gradient. Nothing is pruned:
#    • error flows between layers through BOTH the soma AND the dendrite edges,
#    • the alpha_s / alpha_d / adaptation recurrences and the v-reset are all
#      backpropagated through time (no e-prop truncation),
#    • the dendrite path keeps its s'_s·γ·s'_d factor automatically.
#  The two hard Heaviside thresholds (soma spike o, plateau h) are the only
#  non-differentiable steps; each gets a surrogate backward via `_spike`
#  (custom_vjp) using the same 1/(1+β|x|)² slope as the hand rule. Integer
#  routing (t', the plateau-timing gate, spike-based branch selects) carries
#  zero gradient, exactly as in exact BPTT-with-surrogate.
#
#  Costs O(T) memory (autodiff tapes the whole scan) vs. the hand rule's O(1);
#  that is the price of the exact gradient and is fine as a reference.
#
#  Sign convention: the optimizer does w <- w + lr·g, so we return the ASCENT
#  direction g = -∇_w loss (a straight negation of the autodiff gradient).
#  Unlike the hand rule there is NO _SOMA_GRAD_SCALE and NO dropped 1/temp —
#  this is the un-rescaled true gradient, so the learning rate may need
#  retuning relative to BPTT_fix.
#
#  weights pytree:  {"dend": [w_dend_0..], "soma": [w_soma_0..],
#                    "readout": w_readout}
#  tp_list:         [T_p_0, .., T_p_{L-1}]  (per-layer, per-neuron plateau)
# ══════════════════════════════════════════════════════════════════════


@custom_vjp
def _spike(x, beta):
    """Heaviside Θ(x) on the forward pass, surrogate slope on the backward.

    Forward value is exactly the hard threshold used by TwoCompNeuron, so the
    predictions match the int-threshold forward bit-for-bit. beta rides along
    only to parameterise the surrogate and receives a zero cotangent.
    """
    return jnp.where(x >= 0.0, 1.0, 0.0)


def _spike_fwd(x, beta):
    return jnp.where(x >= 0.0, 1.0, 0.0), (x, beta)


def _spike_bwd(res, g):
    x, beta = res
    return (g * surrogate_sigma(x, beta), jnp.zeros_like(beta))


_spike.defvjp(_spike_fwd, _spike_bwd)


def _tc_step(state, dend_in, soma_in, t, alpha_s, alpha_d, T_p, config, alpha_w):
    """Differentiable mirror of TwoCompNeuron.forward_step.

    Identical forward values to two_comp_neuron.forward_step (so _predict_only,
    which still uses that int-threshold version, agrees), but the two hard
    thresholds go through `_spike` so autodiff sees a surrogate. h is kept as a
    float in {0,1}; integer comparisons on it (h_prev == 0/1) are exact and
    naturally carry no gradient, matching exact BPTT's discrete spike routing.
    """
    mu_prev, v_prev, h_prev, tp_prev, matp_prev, w_prev = state

    t_prime = jnp.where(t == 0, 0, jnp.where(h_prev == 1, tp_prev, t))
    mu = jnp.where(t > 0, alpha_d * mu_prev + (1 - h_prev) * dend_in, dend_in)
    mu_at_tprime = jnp.where(h_prev == 0, mu, matp_prev)

    plateau_duration = t - t_prime
    timing_ok = ((plateau_duration <= T_p) & (plateau_duration >= 0)).astype(mu.dtype)
    # h = 1 iff plateau voltage crosses AND the timing window holds. timing_ok
    # depends only on integers -> zero gradient; the μ crossing carries the
    # dendritic surrogate.
    h = timing_ok * _spike(mu_at_tprime - config.mu_th, config.beta_d)

    v_pre = jnp.where(t > 0, alpha_s * v_prev + soma_in - w_prev, soma_in)
    o = _spike(v_pre + config.gamma * h - config.v_th, config.beta_s)
    v = v_pre * (1 - o)
    w = alpha_w * w_prev + (1 - alpha_w) * config.a_adapt * v_pre + config.b_adapt * o

    new_state = (mu, v, h, t_prime, mu_at_tprime, w)
    return new_state, o


def _init_tc_state(n):
    """Zero carry for _tc_step: (mu, v, h, t_prime, mu_at_tprime, w_adapt).

    No eligibility slots — autodiff handles temporal credit — and t_prime is the
    only integer field."""
    return (
        jnp.zeros(n), jnp.zeros(n), jnp.zeros(n),
        jnp.zeros(n, dtype=jnp.int32), jnp.zeros(n), jnp.zeros(n),
    )


def _loss_and_meanv(
    weights, x_input, alpha_s, alpha_d, alpha_m, tp_list, config, alpha_w,
    target_smoothed, loss_temperature, loss_count_bias, rng_key, dropout_rate,
):
    """Single differentiable forward: spikes -> LI readout -> softmax CE.

    Returns (loss, mean_voltage); mean_voltage is aux (not differentiated).
    weights is argument 0 so value_and_grad(argnums=0) differentiates it.
    """
    w_dend = weights["dend"]
    w_soma = weights["soma"]
    w_readout = weights["readout"]
    L = len(w_dend)

    T = x_input.shape[0]
    time_indices = jnp.arange(T, dtype=jnp.int32)
    dropout_keys = random.split(rng_key, T * L).reshape(T, L, 2)
    dropout_scale = 1.0 / (1.0 - dropout_rate)

    h_states0 = [_init_tc_state(w_dend[l].shape[0]) for l in range(L)]
    n_out = w_readout.shape[0]
    r_v0 = jnp.zeros(n_out)      # readout membrane
    r_sum0 = jnp.zeros(n_out)    # accumulated voltage (for the mean)

    def step(carry, inputs):
        h_states, r_v, r_sum = carry
        x_t, t, drop_keys_t = inputs
        x_t = x_t.astype(jnp.float64)

        new_h_states = list(h_states)
        o_prev = None
        for l in range(L):
            presyn = x_t if l == 0 else o_prev
            dend_in_l = presyn @ w_dend[l].T
            soma_in_l = presyn @ w_soma[l].T
            new_h_states[l], o_l = _tc_step(
                h_states[l], dend_in_l, soma_in_l, t,
                alpha_s, alpha_d, tp_list[l], config, alpha_w,
            )
            mask = random.bernoulli(
                drop_keys_t[l], 1.0 - dropout_rate, o_l.shape,
            ).astype(jnp.float64)
            o_prev = o_l * mask * dropout_scale  # downstream sees the dropped spike

        r_v = alpha_m * r_v + o_prev @ w_readout.T  # LI readout: no threshold/reset
        r_sum = r_sum + r_v
        return (new_h_states, r_v, r_sum), None

    (_, _, r_sum_f), _ = lax.scan(
        step, (h_states0, r_v0, r_sum0), (x_input, time_indices, dropout_keys),
    )
    mean_voltage = r_sum_f / T

    scaled_logits = mean_voltage / loss_temperature + loss_count_bias
    probs = jnp.exp(scaled_logits - jnp.max(scaled_logits))
    probs = probs / jnp.sum(probs)
    loss = -jnp.sum(target_smoothed * jnp.log(probs + _LOG_EPS))
    return loss, mean_voltage


def _forward_backward(
    x_input, weights, alpha_s, alpha_d, alpha_m, tp_list, config, alpha_w,
    h_carry_init, r_carry_init, grad_soma_init, grad_dend_init, sum_Er_init,
    target_smoothed, loss_temperature, loss_count_bias,
    rng_key, dropout_rate,
):
    """Full surrogate-BPTT gradient for one sample, via reverse-mode autodiff.

    Drop-in for the hand-derived rule: identical signature and returns. The
    *_init carry/accumulator arguments are unused here (autodiff builds its own
    tape) but kept so the vmap in_axes and the Network call sites stay unchanged.

    Returns: mean_voltage (J,), loss, prediction,
             grad_readout, grad_soma_list (each N_l,N_{l-1}), grad_dend_list.
    """
    (loss, mean_voltage), grads = value_and_grad(_loss_and_meanv, has_aux=True)(
        weights, x_input, alpha_s, alpha_d, alpha_m, tp_list, config, alpha_w,
        target_smoothed, loss_temperature, loss_count_bias, rng_key, dropout_rate,
    )
    prediction = jnp.argmax(mean_voltage)

    # Optimizer ascends (w <- w + lr·g); return -∇loss.
    grad_readout = -grads["readout"]
    grad_soma_f = [-g for g in grads["soma"]]
    grad_dend_f = [-g for g in grads["dend"]]

    return mean_voltage, loss, prediction, grad_readout, grad_soma_f, grad_dend_f


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
    0, 0, 0, 0, 0,                      # h/r carry inits, grad_soma, grad_dend, sum_Er
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
        return (jnp.zeros(sj), jnp.zeros(sj), jnp.zeros(sn))

    def _acc_zeros(self, B=None):
        """Per-layer (N_l, N_{l-1}) gradient accumulators + readout-elig sum vector."""
        dims = [self.n_inputs] + self.hidden_sizes
        grad_soma, grad_dend = [], []
        for l in range(self.n_layers):
            n, k = self.hidden_sizes[l], dims[l]
            shape = (B, n, k) if B else (n, k)
            grad_soma.append(jnp.zeros(shape))
            grad_dend.append(jnp.zeros(shape))
        er_shape = (B, self.hidden_sizes[-1]) if B else (self.hidden_sizes[-1],)
        sum_Er = jnp.zeros(er_shape)
        return grad_soma, grad_dend, sum_Er

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
        gs0, gd0, Er0 = self._acc_zeros()

        mean_v, loss, pred, g_r, g_s_list, g_d_list = _fb_single(
            x_input, self._weights(), *self._params(),
            self._h_carry(), self._r_carry(), gs0, gd0, Er0,
            self._smooth_targets(target),
            self.config.loss_temperature, self.config.loss_count_bias,
            self._next_key(), self.dropout_rate,
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
        batch_keys = random.split(self._next_key(), B)
        gs0, gd0, Er0 = self._acc_zeros(B)

        mean_v, losses, preds, g_r, g_s_list, g_d_list = _fb_batch(
            x_batch, self._weights(), *self._params(),
            self._h_carry(B), self._r_carry(B), gs0, gd0, Er0,
            self._smooth_targets(targets),
            self.config.loss_temperature, self.config.loss_count_bias,
            batch_keys, self.dropout_rate,
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
