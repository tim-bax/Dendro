import dataclasses
import json

import numpy as np
import jax.numpy as jnp
from jax import random, jit, lax, vmap

from config import NeuronConfig, surrogate_sigma
from two_comp_neuron import TwoCompNeuron
from lif_neuron import LINeuron


# Guards log(0) in the cross-entropy. Small enough not to bias the loss.
_LOG_EPS = 1e-8


# ══════════════════════════════════════════════════════════════════════
#  Core functions — each processes ONE sample.
#
#  These have no decorators so we can wrap them two ways:
#    jit(fn)            → fast single-sample execution
#    jit(vmap(fn, ...)) → fast batched execution (B samples in parallel)
# ══════════════════════════════════════════════════════════════════════

def _forward_and_accum(
    x_input, w_dend, w_soma, w_readout,
    alpha_s, alpha_d, alpha_m, T_p, config, alpha_w,
    h_carry_init, r_carry_init, A_d_init,
    rng_key, dropout_rate,
):
    """Forward pass + gradient accumulator bookkeeping for one sample.

    x_input:      (T, K)  input spike train
    h_carry_init: 9-tuple of hidden-neuron state zeros (see TwoCompNeuron.init_carry)
    r_carry_init: 3-tuple of readout-neuron state zeros (see LINeuron.init_carry)
    A_d_init:     (J, N, K) dendritic accumulator zeros
    rng_key:      PRNG key for dropout masks
    dropout_rate: fraction of hidden spikes to drop (0.0 = no dropout)

    Returns: mean_voltage (J,), A_readout (J,N), A_soma (J,N,K), A_dend (J,N,K)

    mean_voltage is the readout's membrane voltage averaged over the T timesteps
    — NOT a spike count. The LI readout never spikes, so there is nothing to
    count; classification is argmax over this voltage.
    """
    dend_inputs = x_input @ w_dend.T
    soma_inputs = x_input @ w_soma.T
    T = x_input.shape[0]
    n_hidden = w_dend.shape[0]
    time_indices = jnp.arange(T, dtype=jnp.int32)
    dropout_keys = random.split(rng_key, T)
    dropout_scale = 1.0 / (1.0 - dropout_rate)

    def step(carry, inputs):
        h_carry, r_carry, A_d, F_d = carry
        dend_in, soma_in, x_t, t, drop_key = inputs

        h_carry, h_o, h_v_pre, h_h, h_h_prev, h_mu_at_tp = TwoCompNeuron.forward_step(
            h_carry, dend_in, soma_in, t, alpha_s, alpha_d, T_p, config, alpha_w,
        )
        hidden_o_float = h_o.astype(jnp.float64)

        # Dropout: randomly zero out hidden spikes before the readout sees them.
        # Surviving spikes are scaled by 1/(1-p) so expected value is unchanged.
        # At dropout_rate=0.0 the mask is all-ones and scale is 1.0 (no-op).
        mask = random.bernoulli(drop_key, 1.0 - dropout_rate, (n_hidden,)).astype(jnp.float64)
        hidden_o_float = hidden_o_float * mask * dropout_scale

        r_carry, r_v, r_E = LINeuron.forward_step(
            r_carry, hidden_o_float, w_readout, alpha_m,
        )

        mu_c, v_c, h_c, tp_c, matp_c, E_soma_c, dmu_c, dmu_atp_c, w_c = h_carry
        E_soma_new = TwoCompNeuron.update_somatic_eligibility(
            E_soma_c, x_t.astype(jnp.float64), alpha_s,
        )
        dmu_new, dmu_atp_new = TwoCompNeuron.update_dendritic_eligibility(
            dmu_c, dmu_atp_c, x_t.astype(jnp.float64), h_h_prev, alpha_d,
        )
        h_carry = (mu_c, v_c, h_c, tp_c, matp_c, E_soma_new, dmu_new, dmu_atp_new, w_c)

        # LI readout is linear in its inputs: ∂v_j/∂input = 1, no surrogate needed.
        sp_readout = jnp.ones(w_readout.shape[0])
        soma_dist = h_v_pre + config.gamma * h_h - config.v_th
        # Somatic surrogate for the somatic gradient path (goes into A_soma).
        sp_hidden = surrogate_sigma(soma_dist, config.beta_s)
        # Same somatic spike nonlinearity, but re-evaluated at a separate sharpness
        # for the copy that rides the dendritic path (eta_d carries both surrogates).
        sp_hidden_d = surrogate_sigma(soma_dist, config.beta_s_dend)
        hp_hidden = surrogate_sigma(h_mu_at_tp - config.mu_th, config.beta_d)

        eta = sp_readout[:, None] * w_readout * sp_hidden_d[None, :]
        eta_d = eta * (hp_hidden * config.gamma)[None, :]
        # Per-step dendritic gradient contribution, low-pass filtered with the
        # readout kernel alpha_m and then accumulated: the identity
        #   sum_t B[t]*g[t] = sum_t F_d[t],  F_d[t] = alpha_m*F_d[t-1] + g[t]
        # applies the B[t] readout-spread factor (see below) without ever
        # materialising the (T,J,N,K) per-step tensor.
        g_d = jnp.einsum("ji,ik->jik", eta_d, dmu_atp_new)
        F_d = alpha_m * F_d + g_d
        A_d = A_d + F_d

        new_carry = (h_carry, r_carry, A_d, F_d)
        per_step = (sp_readout, sp_hidden, r_E, E_soma_new)
        return new_carry, per_step

    F_d_init = jnp.zeros_like(A_d_init)
    init_carry = (h_carry_init, r_carry_init, A_d_init, F_d_init)
    scan_inputs = (dend_inputs, soma_inputs, x_input, time_indices, dropout_keys)
    final_carry, per_step_all = lax.scan(step, init_carry, scan_inputs)

    sp_r, sp_h, E_r, E_s = per_step_all
    _, r_carry_f, A_d_f, _ = final_carry
    mean_voltage = r_carry_f[1] / T  # sum_v / T

    # B[t] = sum_{s>=t} alpha_m^{s-t}: the readout leaky integrator's forward spread
    # of a hidden spike at t into the time-averaged logit z (Bellec's low-pass
    # filtering of the eligibility trace with the output kernel). It weights every
    # hidden-gradient time-term. A_readout already carries it implicitly via E_r, so
    # it is applied only to soma/dend (dend: via the F_d running filter in step()).
    B = (1.0 - alpha_m ** (T - time_indices)) / (1.0 - alpha_m)
    A_readout = jnp.einsum("ti,tj->ij", sp_r, E_r)
    C_soma = jnp.einsum("tj,ti,tk->jik", sp_r, sp_h, E_s * B[:, None])
    A_soma = w_readout[:, :, None] * C_soma

    return mean_voltage, A_readout, A_soma, A_d_f


def _predict_only(
    x_input, w_dend, w_soma, w_readout,
    alpha_s, alpha_d, alpha_m, T_p, config, alpha_w,
):
    """Forward pass only — no gradient bookkeeping. Returns mean_voltage (J,).

    Runs the same equations as the training path (TwoCompNeuron.dynamics_step,
    LINeuron.integrate), but carries only the state inference needs: the 6 dynamic
    slots and the readout's two, not the three eligibility traces. Those traces
    include two (n_hidden, n_inputs) matrices that a backward-pass-free run would
    only shuttle through the scan untouched.
    """
    dend_inputs = x_input @ w_dend.T
    soma_inputs = x_input @ w_soma.T
    T = x_input.shape[0]
    n_hidden = w_dend.shape[0]
    n_outputs = w_readout.shape[0]
    time_indices = jnp.arange(T, dtype=jnp.int32)

    def step(carry, inputs):
        h_state, r_v, r_sum_v = carry
        dend_in, soma_in, t = inputs

        h_state, o_h, *_ = TwoCompNeuron.dynamics_step(
            h_state, dend_in, soma_in, t, alpha_s, alpha_d, T_p, config, alpha_w,
        )
        r_v, r_sum_v = LINeuron.integrate(
            r_v, r_sum_v, o_h.astype(jnp.float64), w_readout, alpha_m,
        )
        return (h_state, r_v, r_sum_v), None

    h_zeros = (
        jnp.zeros(n_hidden),                   # mu
        jnp.zeros(n_hidden),                   # v
        jnp.zeros(n_hidden, dtype=jnp.int32),  # h
        jnp.zeros(n_hidden, dtype=jnp.int32),  # t_prime
        jnp.zeros(n_hidden),                   # mu_at_tprime
        jnp.zeros(n_hidden),                   # w (adaptation)
    )
    init = (h_zeros, jnp.zeros(n_outputs), jnp.zeros(n_outputs))
    (_, _, sum_v), _ = lax.scan(step, init, (dend_inputs, soma_inputs, time_indices))
    return sum_v / T  # mean voltage


def _loss_and_grads(
    mean_voltage, A_readout, A_soma, A_dend,
    target_smoothed, T, loss_temperature, loss_count_bias,
):
    """Compute loss and weight gradients for one sample."""
    scaled_logits = mean_voltage / loss_temperature + loss_count_bias
    probs = jnp.exp(scaled_logits - jnp.max(scaled_logits))
    probs = probs / jnp.sum(probs)

    prediction = jnp.argmax(mean_voltage)
    loss = -jnp.sum(target_smoothed * jnp.log(probs + _LOG_EPS))
    global_error = target_smoothed - probs

    # Each gradient is averaged over the T timesteps its accumulator summed over.
    # All three now carry only derived factors — the somatic path's empirical /8
    # rescale was removed once B[t] was added to both hidden gradients, which
    # restores their true relative magnitude.
    grad_readout = (global_error[:, None] * A_readout) / T
    grad_soma = jnp.einsum("j,jik->ik", global_error, A_soma) / T
    grad_dend = jnp.einsum("j,jik->ik", global_error, A_dend) / T

    return loss, prediction, grad_readout, grad_soma, grad_dend


def _apply_grads(
    w_dend, w_soma, w_readout, g_dend, g_soma, g_readout,
    lr, weight_decay,
):
    """SGD with decoupled weight decay (no gradient clipping).

    Weights are updated as
        w ← w + lr·g − lr·λ·w,
    which for plain SGD is mathematically equivalent to adding
    (λ/2)·||w||² to the loss (the L2 penalty).
    """
    return (
        w_dend + lr * g_dend - lr * weight_decay * w_dend,
        w_soma + lr * g_soma - lr * weight_decay * w_soma,
        w_readout + lr * g_readout - lr * weight_decay * w_readout,
    )


def _adam_apply(
    w_d, w_s, w_r,
    g_d, g_s, g_r,
    m_d, m_s, m_r,
    v_d, v_s, v_r,
    step, lr, beta1, beta2, eps, weight_decay,
):
    """AdamW-style decoupled weight decay.

    The data gradient flows through the moment estimates and the
    adaptive 1/√v rescaling, but the λ·w term does not — it is
    subtracted directly from w after the Adam update. This matches
    Loshchilov & Hutter (2017) and avoids the parameter-dependent
    decay strength that arises if λ·w is added to the loss before
    Adam normalisation.
    """
    def update_one(w, g, m, v):
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2
        m_hat = m / (1 - beta1 ** step)
        v_hat = v / (1 - beta2 ** step)
        w = w + lr * m_hat / (jnp.sqrt(v_hat) + eps) - lr * weight_decay * w
        return w, m, v

    w_d, m_d, v_d = update_one(w_d, g_d, m_d, v_d)
    w_s, m_s, v_s = update_one(w_s, g_s, m_s, v_s)
    w_r, m_r, v_r = update_one(w_r, g_r, m_r, v_r)
    return (w_d, w_s, w_r, m_d, m_s, m_r, v_d, v_s, v_r)


# ══════════════════════════════════════════════════════════════════════
#  vmap in_axes: which args get a batch dimension (0) vs stay shared (None)
#
#  For _forward_and_accum:
#    x_input → batched (B,T,K)       init carries → batched (B,...)
#    weights → shared                 config/alphas → shared
# ══════════════════════════════════════════════════════════════════════

_FWD_AXES = (
    0,                               # x_input
    None, None, None,                # w_dend, w_soma, w_readout
    None, None, None, None, None,    # alpha_s, alpha_d, alpha_m, T_p, config
    None,                            # alpha_w (shared)
    (0, 0, 0, 0, 0, 0, 0, 0, 0),    # h_carry_init (9-tuple, each batched)
    (0, 0, 0),                       # r_carry_init (3-tuple, each batched)
    0,                               # A_d_init
    0,                               # rng_key (per-sample)
    None,                            # dropout_rate (shared)
)

_PRED_AXES = (
    0,                            # x_input
    None, None, None,             # weights
    None, None, None, None, None, # alphas, T_p, config
    None,                         # alpha_w
)

_LOSS_AXES = (
    0, 0, 0, 0,                   # mean_voltage, A_r, A_s, A_d (per-sample)
    0,                            # target_smoothed (per-sample)
    None, None, None,             # T, loss_temperature, loss_count_bias
)


# ══════════════════════════════════════════════════════════════════════
#  Pre-compiled versions:
#    _*_single = jit(core_fn)                  → one sample
#    _*_batch  = jit(vmap(core_fn, in_axes=…)) → B samples in parallel
# ══════════════════════════════════════════════════════════════════════

_fwd_single = jit(_forward_and_accum)
_pred_single = jit(_predict_only)
_loss_single = jit(_loss_and_grads)
_apply = jit(_apply_grads)
_adam = jit(_adam_apply)

_fwd_batch = jit(vmap(_forward_and_accum, in_axes=_FWD_AXES))
_pred_batch = jit(vmap(_predict_only, in_axes=_PRED_AXES))
_loss_batch = jit(vmap(_loss_and_grads, in_axes=_LOSS_AXES))


# ══════════════════════════════════════════════════════════════════════
#  Activity diagnostics — mean firing rate per hidden layer (forward only).
#  Rate = (total spikes) / (T * n_neurons): fraction of neurons spiking per
#  step, equivalently the per-neuron probability of spiking on a given step.
# ══════════════════════════════════════════════════════════════════════

def _activity_1l(
    x_input, w_dend, w_soma, w_readout,
    alpha_s, alpha_d, alpha_m, T_p, config, alpha_w,
):
    dend_in = x_input @ w_dend.T
    soma_in = x_input @ w_soma.T
    T = x_input.shape[0]
    n = w_dend.shape[0]
    time_indices = jnp.arange(T, dtype=jnp.int32)

    # w_readout and alpha_m are unused here — they exist so this function can share
    # _PRED_AXES with _predict_only rather than duplicate the vmap spec.

    def step(carry, inputs):
        h_state, s = carry
        dend_in_t, soma_in_t, t = inputs
        h_state, o, *_ = TwoCompNeuron.dynamics_step(
            h_state, dend_in_t, soma_in_t, t, alpha_s, alpha_d, T_p, config, alpha_w,
        )
        return (h_state, s + jnp.sum(o.astype(jnp.float64))), None

    h_zeros = (
        jnp.zeros(n), jnp.zeros(n),
        jnp.zeros(n, dtype=jnp.int32), jnp.zeros(n, dtype=jnp.int32),
        jnp.zeros(n), jnp.zeros(n),
    )
    (_, s), _ = lax.scan(step, (h_zeros, 0.0), (dend_in, soma_in, time_indices))
    return s / (T * n)


_act_batch_1l = jit(vmap(_activity_1l, in_axes=_PRED_AXES))


# ══════════════════════════════════════════════════════════════════════
#  Network class — ties everything together
# ══════════════════════════════════════════════════════════════════════

class Network:
    def __init__(
        self,
        key: jnp.ndarray,
        n_inputs: int,
        n_hidden: int,
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
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.config = config
        self.optimizer = optimizer
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay

        key_h, key_r, key_rng = random.split(key, 3)
        self.hidden = TwoCompNeuron(key_h, n_hidden, n_inputs, config)
        self.readout = LINeuron(key_r, n_outputs, n_hidden, config)
        self.rng_key = key_rng

        if optimizer == "adam":
            self.beta1 = beta1
            self.beta2 = beta2
            self.adam_eps = adam_eps
            self.adam_step = jnp.array(0, dtype=jnp.int32)
            self.m_dend = jnp.zeros_like(self.hidden.w_dend)
            self.m_soma = jnp.zeros_like(self.hidden.w_soma)
            self.m_readout = jnp.zeros_like(self.readout.w)
            self.v_dend = jnp.zeros_like(self.hidden.w_dend)
            self.v_soma = jnp.zeros_like(self.hidden.w_soma)
            self.v_readout = jnp.zeros_like(self.readout.w)

    # ── Helpers to build zero-initialized carries ──

    def _h_carry(self, B=None):
        """Hidden neuron carry. B=None → single sample, B=int → batched."""
        return self.hidden.init_carry(B)

    def _r_carry(self, B=None):
        """Readout neuron carry. B=None → single sample, B=int → batched."""
        return self.readout.init_carry(B)

    def _A_d_zeros(self, B=None):
        """Dendritic accumulator zeros."""
        base = (self.n_outputs, self.n_hidden, self.n_inputs)
        return jnp.zeros((B,) + base if B else base)

    def _weights(self):
        return self.hidden.w_dend, self.hidden.w_soma, self.readout.w

    def _params(self):
        return (self.hidden.alpha_s, self.hidden.alpha_d, self.readout.alpha_m,
                self.hidden.T_p, self.config, self.hidden.alpha_w)

    def _smooth_targets(self, targets):
        """Scalar label or (B,) labels → smoothed one-hot vector(s)."""
        cfg = self.config
        one_hot = jnp.eye(self.n_outputs)[targets]
        return one_hot * (1 - cfg.loss_label_smoothing) + cfg.loss_label_smoothing / self.n_outputs

    def _update_weights(self, g_d, g_s, g_r, lr):
        """Apply gradients using the configured optimizer (SGD or Adam).

        Both branches use decoupled weight decay (subtract lr·λ·w
        after the gradient step). For SGD this is equivalent to an
        L2 loss penalty (λ/2)·||w||²; for Adam this is the AdamW
        recipe — the decay does not pass through the 1/√v rescaling.
        """
        if self.optimizer == "adam":
            self.adam_step = self.adam_step + 1
            result = _adam(
                self.hidden.w_dend, self.hidden.w_soma, self.readout.w,
                g_d, g_s, g_r,
                self.m_dend, self.m_soma, self.m_readout,
                self.v_dend, self.v_soma, self.v_readout,
                self.adam_step, lr, self.beta1, self.beta2, self.adam_eps,
                self.weight_decay,
            )
            (self.hidden.w_dend, self.hidden.w_soma, self.readout.w,
             self.m_dend, self.m_soma, self.m_readout,
             self.v_dend, self.v_soma, self.v_readout) = result
        else:
            self.hidden.w_dend, self.hidden.w_soma, self.readout.w = _apply(
                *self._weights(), g_d, g_s, g_r, lr, self.weight_decay,
            )

    def _next_key(self):
        """Advance the PRNG and return a fresh subkey for dropout."""
        self.rng_key, subkey = random.split(self.rng_key)
        return subkey

    # ── Single-sample API ──

    def train_step(self, x_input, target, lr=1e-3):
        """Train on one sample (with dropout during forward pass).
        Returns: (loss, prediction, grad_norms_dict)
        """
        T = x_input.shape[0]

        mean_voltage, A_r, A_s, A_d = _fwd_single(
            x_input, *self._weights(), *self._params(),
            self._h_carry(), self._r_carry(), self._A_d_zeros(),
            self._next_key(), self.dropout_rate,
        )

        loss, pred, g_r, g_s, g_d = _loss_single(
            mean_voltage, A_r, A_s, A_d,
            self._smooth_targets(target), T,
            self.config.loss_temperature, self.config.loss_count_bias,
        )

        gnorms = {
            "readout": float(jnp.linalg.norm(g_r)),
            "soma": float(jnp.linalg.norm(g_s)),
            "dend": float(jnp.linalg.norm(g_d)),
        }

        self._update_weights(g_d, g_s, g_r, lr)
        return float(loss), int(pred), gnorms

    def predict(self, x_input):
        """Predict one sample (no dropout). x_input: (T,K) → int class label."""
        mean_voltage = _pred_single(x_input, *self._weights(), *self._params())
        return int(jnp.argmax(mean_voltage))

    # ── Batched API ──

    def batch_train_step(self, x_batch, targets, lr=1e-3):
        """Train on B samples in parallel (with dropout).
        Returns: (mean_loss, predictions_array (B,), grad_norms_dict)
        """
        B = x_batch.shape[0]
        T = x_batch.shape[1]

        batch_keys = random.split(self._next_key(), B)

        mean_voltage, A_r, A_s, A_d = _fwd_batch(
            x_batch, *self._weights(), *self._params(),
            self._h_carry(B), self._r_carry(B), self._A_d_zeros(B),
            batch_keys, self.dropout_rate,
        )

        losses, preds, g_r, g_s, g_d = _loss_batch(
            mean_voltage, A_r, A_s, A_d,
            self._smooth_targets(targets), T,
            self.config.loss_temperature, self.config.loss_count_bias,
        )

        g_r_avg = jnp.mean(g_r, axis=0)
        g_s_avg = jnp.mean(g_s, axis=0)
        g_d_avg = jnp.mean(g_d, axis=0)

        gnorms = {
            "readout": float(jnp.linalg.norm(g_r_avg)),
            "soma": float(jnp.linalg.norm(g_s_avg)),
            "dend": float(jnp.linalg.norm(g_d_avg)),
        }

        self._update_weights(g_d_avg, g_s_avg, g_r_avg, lr)
        return float(jnp.mean(losses)), preds, gnorms

    def batch_predict(self, x_batch):
        """Predict B samples in parallel. x_batch: (B,T,K) → (B,) int labels."""
        mean_voltage = _pred_batch(x_batch, *self._weights(), *self._params())
        return jnp.argmax(mean_voltage, axis=1)

    def activity(self, x_batch):
        """Mean firing rate of the hidden layer over a batch (no dropout).

        Rate = spikes / (T * n_neurons): the fraction of neurons spiking per
        timestep. Returns {"hidden": r}.
        """
        r = _act_batch_1l(x_batch, *self._weights(), *self._params())
        return {"hidden": float(jnp.mean(r))}

    # ── Persistence ──────────────────────────────────────────────────────
    #
    #  save() writes the *fully trained* model to a single compressed .npz:
    #  the learned weights PLUS the per-neuron random array T_p that the
    #  forward pass needs but which is NOT recoverable from the seed alone in
    #  a stable way. All non-array state (arch, build flags, full NeuronConfig,
    #  provenance) is JSON-encoded under "meta".
    #
    #  load() rebuilds a Network whose arrays are JAX (jnp) arrays — identical
    #  in kind to a freshly trained net, so net.predict / net.activity /
    #  net.batch_predict and the analysis scripts work unchanged.

    def save(self, path, extra=None):
        """Serialize the trained model to a compressed .npz at ``path``.

        ``extra`` (optional dict, e.g. accuracies + CLI args) is stored as
        provenance under the JSON "meta" entry. Returns the path written.
        """
        meta = {
            "arch": {
                "n_inputs": int(self.n_inputs),
                "n_hidden": int(self.n_hidden),
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

        arrays = {}
        arrays.update(
            w_dend=np.asarray(self.hidden.w_dend),
            w_soma=np.asarray(self.hidden.w_soma),
            w_readout=np.asarray(self.readout.w),
            T_p=np.asarray(self.hidden.T_p),
            alpha_d=np.asarray(self.hidden.alpha_d),
            alpha_s=np.asarray(self.hidden.alpha_s),
            alpha_w=np.asarray(self.hidden.alpha_w),
            alpha_m=np.asarray(self.readout.alpha_m),
        )

        if self.optimizer == "adam":
            arrays["adam_step"] = np.asarray(self.adam_step)
            moment_names = (
                "m_dend", "m_soma", "m_readout", "v_dend", "v_soma", "v_readout"
            )
            for nm in moment_names:
                arrays[nm] = np.asarray(getattr(self, nm))

        arrays["meta"] = np.array(json.dumps(meta))
        np.savez_compressed(path, **arrays)
        return path

    @classmethod
    def load(cls, path):
        """Rebuild a Network from a .npz written by ``save``.

        Returns a fully usable model holding JAX arrays. The original
        ``meta`` dict (arch, config, provenance) is attached as ``net.meta``.
        """
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        arch = meta["arch"]
        build = meta["build"]
        config = NeuronConfig(**meta["config"])

        # Construct with a throwaway seed; every stateful array is overwritten
        # from the file next, so the original seed is not needed.
        net = cls(
            random.PRNGKey(0),
            arch["n_inputs"], arch["n_hidden"], arch["n_outputs"], config,
            optimizer=build["optimizer"],
            beta1=build.get("beta1", 0.9),
            beta2=build.get("beta2", 0.999),
            adam_eps=build.get("adam_eps", 1e-8),
            dropout_rate=build["dropout_rate"],
            weight_decay=build["weight_decay"],
        )

        def arr(k):
            return jnp.asarray(data[k])

        net.hidden.w_dend = arr("w_dend")
        net.hidden.w_soma = arr("w_soma")
        net.readout.w = arr("w_readout")
        net.hidden.T_p = arr("T_p").astype(jnp.int32)
        net.hidden.alpha_d = arr("alpha_d")
        net.hidden.alpha_s = arr("alpha_s")
        net.hidden.alpha_w = arr("alpha_w")
        net.readout.alpha_m = arr("alpha_m")

        if build["optimizer"] == "adam" and "adam_step" in data.files:
            net.adam_step = arr("adam_step").astype(jnp.int32)
            moment_names = (
                "m_dend", "m_soma", "m_readout", "v_dend", "v_soma", "v_readout"
            )
            for nm in moment_names:
                if nm in data.files:
                    setattr(net, nm, arr(nm))

        net.meta = meta
        return net
