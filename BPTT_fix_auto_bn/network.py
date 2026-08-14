import dataclasses
import json
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit, lax, value_and_grad, custom_vjp

from config import NeuronConfig, surrogate_sigma
from two_comp_neuron import TwoCompNeuron
from lif_neuron import LINeuron


# Guards log(0) in the cross-entropy. Small enough not to bias the loss.
_LOG_EPS = 1e-8


# ══════════════════════════════════════════════════════════════════════
#  FULL SURROGATE BPTT via jax.grad  +  sparch-style BatchNorm1d.
#
#  This is BPTT_fix_auto (exact autodiff gradient) PLUS the batch normalization
#  used by the SSM-inspired-LIF paper (arXiv:2506.06374), which builds on the
#  sparch codebase (Bittar & Garner). "Their" BN is nn.BatchNorm1d(H,
#  momentum=0.05) applied to the per-layer pre-synaptic current Wx, reshaped to
#  (B*T, H) — i.e. statistics pooled over batch AND time, per hidden channel,
#  with a learned affine (γ, β) and running mean/var used at eval. Dropout is
#  applied AFTER the spikes.
#
#  Batch statistics couple every sample, so — unlike BPTT_fix_auto — the forward
#  CANNOT be vmapped per-sample. We process the batch as an explicit (B, T, H)
#  axis, layer-sequentially (sparch style): per layer, project the whole
#  sequence, BN-pool over B*T, run the neuron dynamics over time, feed the
#  spikes to the next layer. The loss is the mean CE over the batch and we take
#  ONE jax.grad of it — BN's batch-coupled backward is then automatic. The two
#  compartments give two Linear projections per layer, so (faithful to "one BN
#  per nn.Linear") each of soma_in / dend_in gets its own BN; config.bn_targets
#  selects which.
#
#  Sign convention unchanged: optimizer does w <- w + lr·g, so we return the
#  ASCENT direction g = -∇_w loss. BN affine (γ, β) are trained the same way;
#  running mean/var are buffers (stop_gradient) threaded through the Network.
# ══════════════════════════════════════════════════════════════════════


@custom_vjp
def _spike(x, beta):
    """Heaviside Θ(x) on the forward pass, surrogate slope on the backward."""
    return jnp.where(x >= 0.0, 1.0, 0.0)


def _spike_fwd(x, beta):
    return jnp.where(x >= 0.0, 1.0, 0.0), (x, beta)


def _spike_bwd(res, g):
    x, beta = res
    return (g * surrogate_sigma(x, beta), jnp.zeros_like(beta))


_spike.defvjp(_spike_fwd, _spike_bwd)


def _tc_step(state, dend_in, soma_in, t, alpha_s, alpha_d, T_p, config, alpha_w):
    """Differentiable mirror of TwoCompNeuron.forward_step (one timestep).

    Elementwise in the neuron axis, so it broadcasts over a leading batch axis:
    pass (B, H) state and (B, H) currents and it runs the whole batch at once.
    Identical forward values to two_comp_neuron.forward_step; the two hard
    thresholds go through `_spike` so autodiff sees a surrogate.
    """
    mu_prev, v_prev, h_prev, tp_prev, matp_prev, w_prev = state

    t_prime = jnp.where(t == 0, 0, jnp.where(h_prev == 1, tp_prev, t))
    mu = jnp.where(t > 0, alpha_d * mu_prev + (1 - h_prev) * dend_in, dend_in)
    mu_at_tprime = jnp.where(h_prev == 0, mu, matp_prev)

    plateau_duration = t - t_prime
    timing_ok = ((plateau_duration <= T_p) & (plateau_duration >= 0)).astype(mu.dtype)
    h = timing_ok * _spike(mu_at_tprime - config.mu_th, config.beta_d)

    v_pre = jnp.where(t > 0, alpha_s * v_prev + soma_in - w_prev, soma_in)
    o = _spike(v_pre + config.gamma * h - config.v_th, config.beta_s)
    v = v_pre * (1 - o)
    w = alpha_w * w_prev + (1 - alpha_w) * config.a_adapt * v_pre + config.b_adapt * o

    new_state = (mu, v, h, t_prime, mu_at_tprime, w)
    return new_state, o


def _init_tc_state(shape):
    """Zero carry for _tc_step: (mu, v, h, t_prime, mu_at_tprime, w_adapt).

    `shape` may be an int (n,) or a tuple like (B, H). t_prime is the only
    integer field."""
    return (
        jnp.zeros(shape), jnp.zeros(shape), jnp.zeros(shape),
        jnp.zeros(shape, dtype=jnp.int32), jnp.zeros(shape), jnp.zeros(shape),
    )


def _bn_keys(config):
    """Which projections get a BatchNorm, resolved from the (static) config."""
    if config.normalization != "batchnorm":
        return []
    t = config.bn_targets
    if t == "both":
        return ["soma", "dend"]
    if t in ("soma", "dend"):
        return [t]
    return []


def _batchnorm1d(x, gamma, beta, run_mean, run_var, training, momentum, eps):
    """sparch-style BatchNorm1d on a (B, T, H) pre-synaptic current.

    Train: normalize with batch mean/var pooled over (B*T) per channel, and
    update running stats (PyTorch convention: running <- (1-m)·running +
    m·batch_stat, running var Bessel-corrected). The running-stat update is
    stop_gradient-ed — it is a buffer, not a parameter. Eval: normalize with the
    running stats. gamma/beta/run_mean/run_var are (H,).
    """
    if training:
        mean = jnp.mean(x, axis=(0, 1))
        var = jnp.var(x, axis=(0, 1))                      # biased, for normalize
        xhat = (x - mean) / jnp.sqrt(var + eps)
        n = x.shape[0] * x.shape[1]
        new_mean = (1.0 - momentum) * run_mean + momentum * mean
        new_var = (1.0 - momentum) * run_var + momentum * var * (n / (n - 1.0))
        new_mean = lax.stop_gradient(new_mean)
        new_var = lax.stop_gradient(new_var)
    else:
        xhat = (x - run_mean) / jnp.sqrt(run_var + eps)
        new_mean, new_var = run_mean, run_var
    return gamma * xhat + beta, new_mean, new_var


def _run_network(
    weights, bn_stats, x, alpha_s, alpha_d, alpha_m, tp_list, config, alpha_w,
    rng_key, dropout_rate, training,
):
    """Batched, layer-sequential forward. Returns (mean_v (B,J), rates, new_bn).

    x: (B, T, C_in). Per layer: project the whole sequence, BN-pool over (B*T),
    run the neuron dynamics over time (batched), dropout after spikes; the final
    layer feeds the LI readout. `training` (static) toggles BN train/eval mode
    and dropout.
    """
    w_dend = weights["dend"]
    w_soma = weights["soma"]
    w_readout = weights["readout"]
    L = len(w_dend)

    x = x.astype(jnp.float64)
    B, T, _ = x.shape
    keys = random.split(rng_key, L)
    do_bn = config.normalization == "batchnorm"
    bn_keys = _bn_keys(config)
    dropout_scale = 1.0 / (1.0 - dropout_rate)

    new_mean = {k: [] for k in bn_keys}
    new_var = {k: [] for k in bn_keys}
    rates = []
    x_l = x
    for l in range(L):
        dend_cur = x_l @ w_dend[l].T                       # (B, T, H_l)
        soma_cur = x_l @ w_soma[l].T
        if do_bn and "soma" in bn_keys:
            soma_cur, m_s, v_s = _batchnorm1d(
                soma_cur, weights["bn_g"]["soma"][l], weights["bn_b"]["soma"][l],
                bn_stats["mean"]["soma"][l], bn_stats["var"]["soma"][l],
                training, config.bn_momentum, config.bn_eps,
            )
            new_mean["soma"].append(m_s); new_var["soma"].append(v_s)
        if do_bn and "dend" in bn_keys:
            dend_cur, m_d, v_d = _batchnorm1d(
                dend_cur, weights["bn_g"]["dend"][l], weights["bn_b"]["dend"][l],
                bn_stats["mean"]["dend"][l], bn_stats["var"]["dend"][l],
                training, config.bn_momentum, config.bn_eps,
            )
            new_mean["dend"].append(m_d); new_var["dend"].append(v_d)

        H_l = w_dend[l].shape[0]
        state0 = _init_tc_state((B, H_l))

        def _lstep(state, inp):
            dend_t, soma_t, t = inp
            return _tc_step(
                state, dend_t, soma_t, t,
                alpha_s, alpha_d, tp_list[l], config, alpha_w,
            )

        _, o_TBH = lax.scan(
            _lstep, state0,
            (dend_cur.transpose(1, 0, 2), soma_cur.transpose(1, 0, 2),
             jnp.arange(T, dtype=jnp.int32)),
        )
        o_l = o_TBH.transpose(1, 0, 2)                     # (B, T, H_l)
        rates.append(jnp.sum(o_l) / (T * H_l * B))

        # Dropout AFTER the spikes (sparch order). `training` is static; at
        # dropout_rate=0 the mask is all-ones and this is a no-op, so no Python
        # branch on the traced rate is needed.
        if training:
            mask = random.bernoulli(
                keys[l], 1.0 - dropout_rate, o_l.shape,
            ).astype(o_l.dtype) * dropout_scale
            o_l = o_l * mask
        x_l = o_l

    # LI readout over the final layer's spikes.
    J = w_readout.shape[0]
    cur_r = x_l @ w_readout.T                              # (B, T, J)

    def _rstep(v, c):
        v = alpha_m * v + c
        return v, v

    _, v_TBJ = lax.scan(_rstep, jnp.zeros((B, J)), cur_r.transpose(1, 0, 2))
    mean_v = jnp.sum(v_TBJ, axis=0) / T                    # (B, J)

    new_bn = {"mean": new_mean, "var": new_var} if do_bn else {}
    return mean_v, rates, new_bn


def _batched_loss(
    weights, bn_stats, x, alpha_s, alpha_d, alpha_m, tp_list, config, alpha_w,
    target_smoothed, loss_temperature, loss_count_bias, rng_key, dropout_rate,
    training,
):
    """Mean cross-entropy over the batch. Returns (loss, (mean_v, new_bn))."""
    mean_v, _rates, new_bn = _run_network(
        weights, bn_stats, x, alpha_s, alpha_d, alpha_m, tp_list, config, alpha_w,
        rng_key, dropout_rate, training,
    )
    scaled = mean_v / loss_temperature + loss_count_bias
    probs = jnp.exp(scaled - jnp.max(scaled, axis=1, keepdims=True))
    probs = probs / jnp.sum(probs, axis=1, keepdims=True)
    loss = jnp.mean(-jnp.sum(target_smoothed * jnp.log(probs + _LOG_EPS), axis=1))
    return loss, (mean_v, new_bn)


def _batched_fb(
    weights, bn_stats, x, alpha_s, alpha_d, alpha_m, tp_list, config, alpha_w,
    target_smoothed, loss_temperature, loss_count_bias, rng_key, dropout_rate,
    training,
):
    """value_and_grad over the whole batch. Returns loss, preds, grads, new_bn.

    grads has the same pytree structure as `weights` (incl. BN γ/β) and is the
    ASCENT direction (-∇loss) for the w <- w + lr·g optimizer.
    """
    (loss, (mean_v, new_bn)), grads = value_and_grad(_batched_loss, has_aux=True)(
        weights, bn_stats, x, alpha_s, alpha_d, alpha_m, tp_list, config, alpha_w,
        target_smoothed, loss_temperature, loss_count_bias, rng_key, dropout_rate,
        training,
    )
    preds = jnp.argmax(mean_v, axis=1)
    grads = jax.tree_util.tree_map(lambda g: -g, grads)
    return loss, preds, grads, new_bn


def _batched_predict(
    weights, bn_stats, x, alpha_s, alpha_d, alpha_m, tp_list, config, alpha_w,
    training,
):
    """Forward only -> mean voltage (B, J). BN uses running stats when eval."""
    mean_v, _rates, _bn = _run_network(
        weights, bn_stats, x, alpha_s, alpha_d, alpha_m, tp_list, config, alpha_w,
        random.PRNGKey(0), 0.0, training,
    )
    return mean_v


def _batched_activity(
    weights, bn_stats, x, alpha_s, alpha_d, alpha_m, tp_list, config, alpha_w,
    training,
):
    """Mean firing rate per hidden layer over the batch (no dropout)."""
    _mean_v, rates, _bn = _run_network(
        weights, bn_stats, x, alpha_s, alpha_d, alpha_m, tp_list, config, alpha_w,
        random.PRNGKey(0), 0.0, training,
    )
    return rates


def _apply_grads(weights_flat, grads_flat, lr, clip_value, weight_decay):
    """SGD with decoupled weight decay over a flat list of weight tensors."""
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


# ── Pre-compiled entry points (training flag is static, baked in) ──────────
_fb_train = jit(partial(_batched_fb, training=True))
_predict_fn = jit(partial(_batched_predict, training=False))
_activity_fn = jit(partial(_batched_activity, training=False))
_apply = jit(_apply_grads)
_adam = jit(_adam_apply)


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

        # BatchNorm parameters (γ, β) and running buffers (mean, var), one set
        # per selected projection per hidden layer. Each normalizes a current
        # with H_l channels (the layer's own hidden size).
        self.bn_on = config.normalization == "batchnorm"
        self._bnk = _bn_keys(config)
        if self.bn_on:
            self.bn_gamma = {k: [jnp.ones(self.hidden_sizes[l]) for l in range(self.n_layers)] for k in self._bnk}
            self.bn_beta = {k: [jnp.zeros(self.hidden_sizes[l]) for l in range(self.n_layers)] for k in self._bnk}
            self.bn_mean = {k: [jnp.zeros(self.hidden_sizes[l]) for l in range(self.n_layers)] for k in self._bnk}
            self.bn_var = {k: [jnp.ones(self.hidden_sizes[l]) for l in range(self.n_layers)] for k in self._bnk}

        if optimizer == "adam":
            self.beta1 = beta1
            self.beta2 = beta2
            self.adam_eps = adam_eps
            self.adam_step = jnp.array(0, dtype=jnp.int32)
            self.m = [jnp.zeros_like(w) for w in self._weights_flat()]
            self.v = [jnp.zeros_like(w) for w in self._weights_flat()]

    # ── Weight / buffer views ──
    #
    # `_weights` is the DIFFERENTIATED pytree (dend/soma/readout + BN γ,β);
    # `_bn_stats` is the non-differentiated running mean/var. Both are rebuilt
    # from the layer objects each call so they reflect the current state. The
    # optimizer runs over jax.tree_util leaves of `_weights`, so BN params are
    # handled automatically without hand-listing them.

    def _weights(self):
        w = {
            "dend": [h.w_dend for h in self.hidden],
            "soma": [h.w_soma for h in self.hidden],
            "readout": self.readout.w,
        }
        if self.bn_on:
            w["bn_g"] = {k: list(self.bn_gamma[k]) for k in self._bnk}
            w["bn_b"] = {k: list(self.bn_beta[k]) for k in self._bnk}
        return w

    def _bn_stats(self):
        if not self.bn_on:
            return {}
        return {
            "mean": {k: list(self.bn_mean[k]) for k in self._bnk},
            "var": {k: list(self.bn_var[k]) for k in self._bnk},
        }

    def _weights_flat(self):
        return jax.tree_util.tree_leaves(self._weights())

    def _set_weights(self, wtree):
        for l in range(self.n_layers):
            self.hidden[l].w_dend = wtree["dend"][l]
            self.hidden[l].w_soma = wtree["soma"][l]
        self.readout.w = wtree["readout"]
        if self.bn_on:
            for k in self._bnk:
                self.bn_gamma[k] = list(wtree["bn_g"][k])
                self.bn_beta[k] = list(wtree["bn_b"][k])

    def _set_bn_stats(self, new_bn):
        for k in self._bnk:
            self.bn_mean[k] = list(new_bn["mean"][k])
            self.bn_var[k] = list(new_bn["var"][k])

    def _tp_list(self):
        return [h.T_p for h in self.hidden]

    def _params(self):
        # alpha_s/alpha_d/alpha_w are identical across layers (global taus).
        h0 = self.hidden[0]
        return (h0.alpha_s, h0.alpha_d, self.readout.alpha_m,
                self._tp_list(), self.config, h0.alpha_w)

    def _smooth_targets(self, targets):
        cfg = self.config
        one_hot = jnp.eye(self.n_outputs)[targets]
        return one_hot * (1 - cfg.loss_label_smoothing) + cfg.loss_label_smoothing / self.n_outputs

    def _update_weights(self, grads_tree, lr, clip_value):
        wtree = self._weights()
        leaves, treedef = jax.tree_util.tree_flatten(wtree)
        gleaves = jax.tree_util.tree_leaves(grads_tree)  # same structure → aligned
        if self.optimizer == "adam":
            self.adam_step = self.adam_step + 1
            new_leaves, self.m, self.v = _adam(
                leaves, gleaves, self.m, self.v,
                self.adam_step, lr, self.beta1, self.beta2, self.adam_eps,
                clip_value, self.weight_decay,
            )
        else:
            new_leaves = _apply(leaves, gleaves, lr, clip_value, self.weight_decay)
        self._set_weights(jax.tree_util.tree_unflatten(treedef, new_leaves))

    def _next_key(self):
        self.rng_key, subkey = random.split(self.rng_key)
        return subkey

    def _grad_norms(self, grads_tree):
        norms = {"readout": float(jnp.linalg.norm(grads_tree["readout"]))}
        for l in range(self.n_layers):
            norms[f"dend{l}"] = float(jnp.linalg.norm(grads_tree["dend"][l]))
            norms[f"soma{l}"] = float(jnp.linalg.norm(grads_tree["soma"][l]))
        if self.bn_on:
            for k in self._bnk:
                for l in range(self.n_layers):
                    norms[f"bn_{k}_g{l}"] = float(jnp.linalg.norm(grads_tree["bn_g"][k][l]))
        return norms

    # ── Batched API ──

    def batch_train_step(self, x_batch, targets, lr=1e-3, clip_value=1.0):
        key = self._next_key()
        loss, preds, grads, new_bn = _fb_train(
            self._weights(), self._bn_stats(), x_batch, *self._params(),
            self._smooth_targets(targets),
            self.config.loss_temperature, self.config.loss_count_bias,
            key, self.dropout_rate,
        )
        if self.bn_on:
            self._set_bn_stats(new_bn)
        gnorms = self._grad_norms(grads)
        self._update_weights(grads, lr, clip_value)
        return float(loss), preds, gnorms

    def batch_predict(self, x_batch):
        mean_v = _predict_fn(self._weights(), self._bn_stats(), x_batch, *self._params())
        return jnp.argmax(mean_v, axis=1)

    def activity(self, x_batch):
        """Mean firing rate per hidden layer over a batch (no dropout, BN eval)."""
        rates = _activity_fn(self._weights(), self._bn_stats(), x_batch, *self._params())
        return {f"hidden{l}": float(rates[l]) for l in range(self.n_layers)}

    # ── Single-sample API (routed through the batch path, B=1) ──

    def train_step(self, x_input, target, lr=1e-3, clip_value=1.0):
        loss, preds, gnorms = self.batch_train_step(
            x_input[None], jnp.asarray([target]), lr, clip_value,
        )
        return loss, int(preds[0]), gnorms

    def predict(self, x_input):
        mean_v = _predict_fn(
            self._weights(), self._bn_stats(), x_input[None], *self._params(),
        )
        return int(jnp.argmax(mean_v[0]))

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

        if self.bn_on:
            for k in self._bnk:
                for l in range(self.n_layers):
                    arrays[f"bn_{k}_gamma_{l}"] = np.asarray(self.bn_gamma[k][l])
                    arrays[f"bn_{k}_beta_{l}"] = np.asarray(self.bn_beta[k][l])
                    arrays[f"bn_{k}_mean_{l}"] = np.asarray(self.bn_mean[k][l])
                    arrays[f"bn_{k}_var_{l}"] = np.asarray(self.bn_var[k][l])

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

        if net.bn_on:
            for k in net._bnk:
                for l in range(net.n_layers):
                    net.bn_gamma[k][l] = arr(f"bn_{k}_gamma_{l}")
                    net.bn_beta[k][l] = arr(f"bn_{k}_beta_{l}")
                    net.bn_mean[k][l] = arr(f"bn_{k}_mean_{l}")
                    net.bn_var[k][l] = arr(f"bn_{k}_var_{l}")

        if build["optimizer"] == "adam" and "adam_step" in data.files:
            net.adam_step = arr("adam_step").astype(jnp.int32)
            n_tensors = len(net._weights_flat())
            net.m = [arr(f"m_{i}") for i in range(n_tensors)]
            net.v = [arr(f"v_{i}") for i in range(n_tensors)]

        net.meta = meta
        return net
