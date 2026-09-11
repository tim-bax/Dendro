"""Multi-layer two-compartment network trained by autograd.

This is the AUTOGRAD sibling of ``Soft_Reset/network.py``. Instead of the
hand-coded two-pass e-prop rule (which is single-layer only), the whole forward
pass is written *differentiably* and gradients come straight from
``jax.value_and_grad``. We can do this because the hand-coded e-prop gradient was
shown to equal reverse-mode autograd of the same forward pass **with the somatic
reset detached** — see ``analysis/verify_detached_reset_grad.py`` and
``verify_autograd_multi.py`` in this folder.

The differentiable forward uses the verified recipe:
  * the somatic spike ``o`` and dendritic plateau ``h`` are hard 0/1 in the
    forward value but carry surrogate gradients (the spike's true gradient is
    zero almost everywhere, so a surrogate is mandatory — not optional);
  * the soft reset ``o * v_th_eff`` is wrapped in ``stop_gradient`` — this
    "detached reset" is what keeps the gradient BOUNDED. Plain autograd through
    the plateau explodes exponentially (see the BPTT_fix_auto experiments);
    detaching the reset is the bridge back to e-prop.
  * integer state that acts purely as a gate (the plateau flag ``h_prev``, the
    ``t_prime`` latch index, the plateau-duration window) is carried as a hard
    int, so no gradient flows through it.

Because autograd handles inter-layer credit assignment automatically (through the
spike surrogates), stacking layers is free: a layer's spike train is the next
layer's input, and the last hidden layer drives one LI readout.

Surrogate choices are made to REPRODUCE ``Soft_Reset``'s e-prop gradient, so that
a single-layer net here matches it exactly (readout/soma/dend). The somatic spike
uses a two-path ``custom_vjp`` (``somatic_spike``): the soma path (∂o/∂v_pre)
uses ``surrogate_sigma(·, beta_s)``; the dendritic path (∂o/∂h) uses the
``dend_surrogate_roof`` flag — the flat "roof" over [v_th-gamma, v_th]
(``surrogate_roof(·, beta_s_dend)``) when set, else the single bump
``gamma·surrogate_sigma(·, beta_s_dend)``. The plateau surrogate ``h`` is applied
UNGATED by the duration window (matching Soft_Reset's ``hp_hidden``).

Exact single-layer parity with Soft_Reset holds when ``tau_dend == 15`` (so the
autograd eligibility, which decays at ``alpha_d = exp(-dt/tau_dend)``, equals
Soft_Reset's fixed 15 ms ``_TAU_ELIG_DEND_MS``) and adaptation is off
(``a_adapt == b_adapt == 0``). Soft_Reset's update also folds a ``-temperature``
factor and does ascent, so identical weight steps need ``lr = lr_soft_reset *
loss_temperature`` (this net does plain descent ``w -= lr·dL/dw``).
"""
import dataclasses
import json
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit, lax, vmap

from config import NeuronConfig, surrogate_sigma, surrogate_roof
from two_comp_neuron import TwoCompNeuron
from lif_neuron import LINeuron


# Guards log(0) in the cross-entropy. Small enough not to bias the loss.
_LOG_EPS = 1e-8


# ══════════════════════════════════════════════════════════════════════
#  Differentiable spike: forward = Heaviside, backward = surrogate_sigma.
#
#  ``beta`` is a normal argument (not nondiff_argnums) so it may be a JAX
#  tracer when ``config`` is threaded through jit; its cotangent is None
#  (zero) because we never differentiate w.r.t. the surrogate sharpness.
# ══════════════════════════════════════════════════════════════════════

@jax.custom_vjp
def sg_spike(x, beta):
    """H(x) with a surrogate derivative surrogate_sigma(x, beta) w.r.t. x."""
    return (x >= 0).astype(x.dtype)


def _sg_spike_fwd(x, beta):
    return (x >= 0).astype(x.dtype), (x, beta)


def _sg_spike_bwd(res, g):
    x, beta = res
    return (g * surrogate_sigma(x, beta), None)


sg_spike.defvjp(_sg_spike_fwd, _sg_spike_bwd)


def _straight_through(hard, soft):
    """Value of ``hard``, gradient of ``soft``."""
    return lax.stop_gradient(hard - soft) + soft


# ══════════════════════════════════════════════════════════════════════
#  Two-path somatic spike  o = H(v_pre - (v_th - gamma*h)).
#
#  Forward is the hard 0/1. The backward gives the two incoming edges
#  DIFFERENT surrogates, reproducing Soft_Reset's split (sp_hidden vs
#  sp_hidden_d):
#    ∂o/∂v_pre = surrogate_sigma(v_pre - v_th_eff, beta_s)           [soma path]
#    ∂o/∂h     = surrogate_roof(v_pre, v_th-gamma, v_th, beta_s_dend) [dend, roof]
#             or gamma * surrogate_sigma(v_pre - v_th_eff, beta_s_dend) [dend, bump]
#  ``use_roof`` is a static Python bool (config.dend_surrogate_roof) so it can
#  pick the branch at trace time. In the roof branch gamma is folded into the
#  roof's width (no explicit gamma), exactly as Soft_Reset/network.py does.
# ══════════════════════════════════════════════════════════════════════

@partial(jax.custom_vjp, nondiff_argnums=(2,))
def somatic_spike(v_pre, h, use_roof, v_th, gamma, beta_s, beta_s_dend):
    return (v_pre >= (v_th - gamma * h)).astype(v_pre.dtype)


def _somatic_spike_fwd(v_pre, h, use_roof, v_th, gamma, beta_s, beta_s_dend):
    o = (v_pre >= (v_th - gamma * h)).astype(v_pre.dtype)
    return o, (v_pre, h, v_th, gamma, beta_s, beta_s_dend)


def _somatic_spike_bwd(use_roof, res, g):
    v_pre, h, v_th, gamma, beta_s, beta_s_dend = res
    v_th_eff = v_th - gamma * h
    d_vpre = g * surrogate_sigma(v_pre - v_th_eff, beta_s)
    if use_roof:
        d_h = g * surrogate_roof(v_pre, v_th - gamma, v_th, beta_s_dend)
    else:
        d_h = g * gamma * surrogate_sigma(v_pre - v_th_eff, beta_s_dend)
    # cotangents for (v_pre, h, v_th, gamma, beta_s, beta_s_dend)
    return (d_vpre, d_h, None, None, None, None)


somatic_spike.defvjp(_somatic_spike_fwd, _somatic_spike_bwd)


# ══════════════════════════════════════════════════════════════════════
#  One differentiable two-compartment layer, scanned over time.
#
#  Equation-for-equation identical to TwoCompNeuron.dynamics_step, so the
#  forward VALUES match it exactly (straight-through preserves the hard
#  0/1). The only additions are the surrogate gradients and the detached
#  reset. dynamics_step stays the hard-forward reference / source of truth.
# ══════════════════════════════════════════════════════════════════════

def _layer_forward(w_dend, w_soma, T_p, s_in, alpha_s, alpha_d, alpha_w, config):
    """Differentiable forward of one layer. s_in: (T, K) → o_train: (T, N)."""
    dend_inputs = s_in @ w_dend.T   # (T, N)
    soma_inputs = s_in @ w_soma.T   # (T, N)
    T = s_in.shape[0]
    N = w_dend.shape[0]
    f = s_in.dtype
    time_indices = jnp.arange(T, dtype=jnp.int32)

    vth, gamma, mu_th = config.v_th, config.gamma, config.mu_th
    beta_s, beta_d, beta_s_dend = config.beta_s, config.beta_d, config.beta_s_dend
    a_adapt, b_adapt = config.a_adapt, config.b_adapt
    use_roof = config.dend_surrogate_roof   # static bool → picks the dend surrogate

    def step(carry, inp):
        mu_p, v_p, h_p, tprime_p, matp_p, w_p = carry
        dend_in, soma_in, t = inp

        # ── dendrite (h_p, tprime_p are hard ints → gates carry no gradient) ──
        tprime = jnp.where(t == 0, 0, jnp.where(h_p == 1, tprime_p, t))
        mu = jnp.where(t > 0, alpha_d * mu_p + (1 - h_p) * dend_in, dend_in)
        matp = jnp.where(h_p == 0, mu, matp_p)               # differentiable latch
        plateau_dur = t - tprime
        dur_ok = (plateau_dur <= T_p) & (plateau_dur >= 0)

        h_hard = jnp.where((matp >= mu_th) & dur_ok, 1, 0).astype(jnp.int32)
        # Plateau surrogate, UNGATED by the duration window — matches Soft_Reset's
        # hp_hidden (applied every step). value = h_hard; grad d/dmatp =
        # surrogate_sigma(matp - mu_th, beta_d).
        h_grad = sg_spike(matp - mu_th, beta_d)
        h_soft = _straight_through(h_hard.astype(f), h_grad)

        # ── soma ──
        v_pre = jnp.where(t > 0, alpha_s * v_p + soma_in - w_p, soma_in)
        v_th_eff = vth - gamma * h_hard.astype(f)            # hard effective threshold
        # Two-path spike: soma edge uses surrogate_sigma(·,beta_s); dend edge
        # (via h) uses the roof or the gamma·bump with beta_s_dend.
        o_soft = somatic_spike(v_pre, h_soft, use_roof, vth, gamma, beta_s, beta_s_dend)

        # Soft reset, DETACHED: gradient does not flow back through the reset.
        v = v_pre - lax.stop_gradient(o_soft * v_th_eff)
        w = alpha_w * w_p + (1 - alpha_w) * a_adapt * v_pre + b_adapt * o_soft

        return (mu, v, h_hard, tprime, matp, w), o_soft

    init = (
        jnp.zeros(N, dtype=f), jnp.zeros(N, dtype=f),
        jnp.zeros(N, dtype=jnp.int32), jnp.zeros(N, dtype=jnp.int32),
        jnp.zeros(N, dtype=f), jnp.zeros(N, dtype=f),
    )
    _, o_train = lax.scan(step, init, (dend_inputs, soma_inputs, time_indices))
    return o_train   # (T, N), hard values with surrogate gradient


# ══════════════════════════════════════════════════════════════════════
#  Full forward: L two-compartment layers → LI readout.
#
#  params = {"w_dend": [ (N0,K), (N1,N0), ... ],
#            "w_soma": [ ... same shapes ... ],
#            "w_readout": (J, N_last)}
#  T_p    = tuple of per-layer (N,) int arrays.
# ══════════════════════════════════════════════════════════════════════

def _sample_forward(params, T_p, x,
                    alpha_s, alpha_d, alpha_m, alpha_w, config,
                    rng_key, dropout_rate):
    """One sample. Returns (mean_readout_voltage (J,), per_layer_rates tuple).

    per_layer_rates[i] is the (N_i,) per-neuron mean firing rate (pre-dropout),
    used by the firing-rate penalty and the activity diagnostic. Dropout only
    masks what the NEXT layer / readout sees; at dropout_rate=0 the mask is
    all-ones (bernoulli p=1) and the scale is 1, so the forward is deterministic
    and identical to inference.
    """
    w_dend_list = params["w_dend"]
    w_soma_list = params["w_soma"]
    w_readout = params["w_readout"]
    L = len(w_dend_list)
    T = x.shape[0]
    f = x.dtype
    dropout_scale = 1.0 / (1.0 - dropout_rate)
    layer_keys = random.split(rng_key, max(L, 1))

    s = x
    rates = []
    for i in range(L):
        o_train = _layer_forward(
            w_dend_list[i], w_soma_list[i], T_p[i], s,
            alpha_s, alpha_d, alpha_w, config,
        )                                          # (T, N_i)
        rates.append(jnp.mean(o_train, axis=0))    # (N_i,) per-neuron rate
        mask = random.bernoulli(
            layer_keys[i], 1.0 - dropout_rate, o_train.shape
        ).astype(f)
        s = o_train * mask * dropout_scale

    # LI readout over the final (masked) spike train.
    J = w_readout.shape[0]

    def r_step(carry, s_t):
        v, sum_v = carry
        v, sum_v = LINeuron.integrate(v, sum_v, s_t, w_readout, alpha_m)
        return (v, sum_v), None

    (_, sum_v), _ = lax.scan(
        r_step, (jnp.zeros(J, dtype=f), jnp.zeros(J, dtype=f)), s
    )
    return sum_v / T, tuple(rates)


def _batch_loss(params, T_p, x_b, tgt_b,
                alpha_s, alpha_d, alpha_m, alpha_w, config,
                loss_temperature, loss_count_bias,
                keys, dropout_rate, rate_reg_strength, rate_target):
    """Mean cross-entropy + firing-rate penalty over a batch.

    has_aux payload is the per-sample mean voltages (B, J) for predictions.
    The rate penalty (λ/2) Σ_n max(0, r_n - target)² is added directly to the
    loss; autograd differentiates it through the surrogate spikes, so it rides
    both compartments of every layer exactly as the e-prop hand-fold intended.
    """
    def per_sample(x, tgt, key):
        mean_v, rates = _sample_forward(
            params, T_p, x, alpha_s, alpha_d, alpha_m, alpha_w, config,
            key, dropout_rate,
        )
        scaled = mean_v / loss_temperature + loss_count_bias
        ce = -jnp.sum(tgt * jax.nn.log_softmax(scaled))
        reg = 0.0
        for r_n in rates:
            reg = reg + jnp.sum(jnp.maximum(r_n - rate_target, 0.0) ** 2)
        return ce + 0.5 * rate_reg_strength * reg, mean_v

    losses, mean_vs = vmap(per_sample)(x_b, tgt_b, keys)
    return jnp.mean(losses), mean_vs


# ── Pre-compiled entry points ─────────────────────────────────────────
_batch_value_and_grad = jit(
    jax.value_and_grad(_batch_loss, argnums=0, has_aux=True)
)
_forward_single = jit(_sample_forward)
_forward_batch = jit(
    vmap(
        _sample_forward,
        in_axes=(None, None, 0, None, None, None, None, None, 0, None),
    )
)


# ══════════════════════════════════════════════════════════════════════
#  Pytree optimizers (SGD / AdamW) — depth-agnostic via tree_map.
#  Standard DESCENT: w ← w - lr·g (- lr·λ·w decoupled weight decay).
# ══════════════════════════════════════════════════════════════════════

@jit
def _sgd_update(params, grads, lr, weight_decay):
    return jax.tree_util.tree_map(
        lambda p, g: p - lr * g - lr * weight_decay * p, params, grads
    )


@jit
def _adamw_update(params, grads, m, v, step, lr, beta1, beta2, eps, weight_decay):
    tm = jax.tree_util.tree_map
    m = tm(lambda m_, g: beta1 * m_ + (1 - beta1) * g, m, grads)
    v = tm(lambda v_, g: beta2 * v_ + (1 - beta2) * g * g, v, grads)
    bc1 = 1 - beta1 ** step
    bc2 = 1 - beta2 ** step
    params = tm(
        lambda p, m_, v_: p
        - lr * (m_ / bc1) / (jnp.sqrt(v_ / bc2) + eps)
        - lr * weight_decay * p,
        params, m, v,
    )
    return params, m, v


def _global_norm(leaves):
    if not leaves:
        return 0.0
    return float(jnp.sqrt(sum(jnp.sum(jnp.square(l)) for l in leaves)))


# ══════════════════════════════════════════════════════════════════════
#  Network — flexible feed-forward stack of two-comp layers + LI readout.
# ══════════════════════════════════════════════════════════════════════

class Network:
    def __init__(
        self,
        key: jnp.ndarray,
        n_inputs: int,
        n_hidden,                      # int or list[int] of per-layer widths
        n_outputs: int,
        config: NeuronConfig,
        optimizer: str = "sgd",
        beta1: float = 0.9,
        beta2: float = 0.999,
        adam_eps: float = 1e-8,
        dropout_rate: float = 0.0,
        weight_decay: float = 0.0,
    ):
        if isinstance(n_hidden, int):
            n_hidden = [n_hidden]
        self.n_inputs = n_inputs
        self.n_hidden = list(n_hidden)
        self.n_outputs = n_outputs
        self.config = config
        self.optimizer = optimizer
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay

        keys = random.split(key, len(self.n_hidden) + 2)
        key_rng = keys[-1]
        key_r = keys[-2]

        # One TwoCompNeuron per hidden layer (reuses its Xavier init + T_p
        # sampling); fan-in is the previous layer's width, input for layer 0.
        self.hidden_layers = []
        fan_in = n_inputs
        for i, width in enumerate(self.n_hidden):
            self.hidden_layers.append(
                TwoCompNeuron(keys[i], width, fan_in, config)
            )
            fan_in = width
        self.readout = LINeuron(key_r, n_outputs, fan_in, config)
        self.rng_key = key_rng

        # Alphas are shared across layers (one config); take the canonical copy.
        self.alpha_s = self.hidden_layers[0].alpha_s
        self.alpha_d = self.hidden_layers[0].alpha_d
        self.alpha_w = self.hidden_layers[0].alpha_w
        self.alpha_m = self.readout.alpha_m

        if optimizer == "adam":
            self.beta1 = beta1
            self.beta2 = beta2
            self.adam_eps = adam_eps
            self.adam_step = jnp.array(0, dtype=jnp.int32)
            zeros = jax.tree_util.tree_map(jnp.zeros_like, self._params_tree())
            self.m = zeros
            self.v = jax.tree_util.tree_map(jnp.zeros_like, self._params_tree())

    # ── Parameter <-> pytree plumbing ──

    def _params_tree(self):
        return {
            "w_dend": [l.w_dend for l in self.hidden_layers],
            "w_soma": [l.w_soma for l in self.hidden_layers],
            "w_readout": self.readout.w,
        }

    def _set_params_tree(self, p):
        for i, l in enumerate(self.hidden_layers):
            l.w_dend = p["w_dend"][i]
            l.w_soma = p["w_soma"][i]
        self.readout.w = p["w_readout"]

    def _T_p(self):
        return tuple(l.T_p for l in self.hidden_layers)

    def _alphas(self):
        return (self.alpha_s, self.alpha_d, self.alpha_m, self.alpha_w)

    def _smooth_targets(self, targets):
        """Scalar label or (B,) labels → smoothed one-hot vector(s)."""
        cfg = self.config
        one_hot = jnp.eye(self.n_outputs)[targets]
        return one_hot * (1 - cfg.loss_label_smoothing) + cfg.loss_label_smoothing / self.n_outputs

    def _next_key(self):
        self.rng_key, subkey = random.split(self.rng_key)
        return subkey

    # ── Training ──

    def _apply_update(self, params, grads, lr):
        if self.optimizer == "adam":
            self.adam_step = self.adam_step + 1
            params, self.m, self.v = _adamw_update(
                params, grads, self.m, self.v, self.adam_step, lr,
                self.beta1, self.beta2, self.adam_eps, self.weight_decay,
            )
        else:
            params = _sgd_update(params, grads, lr, self.weight_decay)
        self._set_params_tree(params)

    def _step(self, x_batch, targets, lr):
        """Core update on a batch. Returns (loss, preds (B,), gnorms)."""
        B = x_batch.shape[0]
        keys = random.split(self._next_key(), B)
        params = self._params_tree()
        (loss, mean_vs), grads = _batch_value_and_grad(
            params, self._T_p(), x_batch, self._smooth_targets(targets),
            *self._alphas(), self.config,
            self.config.loss_temperature, self.config.loss_count_bias,
            keys, self.dropout_rate,
            self.config.rate_reg_strength, self.config.rate_target,
        )
        gnorms = {
            "dend": _global_norm(grads["w_dend"]),
            "soma": _global_norm(grads["w_soma"]),
            "readout": float(jnp.linalg.norm(grads["w_readout"])),
        }
        self._apply_update(params, grads, lr)
        return float(loss), jnp.argmax(mean_vs, axis=1), gnorms

    def train_step(self, x_input, target, lr=1e-3):
        """Train on one sample. Returns (loss, prediction, grad_norms_dict)."""
        loss, preds, gnorms = self._step(
            x_input[None, ...], jnp.asarray([int(target)]), lr,
        )
        return loss, int(preds[0]), gnorms

    def batch_train_step(self, x_batch, targets, lr=1e-3):
        """Train on B samples in parallel. Returns (loss, preds (B,), gnorms)."""
        loss, preds, gnorms = self._step(x_batch, jnp.asarray(targets), lr)
        return loss, preds, gnorms

    # ── Inference ──

    def predict(self, x_input):
        mean_v, _ = _forward_single(
            self._params_tree(), self._T_p(), x_input,
            *self._alphas(), self.config, random.PRNGKey(0), 0.0,
        )
        return int(jnp.argmax(mean_v))

    def batch_predict(self, x_batch):
        B = x_batch.shape[0]
        keys = random.split(random.PRNGKey(0), B)
        mean_v, _ = _forward_batch(
            self._params_tree(), self._T_p(), x_batch,
            *self._alphas(), self.config, keys, 0.0,
        )
        return jnp.argmax(mean_v, axis=1)

    def activity(self, x_batch):
        """Per-layer mean firing rate over a batch (no dropout).

        Rate = spikes / (T * n_neurons). Returns {"hidden0": r0, ...}.
        """
        B = x_batch.shape[0]
        keys = random.split(random.PRNGKey(0), B)
        _, rates = _forward_batch(
            self._params_tree(), self._T_p(), x_batch,
            *self._alphas(), self.config, keys, 0.0,
        )
        # rates[i]: (B, N_i) per-neuron rate; mean over batch and neurons.
        return {f"hidden{i}": float(jnp.mean(r)) for i, r in enumerate(rates)}

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, path, extra=None):
        """Serialize the trained model to a compressed .npz at ``path``."""
        meta = {
            "arch": {
                "n_inputs": int(self.n_inputs),
                "n_hidden": [int(h) for h in self.n_hidden],
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

        arrays = {
            "w_readout": np.asarray(self.readout.w),
            "alpha_s": np.asarray(self.alpha_s),
            "alpha_d": np.asarray(self.alpha_d),
            "alpha_w": np.asarray(self.alpha_w),
            "alpha_m": np.asarray(self.alpha_m),
        }
        for i, l in enumerate(self.hidden_layers):
            arrays[f"w_dend_{i}"] = np.asarray(l.w_dend)
            arrays[f"w_soma_{i}"] = np.asarray(l.w_soma)
            arrays[f"T_p_{i}"] = np.asarray(l.T_p)

        if self.optimizer == "adam":
            arrays["adam_step"] = np.asarray(self.adam_step)
            arrays["m_w_readout"] = np.asarray(self.m["w_readout"])
            arrays["v_w_readout"] = np.asarray(self.v["w_readout"])
            for i in range(len(self.hidden_layers)):
                arrays[f"m_w_dend_{i}"] = np.asarray(self.m["w_dend"][i])
                arrays[f"m_w_soma_{i}"] = np.asarray(self.m["w_soma"][i])
                arrays[f"v_w_dend_{i}"] = np.asarray(self.v["w_dend"][i])
                arrays[f"v_w_soma_{i}"] = np.asarray(self.v["w_soma"][i])

        arrays["meta"] = np.array(json.dumps(meta))
        np.savez_compressed(path, **arrays)
        return path

    @classmethod
    def load(cls, path):
        """Rebuild a Network from a .npz written by ``save``."""
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        arch = meta["arch"]
        build = meta["build"]
        config = NeuronConfig(**meta["config"])

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

        net.readout.w = arr("w_readout")
        net.alpha_s = arr("alpha_s")
        net.alpha_d = arr("alpha_d")
        net.alpha_w = arr("alpha_w")
        net.alpha_m = arr("alpha_m")
        for i, l in enumerate(net.hidden_layers):
            l.w_dend = arr(f"w_dend_{i}")
            l.w_soma = arr(f"w_soma_{i}")
            l.T_p = arr(f"T_p_{i}").astype(jnp.int32)

        if build["optimizer"] == "adam" and "adam_step" in data.files:
            net.adam_step = arr("adam_step").astype(jnp.int32)
            net.m["w_readout"] = arr("m_w_readout")
            net.v["w_readout"] = arr("v_w_readout")
            for i in range(len(net.hidden_layers)):
                net.m["w_dend"][i] = arr(f"m_w_dend_{i}")
                net.m["w_soma"][i] = arr(f"m_w_soma_{i}")
                net.v["w_dend"][i] = arr(f"v_w_dend_{i}")
                net.v["w_soma"][i] = arr(f"v_w_soma_{i}")

        net.meta = meta
        return net
