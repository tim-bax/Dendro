"""Network — the top level of abstraction: a stack of hidden layers + LI readout.

``Network`` is a ``flax.struct.dataclass`` pytree that composes ``TwoCompLayer``
hidden units and a ``ReadoutLayer``, and carries the one network-level component: the
credit routing (``credit_method`` — how the top-down error reaches each hidden layer's
soma) plus its fixed random feedback matrices ``B`` (one per hidden layer for dfa;
``()`` for symmetric/backprop routing). The per-layer components (surrogate,
dendritic-credit) live on the layers.

The training pass is one all-layers time-scan: a bottom-up dynamics sweep
(``_dynamics_sweep``) with the input projection inside the scan (each layer's input is
the previous layer's dropout-scaled spikes), plus on-the-fly eligibility/credit
accumulation, no trajectory storage. Inference (``forward``/``activity``) runs the same
sweep with the credit machinery dropped and no dropout. Grad assembly happens post-scan:
the readout owns the shared, routing-invariant ``grad_r``; the ``CreditStrategy``
(dispatched once on the static ``credit_method``) routes ``delta`` onto each hidden
layer's soma and each layer assembles its own ``(grad_s, grad_d)``.

Backward topology (do not change): inter-layer credit is soma(L)->soma(L-1) only; the
dendrite's gradient is local (no ``dendrite_backward``, 2**L guard); DFA is the only
decoupled method and is fully implemented (``e^L = delta @ B_L`` per layer); the coupled
bptt/rfa soma->soma recursion for >1 hidden layer is the hand-derived
``dendroprop_methods.pdf`` math and raises via ``credit.soma_backward``. The single
hidden-layer path is checked against the frozen golden (within rtol=1e-9 from the
projection-inside-loop reassociation); ``dynamics.predict``/``activity`` are the
bit-for-bit single-layer reference. See docs/architecture.md.

The module-level free functions at the bottom (``forward_accumulate`` etc.) are the
jit/vmap boundary — a Network is batch-shared (``in_axes=None``); only the input and the
per-sample dropout key are batched.
"""
import jax.numpy as jnp
from jax import jit, lax, random, vmap
from flax import struct

from . import credit
from . import feedback
from .config import NeuronConfig
from .state import (
    DynamicsState, EligState, LayerCredit, TrainAccum, TrainableParams,
)
from .layers import TwoCompLayer, ReadoutLayer

_LOG_EPS = 1e-8  # guards log(0) in the cross-entropy


def smooth_targets(labels, n_classes, label_smoothing):
    """One-hot with label smoothing. Scalar label -> (J,); (B,) labels -> (B,J).

    Pure array function — the training loop applies it at the label boundary."""
    one_hot = jnp.eye(n_classes)[labels]
    return one_hot * (1 - label_smoothing) + label_smoothing / n_classes


def _split_layer_keys(dkey, n_layers):
    """Per-hidden-layer dropout keys for one timestep. At ``n_layers == 1`` return
    ``[dkey]`` unsplit so the single-layer dropout mask matches the golden bit-for-bit;
    otherwise ``random.split(dkey, n_layers)``."""
    return [dkey] if n_layers == 1 else list(random.split(dkey, n_layers))


def _dynamics_sweep(hidden, dyns, x_t, t, dkeys, rate, scale, config, apply_dropout):
    """One timestep, bottom-up over the static ``hidden`` tuple. The input projection
    is inside the loop (``dend = layer_in @ w_dend.T``). In train mode each hidden
    layer's output spikes are dropout-masked (scaled by ``1/(1-rate)``) before they feed
    the next layer's dynamics + eligibility and the readout; the external input to layer
    0 (``x_t``, already float64) is never dropped.

    Returns ``(new_dyns, obs_list, spikes_list)``: ``new_dyns`` a tuple of the advanced
    per-layer ``DynamicsState``; ``obs_list[i] = (v_pre, h_new, h_prev, mu_at_tp)`` the
    surrogate observables; ``spikes_list[i]`` the (possibly dropped) float64 output
    spikes of layer i. ``dkeys`` is ignored when ``apply_dropout`` is False.
    """
    new_dyns = []
    obs_list = []
    spikes_list = []
    layer_in = x_t
    for i, layer in enumerate(hidden):
        dend_in = layer_in @ layer.w_dend.T
        soma_in = layer_in @ layer.w_soma.T
        dyn_i, o, v_pre, h_new, h_prev, mu_at_tp = layer.dyn_step(
            dyns[i], dend_in, soma_in, t, config
        )
        spikes = o.astype(jnp.float64)
        if apply_dropout:
            mask = random.bernoulli(dkeys[i], 1.0 - rate, (layer.n,)).astype(jnp.float64)
            spikes = spikes * mask * scale
        new_dyns.append(dyn_i)
        obs_list.append((v_pre, h_new, h_prev, mu_at_tp))
        spikes_list.append(spikes)
        layer_in = spikes
    return tuple(new_dyns), obs_list, spikes_list


@struct.dataclass
class Network:
    """Composition of hidden ``TwoCompLayer``s + an LI ``ReadoutLayer``.

    Array leaves: everything inside ``hidden``/``readout``/``config`` and the fixed
    feedback matrices ``B``. Static aux data (jit cache key): ``credit_method`` — the
    credit routing.
    """

    hidden: tuple            # tuple[TwoCompLayer, ...]  (bottom-up)
    readout: ReadoutLayer
    config: NeuronConfig
    credit_method: str = struct.field(pytree_node=False, default="bptt")
    B: tuple = ()            # per-hidden-layer fixed random feedback matrices; () for bptt

    # ── inference forward (dynamics only — no eligibility, no credit, no dropout) ─
    def forward(self, x):
        """Mean readout voltage ``(J,)``. Network-owned N-layer dynamics scan built on
        ``_dynamics_sweep``; ``dynamics.predict`` is the frozen single-layer reference."""
        hidden, ro, cfg = self.hidden, self.readout, self.config
        T = x.shape[0]
        J = ro.n
        time_idx = jnp.arange(T, dtype=jnp.int32)

        def step(carry, inp):
            dyns, r_v, r_sum = carry
            x_t, t = inp
            new_dyns, _, spikes_list = _dynamics_sweep(
                hidden, dyns, x_t.astype(jnp.float64), t, None, 0.0, 1.0, cfg,
                apply_dropout=False,
            )
            r_v, r_sum = ro.integrate(r_v, r_sum, spikes_list[-1])
            return (new_dyns, r_v, r_sum), None

        init = (
            tuple(DynamicsState.zeros(layer.n) for layer in hidden),
            jnp.zeros(J), jnp.zeros(J),
        )
        (_, _, sum_v), _ = lax.scan(step, init, (x, time_idx))
        return sum_v / T

    def activity(self, x):
        """Mean hidden firing rate over ALL hidden neurons across all layers (spikes /
        (T * total_hidden)). For a single layer this equals ``dynamics.activity``."""
        hidden, cfg = self.hidden, self.config
        T = x.shape[0]
        total_n = sum(layer.n for layer in hidden)
        time_idx = jnp.arange(T, dtype=jnp.int32)

        def step(carry, inp):
            dyns, s = carry
            x_t, t = inp
            new_dyns, _, spikes_list = _dynamics_sweep(
                hidden, dyns, x_t.astype(jnp.float64), t, None, 0.0, 1.0, cfg,
                apply_dropout=False,
            )
            s = s + sum(jnp.sum(sp) for sp in spikes_list)
            return (new_dyns, s), None

        init = (tuple(DynamicsState.zeros(layer.n) for layer in hidden), 0.0)
        (_, s), _ = lax.scan(step, init, (x, time_idx))
        return s / (T * total_n)

    # ── training forward + on-the-fly credit accumulation (one all-layers scan) ─
    def forward_accumulate(self, x, rng_key, dropout_rate):
        """``(mean_voltage, TrainAccum)`` for one sample. The forward accumulation is
        routing-independent for the decoupled path (the strategy's ``init``/``step``
        select the carrier); routing enters the grads only in ``loss_and_grads``. The
        coupled bptt/rfa carrier for >1 hidden layer raises (PDF placeholder) at trace
        time via ``strat.init``."""
        strat = credit.get_strategy(self.credit_method)
        hidden, ro, cfg = self.hidden, self.readout, self.config
        n_layers = len(hidden)
        T = x.shape[0]
        J = ro.n
        time_idx = jnp.arange(T, dtype=jnp.int32)
        dropout_keys = random.split(rng_key, T)
        scale = 1.0 / (1.0 - dropout_rate)
        credit_init = strat.init(self)  # raises here for coupled multilayer (pre-scan)

        def step(carry, inp):
            dyns, eligs, r_v, r_sum, G_r, credit_carry = carry
            x_t, t, dkey = inp
            x_tf = x_t.astype(jnp.float64)
            dkeys = _split_layer_keys(dkey, n_layers)
            new_dyns, obs_list, spikes_list = _dynamics_sweep(
                hidden, dyns, x_tf, t, dkeys, dropout_rate, scale, cfg, apply_dropout=True
            )
            # Readout + shared readout eligibility, driven by the TOP layer's dropped
            # spikes; eps_r rides the top layer's elig slot, G_r is shared/method-free.
            top_spikes = spikes_list[-1]
            r_v, r_sum, eps_r = ro.integrate_and_eligibility(
                r_v, r_sum, eligs[-1].eps_r, top_spikes
            )
            G_r = G_r + eps_r
            # Per-layer eligibility + credit ingredients. Layer 0's input is the external
            # x; layer i>0's input is the DROPPED spikes of layer i-1.
            inputs = (x_tf,) + tuple(spikes_list[:-1])
            new_eligs = []
            ingredients = []
            for i, layer in enumerate(hidden):
                v_pre, h_new, h_prev, mu_at_tp = obs_list[i]
                elig_new, ing = layer.credit_ingredients(
                    eligs[i], v_pre, h_new, h_prev, mu_at_tp, inputs[i], cfg
                )
                new_eligs.append(elig_new)
                ingredients.append(ing)
            new_eligs[-1] = new_eligs[-1].replace(eps_r=eps_r)
            credit_carry = strat.step(credit_carry, tuple(ingredients), self)
            return (new_dyns, tuple(new_eligs), r_v, r_sum, G_r, credit_carry), None

        init = (
            tuple(DynamicsState.zeros(layer.n) for layer in hidden),
            tuple(EligState.zeros(layer.n, layer.n_in) for layer in hidden),
            jnp.zeros(J), jnp.zeros(J),
            jnp.zeros(hidden[-1].n),  # G_r (N_top,)
            credit_init,
        )
        (_, _, _, r_sum_f, G_r_f, credit_f), _ = lax.scan(
            step, init, (x, time_idx, dropout_keys)
        )
        return r_sum_f / T, TrainAccum(G_r=G_r_f, credit=credit_f)

    # ── post-scan loss + grads (CreditStrategy dispatched on static credit_method) ─
    def loss_and_grads(self, mean_voltage, accum, target_smoothed, T):
        """``(loss, prediction, TrainableParams)``. The readout owns the
        routing-invariant ``grad_r``; ``strat.finalize`` routes ``delta`` onto each
        hidden layer's soma and assembles its ``(grad_s, grad_d)``. The grad tree is
        congruent with ``trainable()`` so an optimizer maps leaf-wise."""
        ro, cfg = self.readout, self.config
        strat = credit.get_strategy(self.credit_method)

        scaled_logits = mean_voltage / cfg.loss_temperature + cfg.loss_count_bias
        probs = jnp.exp(scaled_logits - jnp.max(scaled_logits))
        probs = probs / jnp.sum(probs)

        prediction = jnp.argmax(mean_voltage)
        loss = -jnp.sum(target_smoothed * jnp.log(probs + _LOG_EPS))
        delta = target_smoothed - probs

        grad_r = ro.grad(delta, accum.G_r, T)                 # shared, routing-invariant
        per_layer = strat.finalize(self, delta, accum.credit, T)  # ((grad_s, grad_d),...)
        grad_s = tuple(g[0] for g in per_layer)
        grad_d = tuple(g[1] for g in per_layer)
        return loss, prediction, TrainableParams(w_dend=grad_d, w_soma=grad_s, w_readout=grad_r)

    # ── trainable-weight partition (pure pytree; optimizer state lives OUTSIDE) ──
    def trainable(self):
        """Extract the trainable-weight tree — per-hidden-layer ``w_dend``/``w_soma`` +
        the readout ``w`` — as a ``TrainableParams``, congruent (same treedef) with the
        grad tree ``loss_and_grads`` returns, so an optimizer maps leaf-wise. The fixed
        intrinsics (``T_p``/``alpha_*``/``B``/``config``) are excluded STRUCTURALLY, so an
        optimizer never touches them."""
        return TrainableParams(
            w_dend=tuple(layer.w_dend for layer in self.hidden),
            w_soma=tuple(layer.w_soma for layer in self.hidden),
            w_readout=self.readout.w,
        )

    def replace_trainable(self, tp):
        """Return a new ``Network`` with the trainable weights replaced from a
        ``TrainableParams`` tree; every fixed intrinsic and static field carries over.
        ``net.replace_trainable(net.trainable())`` is the identity."""
        new_hidden = tuple(
            layer.replace(w_dend=wd, w_soma=ws)
            for layer, wd, ws in zip(self.hidden, tp.w_dend, tp.w_soma)
        )
        return self.replace(hidden=new_hidden, readout=self.readout.replace(w=tp.w_readout))


def build_network(key, dims, config, method="bptt", *,
                  surrogate="fast_sigmoid", dendritic_credit=True):
    """Fresh-init a ``Network`` under the fixed key-split order the frozen
    ``init_recipe.npz`` was captured under (so the single-layer init is reproducible).

    ``dims = (n_inputs, [N_1, ..., N_H], n_outputs)`` (hidden dims bottom-up). Recipe:
    ``alpha_s/d/w/m = exp(-dt/tau_*)``; per hidden layer split ``(k1, k2, k3)`` and draw
    ``w_dend`` from k1, ``w_soma`` from k2 (both ``~N * sqrt(2/n_in) * weight_scale``),
    ``T_p = (U[tau_plat_min, tau_plat_max] / dt).int32`` from k3; the readout weight
    ``~N * sqrt(2/N_top) * weight_scale``. Feedback ``B``: for dfa one ``(J, N_L)`` per
    hidden layer; for rfa the single readout edge ``(J, N_top)`` — rfa multilayer raises
    (placeholder). bptt draws no ``B``.
    """
    n_inputs, hidden_dims, n_outputs = dims
    routing = credit.credit_method(method)
    key_h, key_r, key_b = random.split(key, 3)

    alpha_s = jnp.exp(-config.dt / config.tau_soma)
    alpha_d = jnp.exp(-config.dt / config.tau_dend)
    alpha_w = jnp.exp(-config.dt / config.tau_w)
    alpha_m = jnp.exp(-config.dt / config.tau_m)

    layer_keys = [key_h] if len(hidden_dims) == 1 else list(random.split(key_h, len(hidden_dims)))
    layers = []
    prev = n_inputs
    for lk, n in zip(layer_keys, hidden_dims):
        k1, k2, k3 = random.split(lk, 3)
        scale = jnp.sqrt(2.0 / prev) * config.weight_scale
        w_dend = random.normal(k1, (n, prev)) * scale
        w_soma = random.normal(k2, (n, prev)) * scale
        tau_plat = random.uniform(
            k3, (n,), minval=config.tau_plat_min, maxval=config.tau_plat_max
        )
        T_p = (tau_plat / config.dt).astype(jnp.int32)
        layers.append(TwoCompLayer(
            w_dend=w_dend, w_soma=w_soma, T_p=T_p,
            alpha_s=alpha_s, alpha_d=alpha_d, alpha_w=alpha_w,
            surrogate=surrogate, dendritic_credit=dendritic_credit,
        ))
        prev = n

    r_scale = jnp.sqrt(2.0 / prev) * config.weight_scale
    readout = ReadoutLayer(w=random.normal(key_r, (n_outputs, prev)) * r_scale, alpha_m=alpha_m)

    B = ()
    if credit.needs_feedback(routing):
        if routing == "rfa" and len(hidden_dims) > 1:
            raise NotImplementedError(
                "rfa multilayer feedback topology is the hand-derived PDF math "
                "(coupled soma->soma); only dfa multilayer feedback is implemented"
            )
        b_keys = list(random.split(key_b, len(hidden_dims)))
        B = tuple(
            feedback.init_feedback_weights(bk, n_outputs, n, config)
            for bk, n in zip(b_keys, hidden_dims)
        )

    return Network(hidden=tuple(layers), readout=readout, config=config,
                   credit_method=routing, B=B)


# ══════════════════════════════════════════════════════════════════════════════
#  jit/vmap boundary — the public training/inference API. The whole Network is
#  batch-shared (in_axes=None); only x and the dropout key are batched.
# ══════════════════════════════════════════════════════════════════════════════

def _net_forward(net, x):
    return net.forward(x)


def _net_activity(net, x):
    return net.activity(x)


def _net_forward_accumulate(net, x, key, rate):
    return net.forward_accumulate(x, key, rate)


def _net_loss_and_grads(net, mv, accum, sm, T):
    return net.loss_and_grads(mv, accum, sm, T)


predict = jit(_net_forward)
activity = jit(_net_activity)
forward_accumulate = jit(_net_forward_accumulate)
loss_and_grads = jit(_net_loss_and_grads)

predict_batch = jit(vmap(_net_forward, in_axes=(None, 0)))
activity_batch = jit(vmap(_net_activity, in_axes=(None, 0)))
forward_accumulate_batch = jit(vmap(_net_forward_accumulate, in_axes=(None, 0, 0, None)))
loss_and_grads_batch = jit(vmap(_net_loss_and_grads, in_axes=(None, 0, 0, 0, None)))
