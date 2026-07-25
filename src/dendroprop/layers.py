"""Layer modules — the middle level of abstraction.

Three levels of abstraction:

  1. single two-compartment neuron — absent (vectorized over ``(N,)/(N,K)``; there is
     no per-neuron object, and there must not be one).
  2. single two-compartment hidden layer — ``TwoCompLayer`` (this module).
  3. network = stack of hidden layers + LI readout — ``network.Network``.

Layers are ``flax.struct.dataclass`` pytrees of parameters + pure methods, not mutable
objects. A layer owns its trained weights, its fixed constants (``alpha_*``, ``T_p``),
the two per-layer credit-assignment components (the ``surrogate`` and the
``dendritic_credit`` option), and the pure methods that advance its dynamics
(``dyn_step``), emit its eligibility/credit ingredients on-the-fly
(``credit_ingredients``), and assemble its gradients post-scan (``assemble_grads``).
The network-level component (the credit routing) lives on ``Network``.

Backward topology (do not change): inter-layer credit propagates soma(L) -> soma(L-1)
only (the network-level coupled ``soma_backward``). The dendrite's gradient is local:
``assemble_grads`` credits grad_d from this layer's own soma error and never propagates
it to an earlier layer. A dendrite -> previous-layer edge would branch the plateau
trace at every layer, scaling as 2**L for early layers, so there is no
``dendrite_backward``. See docs/architecture.md.
"""
from typing import Protocol, runtime_checkable

import jax.numpy as jnp
from flax import struct

from .dynamics import dynamics_step, readout_integrate
from .surrogate import get_surrogate
from .state import LayerIngredients


def update_somatic_eligibility(eps_s_prev, pre_spike_t, alpha_s):
    """Somatic eligibility trace."""
    return alpha_s * eps_s_prev + pre_spike_t


def update_dendritic_eligibility(eps_d_prev, eps_d_bar_prev, pre_spike_t, h_prev, alpha_d):
    """Dendritic eligibility trace + its plateau-onset latch. ``alpha_d`` is a scalar;
    reshape to (1,1) broadcasts over the (N,K) trace."""
    eps_d = jnp.reshape(alpha_d, (-1, 1)) * eps_d_prev + (1 - h_prev[:, None]) * pre_spike_t[None, :]
    eps_d_bar = jnp.where((h_prev == 0)[:, None], eps_d, eps_d_bar_prev)
    return eps_d, eps_d_bar


@runtime_checkable
class HiddenLayer(Protocol):
    """Structural contract the ``Network`` duck-types over — typing/docs only.

    ``TwoCompLayer`` *satisfies* this without inheriting it (no ABC, no hierarchy).
    Never ``isinstance``-gate the hot path on it; it documents the seam so a future
    hidden-layer type slots into the ``Network`` scan without touching it.
    """

    @property
    def n(self) -> int: ...

    @property
    def n_in(self) -> int: ...

    def dyn_step(self, dyn, dend_in_t, soma_in_t, t, config): ...

    def credit_ingredients(self, elig, v_pre, h_new, h_prev, mu_at_tp, input_t, config): ...

    def assemble_grads(self, e_soma, accum, T, config): ...


@struct.dataclass
class TwoCompLayer:
    """A two-compartment (LIP dendrite + LIF soma) hidden layer, vectorized over N.

    Array leaves (traced): ``w_dend``/``w_soma`` (trained) and the fixed constants
    ``T_p`` (int32) and scalar decays ``alpha_s``/``alpha_d``/``alpha_w``. Static aux
    (jit cache key): the ``surrogate`` and ``dendritic_credit`` components.
    """

    w_dend: jnp.ndarray   # (N, K)   trained
    w_soma: jnp.ndarray   # (N, K)   trained
    T_p: jnp.ndarray      # (N,) int32   fixed per-neuron plateau window
    alpha_s: jnp.ndarray  # scalar
    alpha_d: jnp.ndarray  # scalar
    alpha_w: jnp.ndarray  # scalar
    surrogate: str = struct.field(pytree_node=False, default="fast_sigmoid")  # surrogate
    dendritic_credit: bool = struct.field(pytree_node=False, default=True)    # False=pruned

    @property
    def n(self):
        return self.w_dend.shape[0]

    @property
    def n_in(self):
        return self.w_dend.shape[1]

    # ── dynamics (shared by inference and training) ─────
    def dyn_step(self, dyn, dend_in_t, soma_in_t, t, config):
        """One timestep of the two-compartment dynamics. Wraps ``dynamics_step``.
        Returns ``(dyn, o, v_pre, h_new, h_prev, mu_at_tp)``."""
        return dynamics_step(
            dyn, dend_in_t, soma_in_t, t,
            self.alpha_s, self.alpha_d, self.T_p, config, self.alpha_w,
        )

    # ── training-only: eligibility + on-the-fly credit ingredients ─────────────
    def credit_ingredients(self, elig, v_pre, h_new, h_prev, mu_at_tp, input_t, config):
        """Advance this layer's eligibility traces and emit its per-timestep credit
        ingredients. Applies this layer's surrogate.

        ``input_t`` is the (float) presynaptic activity into this layer — the external
        input for layer 0, the previous layer's dropout-scaled spikes for a deeper
        layer. Returns ``(elig_new, LayerIngredients(g_s, g_d, sp))`` where
        ``g_s = psi^s (x) eps_s``, ``g_d = (psi^s psi^d gamma) (x) eps_d_bar`` are the
        summands of ``LayerCredit``'s ``G_s``/``G_d``, and ``sp`` (the somatic surrogate
        factor) is emitted for the coupled soma->soma running product (DFA ignores it).
        ``eps_r`` (readout eligibility) is the readout's concern and is left untouched.
        """
        eps_s = update_somatic_eligibility(elig.eps_s, input_t, self.alpha_s)
        eps_d, eps_d_bar = update_dendritic_eligibility(
            elig.eps_d, elig.eps_d_bar, input_t, h_prev, self.alpha_d,
        )
        elig_new = elig.replace(eps_s=eps_s, eps_d=eps_d, eps_d_bar=eps_d_bar)

        surrogate_fn = get_surrogate(self.surrogate)  # static field -> resolved at trace time
        sp = surrogate_fn(v_pre + config.gamma * h_new - config.v_th, config.beta_s)
        hp = surrogate_fn(mu_at_tp - config.mu_th, config.beta_d)
        g_s = sp[:, None] * eps_s[None, :]
        g_d = (sp * hp * config.gamma)[:, None] * eps_d_bar
        return elig_new, LayerIngredients(g_s=g_s, g_d=g_d, sp=sp)

    def credit_step(self, elig, v_pre, h_new, h_prev, mu_at_tp, input_t, config):
        """Thin wrapper over ``credit_ingredients`` returning the bare
        ``(elig_new, g_s, g_d)``."""
        elig_new, ing = self.credit_ingredients(
            elig, v_pre, h_new, h_prev, mu_at_tp, input_t, config
        )
        return elig_new, ing.g_s, ing.g_d

    # ── post-scan grad assembly for this layer, given its soma learning signal ─
    def assemble_grads(self, e_soma, accum, T, config):
        """``(grad_s, grad_d)`` from this layer's soma error and its ``LayerCredit``.

        ``soma_grad_scale`` divides grad_s. ``dendritic_credit`` gates grad_d — the
        dendrite's gradient is local, credited from this layer's own soma error and
        never propagated to an earlier layer. ``accum`` is duck-typed on ``.G_s``/
        ``.G_d`` (a ``LayerCredit``).
        """
        grad_s = (e_soma[:, None] * accum.G_s) / (T * config.soma_grad_scale)
        grad_d = (
            (e_soma[:, None] * accum.G_d) / T
            if self.dendritic_credit
            else jnp.zeros_like(accum.G_d)
        )
        return grad_s, grad_d


@struct.dataclass
class ReadoutLayer:
    """Leaky-integrator (LI) readout — linear, no threshold, no reset, no plateau.

    A distinct concrete type, not a ``HiddenLayer``: its gradient shape and role differ
    (it is where the top-down error ``delta`` originates), and it has no
    surrogate/dendrite. It is a singleton at a known position, so ``Network`` references
    it by concrete type in exactly one place.
    """

    w: jnp.ndarray        # (J, N)
    alpha_m: jnp.ndarray  # scalar

    @property
    def n(self):
        return self.w.shape[0]

    @property
    def n_in(self):
        return self.w.shape[1]

    def integrate(self, r_v, r_sum_v, spikes_in_t):
        """LI voltage integration for one timestep (inference — no eligibility).
        ``spikes_in_t`` are the top hidden layer's spikes. Returns ``(r_v, r_sum_v)``."""
        return readout_integrate(r_v, r_sum_v, spikes_in_t, self.w, self.alpha_m)

    def integrate_and_eligibility(self, r_v, r_sum_v, eps_r, spikes_in_t):
        """LI integration + readout eligibility for one timestep (training).
        ``spikes_in_t`` are the (dropout-scaled) top-hidden spikes. Returns
        ``(r_v, r_sum_v, eps_r)``."""
        r_v, r_sum_v = readout_integrate(r_v, r_sum_v, spikes_in_t, self.w, self.alpha_m)
        eps_r = self.alpha_m * eps_r + spikes_in_t
        return r_v, r_sum_v, eps_r

    def grad(self, delta, G_r, T):
        """Readout weight gradient ``outer(delta, G_r) / T``. Never uses the routing
        matrix, so grad_r is invariant to the Credit Assignment Method."""
        return (delta[:, None] * G_r[None, :]) / T
