"""SoA state structs (flax.struct.dataclass).

``DynamicsState`` is the dynamics carry — the six slots the two-compartment
equations evolve, shared by training, inference, and the activity diagnostic so they
cannot drift apart. In the multi-layer scan there is one ``DynamicsState`` per hidden
layer (a tuple rides the carry).

``EligState`` is the training-only eligibility carry — the four traces the analytic
gradient accumulates over: the somatic (``eps_s``), the dendritic (``eps_d``) and its
plateau-onset latch (``eps_d_bar``), and the readout (``eps_r``). One per hidden
layer; ``eps_r`` is meaningful only on the top layer's slot (the readout sees the top
hidden layer's spikes) and rides there.

The J-collapsed gradient state:

  - ``LayerCredit`` (``G_s`` (N,K), ``G_d`` (N,K)) is the per-hidden-layer
    accumulator; the readout weights (or the routing matrix) contract it post-scan.
  - ``LayerIngredients`` (``g_s``,``g_d``,``sp``) is the transient per-layer
    per-timestep credit contribution the sweep emits (never carried across the
    scan). ``sp`` (the somatic surrogate factor) is emitted for the coupled
    soma->soma running product; the decoupled/DFA accumulation ignores it.
  - ``TrainAccum`` (``G_r`` (N_top,), ``credit`` = tuple[LayerCredit]) is what
    ``forward_accumulate`` returns: the shared, routing-invariant readout accumulator
    plus the per-hidden-layer credit carrier.
  - ``TrainableParams`` (``w_dend`` tuple, ``w_soma`` tuple, ``w_readout``) is the
    trainable-weight tree — used as the gradient tree (``loss_and_grads``) and,
    sharing its treedef, as the weight tree (``Network.trainable()``) so an
    optimizer maps leaf-wise.

The ``h`` and ``t_prime`` fields are int32 and must stay int32: the ``lax.scan`` carry
must round-trip its dtypes exactly under x64, and a stray upcast to int64 makes the
scan fail. ``DynamicsState.zeros`` builds them with the right per-field dtype. See
docs/architecture.md.
"""
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class DynamicsState:
    """Two-compartment dynamics carry (eqs 17-23).

    ``w`` is the adaptation current — identically zero when adaptation is off
    (a_adapt = b_adapt = 0), but carried so the shared dynamics also support the
    adaptation path.
    """
    mu: jnp.ndarray            # dendritic potential
    v: jnp.ndarray             # somatic membrane (post-reset)
    h: jnp.ndarray             # plateau flag (int32)
    t_prime: jnp.ndarray       # plateau-onset time index (int32)
    mu_at_tprime: jnp.ndarray  # dendritic potential latched at plateau onset
    w: jnp.ndarray             # adaptation current

    @classmethod
    def zeros(cls, n_neurons, batch=None):
        """Zero carry. batch=None -> one sample, int -> batched (leading axis)."""
        s = (batch, n_neurons) if batch else (n_neurons,)
        return cls(
            mu=jnp.zeros(s),
            v=jnp.zeros(s),
            h=jnp.zeros(s, dtype=jnp.int32),
            t_prime=jnp.zeros(s, dtype=jnp.int32),
            mu_at_tprime=jnp.zeros(s),
            w=jnp.zeros(s),
        )


@struct.dataclass
class EligState:
    """Training-only eligibility carry — the four traces the gradient sums over.

    ``eps_s`` is the somatic eligibility (one per input), ``eps_d``/``eps_d_bar`` the
    dendritic trace and its value latched at plateau onset, and ``eps_r`` the readout
    eligibility. No J axis — the readout weights contract onto these post-scan. One per
    hidden layer; ``eps_r`` is used only on the top layer's slot. All float64 (no int
    fields, so no scan-carry dtype hazard here).
    """
    eps_s: jnp.ndarray      # somatic eligibility        (K,)
    eps_d: jnp.ndarray      # dendritic eligibility       (N,K)
    eps_d_bar: jnp.ndarray  # dendritic trace at plateau  (N,K)
    eps_r: jnp.ndarray      # readout eligibility         (N,)

    @classmethod
    def zeros(cls, n_neurons, n_inputs, batch=None):
        """Zero carry. batch=None -> one sample, int -> batched (leading axis)."""
        sk = (batch, n_inputs) if batch else (n_inputs,)
        sn = (batch, n_neurons) if batch else (n_neurons,)
        snk = (batch, n_neurons, n_inputs) if batch else (n_neurons, n_inputs)
        return cls(
            eps_s=jnp.zeros(sk),
            eps_d=jnp.zeros(snk),
            eps_d_bar=jnp.zeros(snk),
            eps_r=jnp.zeros(sn),
        )


@struct.dataclass
class LayerCredit:
    """Per-hidden-layer J-collapsed gradient accumulator.

    ``G_s`` (N,K) = Σ_t ψ^s ⊗ eps_s, ``G_d`` (N,K) = Σ_t (ψ^s ψ^d γ) · eps_d_bar. The
    readout weights (or the routing matrix) contract these post-scan. ``G_r`` is
    routing-invariant and shared across the network, so it is not here — it rides
    ``TrainAccum``.
    """
    G_s: jnp.ndarray  # somatic accumulator   (N,K)
    G_d: jnp.ndarray  # dendritic accumulator (N,K)

    @classmethod
    def zeros(cls, n_neurons, n_inputs, batch=None):
        """Zero accumulator. batch=None -> one sample, int -> batched."""
        snk = (batch, n_neurons, n_inputs) if batch else (n_neurons, n_inputs)
        return cls(G_s=jnp.zeros(snk), G_d=jnp.zeros(snk))


@struct.dataclass
class LayerIngredients:
    """Transient per-layer per-timestep credit ingredients emitted by the sweep.

    Never carried across the scan — produced by ``TwoCompLayer.credit_ingredients``
    and consumed by the strategy's ``step`` within the same timestep. ``g_s``/``g_d``
    are the summands of ``LayerCredit``'s ``G_s``/``G_d``; ``sp`` is the somatic
    surrogate factor, emitted for the coupled soma->soma running product (the
    decoupled/DFA accumulation ignores it).
    """
    g_s: jnp.ndarray  # (N,K)
    g_d: jnp.ndarray  # (N,K)
    sp: jnp.ndarray   # (N,)


@struct.dataclass
class TrainAccum:
    """What ``forward_accumulate`` returns.

    ``G_r`` (N_top,) is the shared, method-invariant readout accumulator; ``credit``
    is the strategy-shaped per-hidden-layer carrier (``tuple[LayerCredit]`` for the
    decoupled/independent accumulation used by dfa and the single-layer coupled base
    case; the coupled cross-layer running product when that lands).
    """
    G_r: jnp.ndarray  # readout accumulator (N_top,)
    credit: tuple      # tuple[LayerCredit, ...] — one per hidden layer


@struct.dataclass
class TrainableParams:
    """The trainable-weight tree — per-hidden-layer dendrite/soma + the readout.

    Used as the gradient tree (``loss_and_grads``) and, sharing its treedef, as the
    weight tree (``Network.trainable()``) so an optimizer maps leaf-wise. ``w_dend`` /
    ``w_soma`` are tuples (one (N,K) per hidden layer, bottom-up); ``w_readout`` is
    (J, N_top). Excludes the fixed intrinsics (T_p / alpha_* / B / config), which are
    not trained.
    """
    w_dend: tuple           # tuple[(N,K), ...] per hidden layer
    w_soma: tuple           # tuple[(N,K), ...] per hidden layer
    w_readout: jnp.ndarray  # (J, N_top)
