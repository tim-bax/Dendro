import jax.numpy as jnp
from jax import jit
from flax import struct


@struct.dataclass
class NeuronConfig:
    mu_th: float = 1.0
    v_th: float = 1.0
    gamma: float = 0.5
    tau_soma: float = 15.0
    tau_dend: float = 15.0
    tau_plat_min: float = 100.0
    tau_plat_max: float = 350.0
    dt: float = 1.0
    tau_m: float = 20.0
    v_reset: float = 0.0
    tau_w: float = 100.0
    a_adapt: float = 0.0
    b_adapt: float = 0.0
    # Resonance controls. f_max_hz sets the a_adapt clip ceiling A_MAX = a_for_f(f_max_hz);
    # adapt_init selects per-neuron init: "hetero" (uniform over the resonant band, the
    # source of somatic frequency diversity) or "warm" (homogeneous a_adapt/b_adapt above).
    f_max_hz: float = 20.0
    # Static (non-pytree) — a string can't be a jit-traced leaf, and it's only read at
    # construction, never inside the jit'd forward.
    adapt_init: str = struct.field(pytree_node=False, default="hetero")
    beta_s: float = 0.36
    beta_d: float = 0.75
    weight_scale: float = 0.15
    loss_temperature: float = 5.0
    loss_count_bias: float = 0.1
    loss_label_smoothing: float = 0.2


@jit
def surrogate_sigma(x: jnp.ndarray, beta: float) -> jnp.ndarray:
    return 1.0 / (1.0 + beta * jnp.abs(x)) ** 2


def a_for_f(alpha_s, alpha_w, f_hz, dt):
    """Subthreshold-adaptation coupling `a` that puts the soma resonance at f_hz.

    Inverts cos φ = (α_s + α_w − (1−α_w)·a) / (2√(α_s α_w)) with φ = 2π·f·dt/1000
    (dt in ms). At a = a_for_f(f) the (v, w) subsystem's complex eigenvalues sit at
    angle φ, i.e. it rings at f_hz. Used as the clip ceiling A_MAX and the upper end
    of the heterogeneous init band. Δt-general.
    """
    return (alpha_s + alpha_w
            - 2.0 * jnp.sqrt(alpha_s * alpha_w) * jnp.cos(2.0 * jnp.pi * f_hz * dt / 1000.0)
            ) / (1.0 - alpha_w)
