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
    beta_s: float = 0.36
    beta_d: float = 0.75
    weight_scale: float = 0.15
    loss_temperature: float = 5.0
    loss_count_bias: float = 0.1
    loss_label_smoothing: float = 0.2
    # sparch-style BatchNorm1d on the per-layer pre-synaptic current (Bittar &
    # Garner). "batchnorm" enables it; "none" disables (then this variant equals
    # BPTT_fix_auto). bn_targets picks which of the two projections get a BN:
    # "both" | "soma" | "dend". String fields are static (pytree_node=False) so
    # the forward can branch on them at trace time; momentum/eps are dynamic.
    normalization: str = struct.field(pytree_node=False, default="batchnorm")
    bn_targets: str = struct.field(pytree_node=False, default="both")
    bn_momentum: float = 0.05
    bn_eps: float = 1e-5


@jit
def surrogate_sigma(x: jnp.ndarray, beta: float) -> jnp.ndarray:
    return 1.0 / (1.0 + beta * jnp.abs(x)) ** 2
