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
    beta_s: float = 0.36        # somatic surrogate s'_s in the SOMATIC pathway
    beta_s_dend: float = 0.36   # somatic surrogate s'_s in the DENDRITIC pathway
                                # (∂o/∂h = γ·s'_s). Set == beta_s for single-beta.
    beta_d: float = 0.75        # dendritic plateau surrogate s'_d
    weight_scale: float = 0.15
    loss_temperature: float = 5.0
    loss_count_bias: float = 0.1
    loss_label_smoothing: float = 0.2


@jit
def surrogate_sigma(x: jnp.ndarray, beta: float) -> jnp.ndarray:
    return 1.0 / (1.0 + beta * jnp.abs(x)) ** 2
