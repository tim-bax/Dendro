import jax.numpy as jnp
from jax import jit
from flax import struct


@struct.dataclass
class NeuronConfig:
    """All these values are overwritten by command line flags, so they are not hyperparameters used by default.
    """

    # ── Thresholds and compartment coupling ──
    mu_th: float = 1.0            # dendritic plateau threshold
    v_th: float = 1.0             # somatic spike threshold
    gamma: float = 0.5            # plateau-induced threshold drop: v_th_eff = v_th - gamma*h

    # ── Time constants (ms) ──
    tau_soma: float = 15.0        # somatic membrane
    tau_dend: float = 15.0        # dendritic membrane
    tau_m: float = 20.0           # LI readout membrane
    tau_plat_min: float = 100.0   # plateau duration T_p ~ U[min, max], sampled per neuron
    tau_plat_max: float = 350.0
    dt: float = 4.0               # timestep; set from --bin_size_ms, NOT its own flag

    # ── Adaptation (disabled by default) ──
    tau_w: float = 100.0          # adaptation current time constant
    a_adapt: float = 0.0          # subthreshold coupling; 0.0 = off
    b_adapt: float = 0.0          # spike-triggered jump; 0.0 = off

    # ── Surrogate gradient sharpness (see surrogate_sigma) ──
    beta_s: float = 1.0           # somatic surrogate, somatic gradient path
    beta_s_dend: float = 1.0      # somatic surrogate as it enters the dendritic path
    beta_d: float = 1.5           # dendritic

    # ── Init and loss ──
    weight_scale: float = 0.25    # multiplies the Xavier std at init
    loss_temperature: float = 2.7     # logits = mean_voltage / temperature + bias
    loss_count_bias: float = 0.18
    loss_label_smoothing: float = 0.13

    v_reset: float = 0.0          # unused: soft reset subtracts v_th_eff (v - o*v_th_eff)


@jit
def surrogate_sigma(x: jnp.ndarray, beta: float) -> jnp.ndarray:
    """Surrogate derivative for the non-differentiable spike threshold.

    1 / (1 + beta*|x|)^2 — a bump peaking at 1.0 where x = 0, i.e. exactly at the
    threshold. `x` is the distance to threshold, so a neuron sitting on its
    threshold gets full gradient and one far away gets almost none. Larger beta =
    narrower bump = sparser credit assignment.
    """
    return 1.0 / (1.0 + beta * jnp.abs(x)) ** 2
