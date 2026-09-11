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
    # NOTE (autograd variant): these are all HONORED, wired to reproduce
    # Soft_Reset's e-prop gradient via the two-path `somatic_spike` custom_vjp in
    # network.py — the soma edge uses beta_s, the dendritic edge uses beta_s_dend
    # (+ the roof when dend_surrogate_roof is set), and the plateau uses beta_d.
    beta_s: float = 1.0           # somatic surrogate, somatic gradient path
    beta_s_dend: float = 1.0      # somatic surrogate on the DENDRITIC gradient edge
    beta_d: float = 1.5           # dendritic

    # HONORED in the autograd variant: switches the dendritic gradient edge (∂o/∂h)
    # to the flat "roof" over [v_th-gamma, v_th] (surrogate_roof, gamma folded into
    # the width) instead of the single bump. Same effect as in Soft_Reset.
    dend_surrogate_roof: bool = struct.field(pytree_node=False, default=False)

    # ── Init and loss ──
    weight_scale: float = 0.25    # multiplies the Xavier std at init
    loss_temperature: float = 2.7     # logits = mean_voltage / temperature + bias
    loss_count_bias: float = 0.18
    loss_label_smoothing: float = 0.13

    # ── Firing-rate regularization (disabled by default) ──
    # Hinge-L2 penalty L_reg = (strength/2) Σ_n max(0, r_n - target)² on the
    # per-neuron mean spike rate r_n. Only ABOVE-target firing is penalized, so
    # quiet neurons are untouched. 0.0 = off.
    rate_reg_strength: float = 0.0
    rate_target: float = 0.05     # target mean rate (spikes/step); ~0.05 ≈ 12.5 Hz at dt=4ms

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


@jit
def surrogate_roof(v: jnp.ndarray, lo: float, hi: float, beta: float) -> jnp.ndarray:
    """Flat-top ("roof") surrogate for the DENDRITIC gradient path.

    Exactly 1 across the band [lo, hi] (the plateau's reachable range,
    lo = v_th - gamma .. hi = v_th) and decaying SYMMETRICALLY outside via the same
    fast-sigmoid kernel as surrogate_sigma, applied to the distance to the nearest
    band edge. This is the smoothed secant of the binary somatic spike o(h) across
    the plateau gate's jump h: 0 -> 1: the exact finite difference is a boxcar of
    height 1 over [lo, hi]; this rounds only its OUTSIDE shoulders, leaving the top
    flat. So a neuron whose voltage sits anywhere in the band a plateau could flip
    gets full credit, not just one at the current effective threshold.

    `beta` sets the out-of-band decay rate (its original surrogate_sigma meaning);
    the top is ALWAYS flat at 1, independent of beta. As lo -> hi it reduces exactly
    to surrogate_sigma(v - v_th, beta), recovering the single-bump surrogate.

    Only ever used as a multiplicative gradient factor (never differentiated), so
    the C0 kink where the flat top meets the decay at lo/hi is harmless.
    """
    d = jnp.maximum(jnp.maximum(lo - v, v - hi), 0.0)  # 0 in band; dist to edge outside
    return surrogate_sigma(d, beta)
