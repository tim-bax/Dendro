"""Configuration structs.

``NeuronConfig`` holds the fixed neuron parameters (decays, thresholds, plateau
window, surrogate sharpness) plus ``soma_grad_scale`` — the divisor applied to the
somatic gradient (``grad_s`` is divided by ``T · soma_grad_scale``; default 8.0,
without which somatic grads come out ~8× too large). Every field has a default, so a
plain dict round-trips through ``NeuronConfig(**cfg)``.

``RunConfig`` is the optimization configuration — optimizer choice, learning rate,
gradient clip, weight decay, Adam betas/eps, and dropout rate. It is separate from
``NeuronConfig`` (fixed neuron parameters): ``lr``/``clip_value`` may be driven per
step by the train loop, the rest are stored. The optimizer primitives that consume it
live in ``optim.py``.

The surrogate function is a backward-pass concern and lives in ``surrogate.py``, not
here. See docs/architecture.md.
"""
from flax import struct


@struct.dataclass
class NeuronConfig:
    """All these values are overwritten by command line flags, so they are not hyperparameters used by default."""

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

    # ── Surrogate gradient sharpness (see surrogate.py) ──
    beta_s: float = 1.0           # somatic
    beta_d: float = 1.5           # dendritic

    # ── Init and loss ──
    weight_scale: float = 0.25    # multiplies the Xavier std at init
    loss_temperature: float = 2.7     # logits = mean_voltage / temperature + bias
    loss_count_bias: float = 0.18
    loss_label_smoothing: float = 0.13

    # ── Gradient balancing (empirical) ──
    # Extra divisor on the somatic gradient only, balancing it against the dendritic
    # gradient (the somatic accumulator runs ~8x larger). Tuned, not derived; 8.0 is
    # the working value. Applied during grad assembly, not the forward pass.
    soma_grad_scale: float = 8.0

    v_reset: float = 0.0          # unused: reset is implemented as v * (1 - o)


@struct.dataclass
class RunConfig:
    """Run/optimization configuration — separate from ``NeuronConfig`` (the fixed
    neuron parameters).

    ``optimizer`` selects the ``optim.py`` primitive ("adam" | "sgd") and is a static
    (jit cache-key) field, never a traced leaf. ``lr``/``clip_value`` are per-step
    (the train loop may drive ``lr`` from a schedule); ``weight_decay``/``beta1``/
    ``beta2``/``adam_eps`` are the stored AdamW hyperparameters; ``dropout_rate`` is the
    hidden→next-edge dropout used by the training forward pass.
    """

    optimizer: str = struct.field(pytree_node=False, default="adam")
    lr: float = 1e-3
    clip_value: float = 1.0
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    adam_eps: float = 1e-8
    dropout_rate: float = 0.0
