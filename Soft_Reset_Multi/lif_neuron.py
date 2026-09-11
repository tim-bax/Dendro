import jax.numpy as jnp
from jax import random

from config import NeuronConfig


class LINeuron:
    """Leaky integrator readout — no threshold, no reset."""

    def __init__(self, key: jnp.ndarray, n_neurons: int, n_inputs: int, config: NeuronConfig):
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs
        self.config = config
        self.alpha_m = jnp.exp(-config.dt / config.tau_m)

        xavier_std = jnp.sqrt(2.0 / n_inputs)
        self.w = random.normal(key, (n_neurons, n_inputs)) * xavier_std * config.weight_scale

    def init_carry(self, batch=None):
        """Zero carry. batch=None → one sample, int → batched.

        The single definition of the 3-slot layout; network.py builds its carries
        through here. E_readout is only used by the training path (forward_step).
        """
        j = self.n_neurons
        n = self.n_inputs
        sj = (batch, j) if batch else (j,)
        sn = (batch, n) if batch else (n,)
        return (
            jnp.zeros(sj),  # v
            jnp.zeros(sj),  # sum_v (accumulated for mean)
            jnp.zeros(sn),  # E_readout
        )

    @staticmethod
    def integrate(v_prev, sum_v_prev, spike_input, w, alpha_m):
        """The readout dynamics for one timestep. JIT-friendly.

        THE single definition — training and inference both route through here.
        Leaky integration plus the running voltage sum; no threshold, no reset,
        so the neuron never spikes and there is nothing to count. The eligibility
        trace E is not part of the dynamics and is not needed for inference, so
        forward_step adds it for the training path.
        """
        v = alpha_m * v_prev + spike_input @ w.T
        return v, sum_v_prev + v

    @staticmethod
    def forward_step(carry, spike_input, w, alpha_m):
        """Training-path forward step: integrate() plus the eligibility trace."""
        v_prev, sum_v_prev, E_prev = carry

        v, sum_v = LINeuron.integrate(v_prev, sum_v_prev, spike_input, w, alpha_m)
        E = alpha_m * E_prev + spike_input

        new_carry = (v, sum_v, E)
        return new_carry, v, E
