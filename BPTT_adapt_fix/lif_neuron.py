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

    def init_carry(self):
        return (
            jnp.zeros(self.n_neurons),  # v
            jnp.zeros(self.n_neurons),  # sum_v (accumulated for mean)
            jnp.zeros(self.n_inputs),   # E_readout
        )

    @staticmethod
    def forward_step(carry, spike_input, w, alpha_m):
        """Pure-function forward step for one timestep. JIT-friendly."""
        v_prev, sum_v_prev, E_prev = carry

        v = alpha_m * v_prev + spike_input @ w.T  # no threshold, no reset
        sum_v = sum_v_prev + v
        E = alpha_m * E_prev + spike_input

        new_carry = (v, sum_v, E)
        return new_carry, v, E
