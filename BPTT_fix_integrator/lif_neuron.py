import jax.numpy as jnp
from jax import random

from config import NeuronConfig


class LINeuron:
    """Pure-integrator readout — no leak, no threshold, no reset.

    The output logit is the FINAL membrane potential U[T] = Σ_k W·o[k]. With no
    leak (alpha_m = 1) every spike lands in U[T] with the same weight W, so the
    credit a spike gets is constant in time: ∂U[T]/∂o[t] = W (rho[t] ≡ 1). The
    eligibility E[t] = Σ_{k≤t} o[k] is a plain cumulative presynaptic spike
    count; its final value E[T] is exactly ∂U[T]/∂W.
    """

    def __init__(self, key: jnp.ndarray, n_neurons: int, n_inputs: int, config: NeuronConfig):
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs
        self.config = config
        self.alpha_m = 1.0  # pure integrator (no leak)

        xavier_std = jnp.sqrt(2.0 / n_inputs)
        self.w = random.normal(key, (n_neurons, n_inputs)) * xavier_std * config.weight_scale

    def init_carry(self):
        return (
            jnp.zeros(self.n_neurons),  # v  (running integrator; final value = logit)
            jnp.zeros(self.n_inputs),   # E  (cumulative presynaptic spike count)
        )

    @staticmethod
    def forward_step(carry, spike_input, w, alpha_m):
        """Pure-function forward step for one timestep. JIT-friendly.

        alpha_m is kept in the signature but is 1.0 for this integrator readout.
        """
        v_prev, E_prev = carry

        v = alpha_m * v_prev + spike_input @ w.T  # no leak, no threshold, no reset
        E = alpha_m * E_prev + spike_input

        new_carry = (v, E)
        return new_carry, v, E
