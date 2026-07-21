import jax.numpy as jnp
from jax import random

from config import NeuronConfig


class TwoCompNeuron:
    def __init__(self, key: jnp.ndarray, n_neurons: int, n_inputs: int, config: NeuronConfig):
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs
        self.config = config
        self.alpha_s = jnp.exp(-config.dt / config.tau_soma)
        self.alpha_w = jnp.exp(-config.dt / config.tau_w)

        key1, key2, key3 = random.split(key, 3)

        self.alpha_d = jnp.exp(-config.dt / config.tau_dend)

        tau_plat_values = random.uniform(
            key3, shape=(n_neurons,), minval=config.tau_plat_min, maxval=config.tau_plat_max
        )
        self.T_p = (tau_plat_values / config.dt).astype(jnp.int32)

        xavier_std = jnp.sqrt(2.0 / n_inputs)
        scale = xavier_std * config.weight_scale
        self.w_dend = random.normal(key1, (n_neurons, n_inputs)) * scale
        self.w_soma = random.normal(key2, (n_neurons, n_inputs)) * scale

    def init_carry(self):
        n = self.n_neurons
        k = self.n_inputs
        return (
            jnp.zeros(n),                  # mu
            jnp.zeros(n),                  # v
            jnp.zeros(n, dtype=jnp.int32), # h
            jnp.zeros(n, dtype=jnp.int32), # t_prime
            jnp.zeros(n),                  # mu_at_tprime
            jnp.zeros(k),                  # E_soma
            jnp.zeros((n, k)),             # dmu_dw
            jnp.zeros((n, k)),             # dmu_dw_at_tprime
            jnp.zeros(n),                  # w (adaptation variable)
        )

    @staticmethod
    def forward_step(carry, dend_in, soma_in, t, alpha_s, alpha_d, T_p, config, alpha_w):
        """Pure-function forward step for one timestep. JIT-friendly."""
        mu_prev, v_prev, h_prev, t_prime_prev, mu_at_tprime_prev, E_soma, dmu_dw, dmu_dw_at_tprime, w_prev = carry

        t_prime = jnp.where(t == 0, 0, jnp.where(h_prev == 1, t_prime_prev, t))
        mu = jnp.where(t > 0, alpha_d * mu_prev + (1 - h_prev) * dend_in, dend_in)
        mu_at_tprime = jnp.where(h_prev == 0, mu, mu_at_tprime_prev)

        plateau_duration = t - t_prime
        h = jnp.where(
            (mu_at_tprime >= config.mu_th)
            & (plateau_duration <= T_p)
            & (plateau_duration >= 0),
            1, 0,
        ).astype(jnp.int32)

        v_pre_reset = jnp.where(t > 0, alpha_s * v_prev + soma_in - w_prev, soma_in)
        o = jnp.where(v_pre_reset >= config.v_th - config.gamma * h, 1, 0).astype(jnp.int32)
        v = v_pre_reset * (1 - o)
        w = alpha_w * w_prev + (1 - alpha_w) * config.a_adapt * v_pre_reset + config.b_adapt * o

        new_carry = (mu, v, h, t_prime, mu_at_tprime, E_soma, dmu_dw, dmu_dw_at_tprime, w)
        return new_carry, o, v_pre_reset, h, h_prev, mu_at_tprime

    @staticmethod
    def update_somatic_eligibility(E_soma_prev, pre_spike_t, alpha_s):
        return alpha_s * E_soma_prev + pre_spike_t

    @staticmethod
    def update_dendritic_eligibility(dmu_dw_prev, dmu_dw_at_tprime_prev, pre_spike_t, h_prev, alpha_d):
        # alpha_d is a scalar; reshape to (1,1) broadcasts over the (n,k) trace.
        dmu_dw = jnp.reshape(alpha_d, (-1, 1)) * dmu_dw_prev + (1 - h_prev[:, None]) * pre_spike_t[None, :]
        dmu_dw_at_tprime = jnp.where(
            (h_prev == 0)[:, None],
            dmu_dw,
            dmu_dw_at_tprime_prev,
        )
        return dmu_dw, dmu_dw_at_tprime
