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

    def init_carry(self, batch=None):
        """Zero carry for the training scan. batch=None → one sample, int → batched.

        The single definition of the 9-slot layout; network.py builds its carries
        through here rather than repeating the shapes. Slots 5-7 are the
        eligibility traces, which only the training path uses — see forward_step.
        """
        n = self.n_neurons
        k = self.n_inputs
        s = (batch, n) if batch else (n,)
        sk = (batch, k) if batch else (k,)
        snk = (batch, n, k) if batch else (n, k)
        return (
            jnp.zeros(s),                  # mu
            jnp.zeros(s),                  # v
            jnp.zeros(s, dtype=jnp.int32), # h
            jnp.zeros(s, dtype=jnp.int32), # t_prime
            jnp.zeros(s),                  # mu_at_tprime
            jnp.zeros(sk),                 # E_soma
            jnp.zeros(snk),                # dmu_dw
            jnp.zeros(snk),                # dmu_dw_at_tprime
            jnp.zeros(s),                  # w (adaptation variable)
        )

    @staticmethod
    def dynamics_step(state, dend_in, soma_in, t, alpha_s, alpha_d, T_p, config, alpha_w):
        """The two-compartment dynamics for one timestep. JIT-friendly.

        THE single definition of this neuron's behaviour — training, inference and
        the activity diagnostic all route through here, so they cannot drift apart.

        state: (mu, v, h, t_prime, mu_at_tprime, w) — the 6 slots the equations
        below actually evolve. Training carries three eligibility traces as well,
        but no equation here reads them, so they are not this function's business;
        forward_step threads them through for the training path.

        Returns (new_state, o, v_pre_reset, h, h_prev, mu_at_tprime), where the
        trailing values are the observables the caller needs for surrogate
        gradients: the spike, the pre-reset voltage, the new and previous plateau
        flags, and the dendritic potential latched at plateau onset.
        """
        mu_prev, v_prev, h_prev, t_prime_prev, mu_at_tprime_prev, w_prev = state

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
        v_th_eff = config.v_th - config.gamma * h
        o = jnp.where(v_pre_reset >= v_th_eff, 1, 0).astype(jnp.int32)
        # Soft (subtractive) reset: on a spike, subtract the *effective* (plateau-
        # lowered) threshold that was crossed, rather than zeroing v. This conserves
        # supra-threshold charge and makes ∂v_pre_reset/∂v_prev = alpha_s (clean), so
        # the ungated somatic eligibility E_soma is exactly the correct e-prop trace.
        v = v_pre_reset - o * v_th_eff
        w = alpha_w * w_prev + (1 - alpha_w) * config.a_adapt * v_pre_reset + config.b_adapt * o

        new_state = (mu, v, h, t_prime, mu_at_tprime, w)
        return new_state, o, v_pre_reset, h, h_prev, mu_at_tprime

    @staticmethod
    def forward_step(carry, dend_in, soma_in, t, alpha_s, alpha_d, T_p, config, alpha_w):
        """Training-path forward step: dynamics_step over the full 9-slot carry.

        Identical equations — the only thing this adds is passing the three
        eligibility slots through untouched. They are updated by the caller
        (see _forward_and_accum in network.py), not here.
        """
        mu, v, h, t_prime, mu_at_tprime, E_soma, dmu_dw, dmu_dw_at_tprime, w = carry

        state, o, v_pre_reset, h_new, h_prev, mu_at_tprime_new = TwoCompNeuron.dynamics_step(
            (mu, v, h, t_prime, mu_at_tprime, w),
            dend_in, soma_in, t, alpha_s, alpha_d, T_p, config, alpha_w,
        )
        mu, v, h, t_prime, mu_at_tprime, w = state

        new_carry = (mu, v, h, t_prime, mu_at_tprime, E_soma, dmu_dw, dmu_dw_at_tprime, w)
        return new_carry, o, v_pre_reset, h_new, h_prev, mu_at_tprime_new

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
