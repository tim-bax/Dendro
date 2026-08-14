import jax.numpy as jnp
from jax import random

from config import NeuronConfig, a_for_f


class TwoCompNeuron:
    def __init__(self, key: jnp.ndarray, n_neurons: int, n_inputs: int, config: NeuronConfig):
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs
        self.config = config
        self.alpha_s = jnp.exp(-config.dt / config.tau_soma)
        self.alpha_w = jnp.exp(-config.dt / config.tau_w)

        key1, key2, key3, key4, key5 = random.split(key, 5)

        self.alpha_d = jnp.exp(-config.dt / config.tau_dend)

        tau_plat_values = random.uniform(
            key3, shape=(n_neurons,), minval=config.tau_plat_min, maxval=config.tau_plat_max
        )
        self.T_p = (tau_plat_values / config.dt).astype(jnp.int32)

        xavier_std = jnp.sqrt(2.0 / n_inputs)
        scale = xavier_std * config.weight_scale
        self.w_dend = random.normal(key1, (n_neurons, n_inputs)) * scale
        self.w_soma = random.normal(key2, (n_neurons, n_inputs)) * scale

        # Clip ceiling / init band: the a value that puts the resonance at f_max_hz.
        # α_s/α_w/α_d are global scalars, so a_adapt is the ONLY source of per-neuron
        # somatic frequency diversity.
        self.a_max = a_for_f(self.alpha_s, self.alpha_w, config.f_max_hz, config.dt)

        # Per-neuron adaptation parameters (trainable).
        if config.adapt_init == "warm":
            # Homogeneous warm-start at the fixed config values — training starts
            # identical to the fixed-value baseline (opt-in for comparison runs).
            self.a_adapt = jnp.full((n_neurons,), config.a_adapt)
            self.b_adapt = jnp.full((n_neurons,), config.b_adapt)
        else:
            # Heterogeneous: spread a across the resonant band [0, a_max] so the
            # population covers a range of tuned frequencies; b over [0, 2].
            self.a_adapt = random.uniform(key4, (n_neurons,), minval=0.0, maxval=self.a_max)
            self.b_adapt = random.uniform(key5, (n_neurons,), minval=0.0, maxval=2.0)

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
            jnp.zeros(n),                  # elig_a (d w / d a_adapt)
            jnp.zeros(n),                  # elig_b (d w / d b_adapt)
        )

    @staticmethod
    def forward_step(carry, dend_in, soma_in, t, alpha_s, alpha_d, T_p, config, alpha_w,
                     a_adapt, b_adapt):
        """Pure-function forward step for one timestep. JIT-friendly.

        a_adapt/b_adapt are per-neuron (n,) adaptation parameters. The two
        eligibility slots (elig_a, elig_b) are carried through unchanged here; they
        are advanced by _layer_forward, which owns the eligibility bookkeeping.
        """
        (mu_prev, v_prev, h_prev, t_prime_prev, mu_at_tprime_prev,
         E_soma, dmu_dw, dmu_dw_at_tprime, w_prev, elig_a, elig_b) = carry

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
        # Adaptation couples to the POST-reset voltage v (which is 0 on spike steps)
        # plus a spike-triggered jump b·o. Using v instead of v_pre_reset means the
        # subthreshold a-term and the spike b-term never contribute on the same step.
        w = alpha_w * w_prev + (1 - alpha_w) * a_adapt * v + b_adapt * o

        new_carry = (mu, v, h, t_prime, mu_at_tprime, E_soma, dmu_dw, dmu_dw_at_tprime,
                     w, elig_a, elig_b)
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
