"""Single-neuron comparison of Original, BPTT_fix and BPTT_fix_auto.

Goal
----
Drive ONE two-compartment neuron (1 hidden neuron, 4 input channels, 1 readout)
with a hand-made spike pattern and compare, across the three surviving models:

  * the forward pass          -> should be BIT-IDENTICAL (same dynamics_step)
  * the per-timestep gradient  -> where the three methods actually differ

The readout error is set to e = 1 (single output, "no error" as requested), so
the learning signal is just the neuron's own gradient factor. With e = 1 the
objective the readout maximises is simply its mean membrane voltage y_hat, so
for the autodiff model we differentiate y_hat directly.

The forward equations here are copied verbatim from
``TwoCompNeuron.dynamics_step`` (identical in Original/, BPTT_fix/,
BPTT_fix_auto/). The autodiff column reuses the SAME custom_vjp spike as
``BPTT_fix_auto/network.py`` (Heaviside forward, surrogate backward), so this is
genuinely the three models' own maths, not a re-derivation.

What each column plots on the gradient row
-------------------------------------------
  Original      sigma_s(t) = s'_s(t) / T                 (flat readout kernel)
                sigma_d(t) = gamma * s'_s(t) * s'_d(t) / T
  BPTT_fix      sigma_s(t) = s'_s(t) * rho(t)            (causal readout kernel)
                sigma_d(t) = gamma * s'_s(t) * s'_d(t) * rho(t)
  BPTT_fix_auto lambda_s(t) = d y_hat / d I_soma(t)      (TRUE adjoint, jax.grad)
                lambda_d(t) = d y_hat / d I_dend(t)

Original / BPTT_fix are INSTANTANEOUS e-prop signals (temporal credit lives in
the eligibility trace, which is summed against these). BPTT_fix_auto is the full
reverse-mode adjoint, so it also carries credit backward through the alpha_s
membrane recurrence, the reset, and the plateau latch. The shape difference on
row 3 is exactly the e-prop-truncation-vs-exact-BPTT story.

Run:  conda run -n dendroprop python analysis/compare_gradient_signals.py
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # one neuron; CPU is instant

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import custom_vjp, grad, lax

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════════
#  Experiment setup (all times in ms, dt = 1 ms so index == ms)
# ══════════════════════════════════════════════════════════════════════
DT = 1.0
DURATION_MS = 800
T = DURATION_MS + 1                      # indices 0 .. 800 ms inclusive

W_DEND = np.array([0.1, 1.0, 0.9, 0.3])  # dendritic input weights (one neuron)
W_SOMA = np.array([0.7, 0.2, 0.45, 1.1])  # somatic input weights
SPIKE_MS = [10, 40, 200, 600]            # channel k spikes once, at SPIKE_MS[k]
K = len(W_DEND)

# Neuron / config constants (NeuronConfig defaults; adaptation off).
TAU_SOMA = 15.0
TAU_DEND = 15.0
TAU_M = 20.0
TAU_PLAT_MS = 400.0                      # requested plateau duration
MU_TH = 1.0
V_TH = 1.0
GAMMA = 0.5
# The somatic surrogate s'_s = do/d(soma arg) appears in BOTH pathways:
#   - somatic pathway  (grad W_soma): s'_s = do/dv_pre        -> BETA_S_SOMA
#   - dendritic pathway (grad W_dend): do/dh = gamma * s'_s    -> BETA_S_DEND
# Set them equal to recover the usual single-beta behaviour.
BETA_S_SOMA = 10.0   # somatic surrogate, somatic pathway
BETA_S_DEND = 1.0    # somatic surrogate, dendritic pathway
BETA_D = 10.0         # dendritic (plateau) surrogate s'_d, unchanged
A_ADAPT = 0.0
B_ADAPT = 0.0
TAU_W = 100.0

ALPHA_S = np.exp(-DT / TAU_SOMA)
ALPHA_D = np.exp(-DT / TAU_DEND)
ALPHA_M = np.exp(-DT / TAU_M)
ALPHA_W = np.exp(-DT / TAU_W)
T_P = int(TAU_PLAT_MS / DT)              # plateau duration in steps
SOMA_GRAD_SCALE = 8.0                    # Original/BPTT_fix empirical soma /8


def surrogate(x, beta):
    """sigma(x) = 1 / (1 + beta|x|)^2  (config.surrogate_sigma)."""
    return 1.0 / (1.0 + beta * np.abs(x)) ** 2


# Input spike-count train x[t, k]: channel k fires once at SPIKE_MS[k].
x_input = np.zeros((T, K))
for k, ms in enumerate(SPIKE_MS):
    x_input[ms, k] = 1.0

dend_in = x_input @ W_DEND               # (T,) dendritic drive
soma_in = x_input @ W_SOMA               # (T,) somatic drive


# ══════════════════════════════════════════════════════════════════════
#  Forward pass + eligibility traces (verbatim TwoCompNeuron.dynamics_step)
#  Single neuron -> scalars; eligibilities are per input channel (K,).
# ══════════════════════════════════════════════════════════════════════
mu = v = 0.0
h = 0                                     # plateau flag (h_prev at loop top)
t_prime = 0
mu_at_tp = 0.0
w_ad = 0.0
E_s = np.zeros(K)                         # somatic eligibility
dmu = np.zeros(K)                         # dendritic eligibility
Phat = np.zeros(K)                        # dmu latched at plateau onset

tr = {k: np.zeros(T) for k in
      ["mu", "h", "mu_at_tp", "v_pre", "o", "sps_soma", "sps_dend", "spd",
       "vth_eff"]}
E_s_tr = np.zeros((T, K))
Phat_tr = np.zeros((T, K))

for t in range(T):
    din, sin = dend_in[t], soma_in[t]

    # --- dendrite ---
    t_prime_new = 0 if t == 0 else (t_prime if h == 1 else t)
    mu_new = din if t == 0 else ALPHA_D * mu + (1 - h) * din
    mu_at_tp_new = mu_new if h == 0 else mu_at_tp
    plat_dur = t - t_prime_new
    h_new = 1 if (mu_at_tp_new >= MU_TH and 0 <= plat_dur <= T_P) else 0

    # --- soma ---
    v_pre = sin if t == 0 else ALPHA_S * v + sin - w_ad
    vth_eff = V_TH - GAMMA * h_new
    o = 1 if v_pre >= vth_eff else 0
    v_new = v_pre * (1 - o)
    w_new = ALPHA_W * w_ad + (1 - ALPHA_W) * A_ADAPT * v_pre + B_ADAPT * o

    # --- eligibility traces (use h_prev == h at loop top) ---
    E_s = ALPHA_S * E_s + x_input[t]
    dmu = ALPHA_D * dmu + (1 - h) * x_input[t]
    Phat = np.where(h == 0, dmu, Phat)

    # --- surrogates (distance to each threshold) ---
    # Same somatic distance, two betas: one for each pathway s'_s appears in.
    soma_arg = v_pre + GAMMA * h_new - V_TH
    sps_soma = surrogate(soma_arg, BETA_S_SOMA)            # s'_s, somatic path
    sps_dend = surrogate(soma_arg, BETA_S_DEND)            # s'_s, dendritic path
    spd = surrogate(mu_at_tp_new - MU_TH, BETA_D)          # s'_d

    tr["mu"][t] = mu_new
    tr["h"][t] = h_new
    tr["mu_at_tp"][t] = mu_at_tp_new
    tr["v_pre"][t] = v_pre
    tr["o"][t] = o
    tr["sps_soma"][t] = sps_soma
    tr["sps_dend"][t] = sps_dend
    tr["spd"][t] = spd
    tr["vth_eff"][t] = vth_eff
    E_s_tr[t] = E_s
    Phat_tr[t] = Phat

    mu, v, h, t_prime, mu_at_tp, w_ad = \
        mu_new, v_new, h_new, t_prime_new, mu_at_tp_new, w_new

sps_soma_arr = tr["sps_soma"]
sps_dend_arr = tr["sps_dend"]
spd_arr = tr["spd"]
h_arr = tr["h"].astype(bool)
spike_idx = np.where(tr["o"] > 0)[0]


# ══════════════════════════════════════════════════════════════════════
#  Gradient signals — hand-derived rules (e = 1, W_readout = 1)
# ══════════════════════════════════════════════════════════════════════
time = np.arange(T)
# BPTT_fix causal readout kernel: a hidden spike early in the window is
# integrated by the LI readout over many future steps; one at the end barely
# moves the mean. rho[t] = (1/T)(1 - alpha_m^(T-t))/(1 - alpha_m).
rho = (1.0 / T) * (1.0 - ALPHA_M ** (T - time)) / (1.0 - ALPHA_M)
flat = np.full(T, 1.0 / T)               # Original's flat readout kernel

# Soma route uses s'_s at BETA_S_SOMA; dend route's s'_s factor at BETA_S_DEND.
sig_s_O = sps_soma_arr * flat
sig_d_O = GAMMA * sps_dend_arr * spd_arr * flat
sig_s_F = sps_soma_arr * rho
sig_d_F = GAMMA * sps_dend_arr * spd_arr * rho

# Summed weight gradients (exactly as each network.py contracts them).
g_soma_O = (sps_soma_arr[:, None] * E_s_tr).sum(0) / (T * SOMA_GRAD_SCALE)
g_dend_O = ((GAMMA * sps_dend_arr * spd_arr)[:, None] * Phat_tr).sum(0) / T
g_soma_F = ((rho * sps_soma_arr)[:, None] * E_s_tr).sum(0) / SOMA_GRAD_SCALE
g_dend_F = ((rho * GAMMA * sps_dend_arr * spd_arr)[:, None] * Phat_tr).sum(0)


# ══════════════════════════════════════════════════════════════════════
#  Gradient signals — BPTT_fix_auto (true reverse-mode adjoint)
#
#  Same differentiable neuron as BPTT_fix_auto/network.py: Heaviside forward,
#  surrogate backward, via custom_vjp. We differentiate y_hat = mean readout
#  voltage w.r.t. (a) a per-timestep current perturbation -> the adjoint
#  learning signal lambda(t), and (b) the input weights -> the true weight grad.
# ══════════════════════════════════════════════════════════════════════
@custom_vjp
def spike(x, beta):
    return jnp.where(x >= 0.0, 1.0, 0.0)


def _spike_fwd(x, beta):
    return jnp.where(x >= 0.0, 1.0, 0.0), (x, beta)


def _spike_bwd(res, g):
    x, beta = res
    return (g / (1.0 + beta * jnp.abs(x)) ** 2, jnp.zeros_like(beta))


spike.defvjp(_spike_fwd, _spike_bwd)


# Somatic spike o = Theta(v_pre + gamma*h - v_th) as a TWO-input node, so the
# surrogate slope differs per input: the v_pre (somatic) cotangent uses
# BETA_S_SOMA, the h (dendritic) cotangent uses BETA_S_DEND. Autodiff then
# carries beta=BETA_S_SOMA into grad W_soma and beta=BETA_S_DEND into grad
# W_dend. Forward value is the same Heaviside as the single-beta version.
@custom_vjp
def soma_spike(v_pre, h):
    return jnp.where(v_pre + GAMMA * h - V_TH >= 0.0, 1.0, 0.0)


def _soma_spike_fwd(v_pre, h):
    x = v_pre + GAMMA * h - V_TH
    return jnp.where(x >= 0.0, 1.0, 0.0), (x,)


def _soma_spike_bwd(res, g):
    (x,) = res
    d_vpre = g / (1.0 + BETA_S_SOMA * jnp.abs(x)) ** 2           # -> W_soma
    d_h = g * GAMMA / (1.0 + BETA_S_DEND * jnp.abs(x)) ** 2      # -> W_dend (via h)
    return (d_vpre, d_h)


soma_spike.defvjp(_soma_spike_fwd, _soma_spike_bwd)

x_j = jnp.asarray(x_input)
time_j = jnp.arange(T)


def y_hat(w_soma, w_dend, soma_pert, dend_pert):
    """Mean readout voltage of the differentiable single neuron (W_readout=1)."""
    base_soma = x_j @ w_soma
    base_dend = x_j @ w_dend
    s_in = base_soma + soma_pert
    d_in = base_dend + dend_pert

    def step(carry, inp):
        mu, v, h, tp, matp, wad, rv, rsum = carry
        din, sin, t = inp
        t_prime = jnp.where(t == 0, 0, jnp.where(h == 1, tp, t))
        mu_new = jnp.where(t > 0, ALPHA_D * mu + (1 - h) * din, din)
        matp_new = jnp.where(h == 0, mu_new, matp)
        pd = t - t_prime
        timing_ok = ((pd <= T_P) & (pd >= 0)).astype(jnp.float64)
        h_new = timing_ok * spike(matp_new - MU_TH, BETA_D)
        v_pre = jnp.where(t > 0, ALPHA_S * v + sin - wad, sin)
        o = soma_spike(v_pre, h_new)   # beta_soma on v_pre, beta_dend on h
        v_new = v_pre * (1 - o)
        w_new = ALPHA_W * wad + (1 - ALPHA_W) * A_ADAPT * v_pre + B_ADAPT * o
        rv_new = ALPHA_M * rv + o                       # readout, W_r = 1
        return (mu_new, v_new, h_new, t_prime, matp_new, w_new,
                rv_new, rsum + rv_new), None

    init = (0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0)
    (*_, rsum), _ = lax.scan(step, init, (d_in, s_in, time_j))
    return rsum / T


w_soma_j = jnp.asarray(W_SOMA)
w_dend_j = jnp.asarray(W_DEND)
zeros_T = jnp.zeros(T)

# Per-timestep adjoint learning signals lambda(t) = d y_hat / d I(t).
lam_s = np.asarray(grad(y_hat, argnums=2)(w_soma_j, w_dend_j, zeros_T, zeros_T))
lam_d = np.asarray(grad(y_hat, argnums=3)(w_soma_j, w_dend_j, zeros_T, zeros_T))
# True input-weight gradients (ascent direction, matching e = +1).
g_soma_A = np.asarray(grad(y_hat, argnums=0)(w_soma_j, w_dend_j, zeros_T, zeros_T))
g_dend_A = np.asarray(grad(y_hat, argnums=1)(w_soma_j, w_dend_j, zeros_T, zeros_T))


# ══════════════════════════════════════════════════════════════════════
#  Console report — summed weight gradients (the "how similar" numbers)
# ══════════════════════════════════════════════════════════════════════
def _fmt(a):
    return "[" + ", ".join(f"{x:+.4e}" for x in a) + "]"


print("=" * 72)
print(f"Single-neuron gradient comparison  (T={T} steps, dt={DT} ms, "
      f"T_p={T_P} steps)")
print(f"plateau on {np.argmax(h_arr)}..{T - 1 - np.argmax(h_arr[::-1])} ms, "
      f"soma spikes at {list(spike_idx)} ms")
print(f"surrogate beta:  s'_s soma-path={BETA_S_SOMA:g}, "
      f"s'_s dend-path={BETA_S_DEND:g}, s'_d={BETA_D:g}")
print("=" * 72)
print("d y_hat / d w_soma  (per input channel, ascent direction, e=1)")
print(f"  Original      {_fmt(g_soma_O)}")
print(f"  BPTT_fix      {_fmt(g_soma_F)}")
print(f"  BPTT_fix_auto {_fmt(g_soma_A)}   <- true gradient")
print("d y_hat / d w_dend")
print(f"  Original      {_fmt(g_dend_O)}")
print(f"  BPTT_fix      {_fmt(g_dend_F)}")
print(f"  BPTT_fix_auto {_fmt(g_dend_A)}   <- true gradient")
print("=" * 72)


# ══════════════════════════════════════════════════════════════════════
#  Plot: 3 columns (models) x 3 rows (dendrite / soma / gradient)
# ══════════════════════════════════════════════════════════════════════
def shade_plateau(ax):
    ax.fill_between(time, 0, 1, where=h_arr, transform=ax.get_xaxis_transform(),
                    color="tab:orange", alpha=0.10, lw=0)


def mark_inputs(ax):
    for ms in SPIKE_MS:
        ax.axvline(ms, color="0.6", ls=":", lw=0.8, zorder=0)


models = ["Original", "BPTT_fix", "BPTT_fix_auto"]
grad_sig = {
    "Original": (sig_s_O, sig_d_O, "instantaneous e-prop signal (flat 1/T)"),
    "BPTT_fix": (sig_s_F, sig_d_F, "instantaneous e-prop signal (causal rho[t])"),
    "BPTT_fix_auto": (lam_s, lam_d, "true adjoint  d y_hat / d I(t)"),
}

fig, axes = plt.subplots(3, 3, figsize=(16, 9), sharex=True)

for c, name in enumerate(models):
    # Row 0 — dendrite forward (mu, plateau h). Identical across models.
    ax = axes[0, c]
    mark_inputs(ax)
    shade_plateau(ax)
    ax.plot(time, tr["mu"], color="tab:blue", label=r"$\mu(t)$")
    ax.plot(time, tr["mu_at_tp"], color="tab:cyan", ls="--", lw=1,
            label=r"$\mu_{at\,t'}$ (latched)")
    ax.axhline(MU_TH, color="0.4", ls=":", lw=1, label=r"$\mu_{th}$")
    ax.set_title(name, fontsize=13, fontweight="bold")
    if c == 0:
        ax.set_ylabel("dendrite\nforward")
    ax.legend(fontsize=7, loc="upper right")

    # Row 1 — soma forward (v_pre, dynamic threshold, spikes). Identical.
    ax = axes[1, c]
    mark_inputs(ax)
    shade_plateau(ax)
    ax.plot(time, tr["v_pre"], color="tab:green", label=r"$\tilde v(t)$ pre-reset")
    ax.plot(time, tr["vth_eff"], color="tab:red", ls="--", lw=1,
            label=r"$v_{th}-\gamma h$")
    ax.plot(spike_idx, tr["v_pre"][spike_idx], "v", color="k", ms=7,
            label="spike")
    if c == 0:
        ax.set_ylabel("soma\nforward")
    ax.legend(fontsize=7, loc="upper right")

    # Row 2 — gradient signal (error = 1). This is where they differ.
    ax = axes[2, c]
    mark_inputs(ax)
    shade_plateau(ax)
    sig_s, sig_d, sub = grad_sig[name]
    ax.plot(time, sig_s, color="tab:purple", label="soma route")
    ax.plot(time, sig_d, color="tab:brown", label="dend route")
    ax.axhline(0, color="0.7", lw=0.6)
    ax.set_title(sub, fontsize=8, style="italic")
    ax.set_xlabel("time (ms)")
    if c == 0:
        ax.set_ylabel("gradient\nsignal")
    ax.legend(fontsize=7, loc="upper right")

fig.suptitle(
    "One two-compartment neuron: identical forward (rows 1-2), "
    "different backward (row 3)   |   readout error e = 1\n"
    fr"$s'_s$ soma route $\beta$={BETA_S_SOMA:g},  "
    fr"$s'_s$ dend route $\beta$={BETA_S_DEND:g},  $\beta_d$={BETA_D:g}",
    fontsize=11,
)
fig.tight_layout(rect=(0, 0, 1, 0.95))
# Save next to this script, so it works from any working directory.
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "gradient_signal_comparison.png")
fig.savefig(out, dpi=130)
print(f"saved {out}")
