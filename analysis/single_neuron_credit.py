#!/usr/bin/env python3
"""Per-timestep credit assignment for ONE two-compartment neuron, hand rule vs
exact autodiff.

Setup: 3 spiking input channels (2 spikes each) feed BOTH compartments of a
single two-comp neuron; that neuron feeds a 2-class leaky-integrator readout.
One dendritic weight is 1.1, so a single spike on channel 0 latches a plateau.

Top of the figure: the shared forward pass (identical for both rules) — input
spikes, dendrite mu + plateau, soma v + spikes.

Bottom: for EACH of the six weights (W_d[0..2], W_s[0..2]), the per-timestep
contribution to that weight's gradient, hand rule vs exact autodiff, on one axis.
Each curve sums (over t) to that rule's total gradient for the weight.

  hand      contribution[t] = Lv[t]·E_s[t]            (soma)
                            = (Lv[t]·γ·s'_d[t])·P̂[t]  (dend)   — forward eligibility
  autodiff  contribution[t] = (∂L/∂drive[t])·x[t]                — backward adjoint

The hand rule is shown on the TRUE-gradient scale (its 1/temperature and soma /8
rescalings undone) so magnitudes are directly comparable; only the temporal
structure and the #2 explosion then separate the two.
"""
import os
import sys
import argparse

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
sys.path.insert(0, _THIS)

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import compare_backward as cb  # reuse module loader + auto/hand grad helpers


# ── fixed experiment definition (matches the request) ───────────────────
W_DEND = [1.1, 0.4, 0.6]      # channel 0 (1.1) latches a plateau on a single spike
W_SOMA = [0.55, 0.70, 0.60]   # somatic drive — enough for a few spikes once the plateau lowers v_th
W_READ = [[1.0], [0.2]]       # 2-class readout from the single hidden neuron
# 2 spikes per channel. ch0 latches the plateau at t=5; ch1 also spikes BEFORE the
# latch (t=3 → gets dendritic credit) and after; ch2 spikes only AFTER the latch
# (→ its dendritic input is gated out, dead in both rules).
SPIKES = {0: [5, 15], 1: [3, 18], 2: [9, 24]}
TARGET = 0


def build_input(T):
    x = np.zeros((T, 3), dtype=np.float64)
    for ch, times in SPIKES.items():
        for t in times:
            if t < T:
                x[t, ch] = 1.0
    return jnp.asarray(x)


def set_weights(net):
    net.hidden[0].w_dend = jnp.asarray([W_DEND])
    net.hidden[0].w_soma = jnp.asarray([W_SOMA])
    net.readout.w = jnp.asarray(W_READ)


# ── shared forward + all quantities the hand rule contracts ─────────────

def hand_forward(mod, x, params, temp, bias, target_smoothed):
    """One scan over the single neuron; returns forward traces AND the per-t
    hand-rule ingredients (s'_s, s'_d, E_soma, P̂), plus the readout error."""
    TwoComp, LI = mod.TwoCompNeuron, mod.LINeuron
    alpha_s, alpha_d, alpha_m, tp_list, config, alpha_w = params
    tp0 = tp_list[0]
    w_d = jnp.asarray([W_DEND]); w_s = jnp.asarray([W_SOMA]); w_r = jnp.asarray(W_READ)
    T = x.shape[0]
    tidx = jnp.arange(T, dtype=jnp.int32)
    dend_in = x @ w_d.T
    soma_in = x @ w_s.T

    h0 = cb._zero_h(1, 3)
    r0 = (jnp.zeros(2), jnp.zeros(2), jnp.zeros(1))

    def step(carry, xs):
        hc, rc = carry
        din, sin, xt, t = xs
        new_hc, o, v_pre, h, h_prev, matp = TwoComp.forward_step(
            hc, din, sin, t, alpha_s, alpha_d, tp0, config, alpha_w)
        E_soma = TwoComp.update_somatic_eligibility(hc[5], xt, alpha_s)
        dmu, dmu_atp = TwoComp.update_dendritic_eligibility(hc[6], hc[7], xt, h_prev, alpha_d)
        new_hc = (new_hc[0], new_hc[1], new_hc[2], new_hc[3], new_hc[4],
                  E_soma, dmu, dmu_atp, new_hc[8])
        of = o.astype(jnp.float64)
        rc, v_r, E_r = LI.forward_step(rc, of, w_r, alpha_m)
        return (new_hc, rc), (new_hc[0], v_pre, h.astype(jnp.float64), of, v_r,
                              new_hc[4], E_soma, dmu_atp)

    (_, rc_f), (mu, v_pre, h, o, v_r, matp, E_soma, P_hat) = lax.scan(
        step, (h0, r0), (dend_in, soma_in, x, tidx))

    cfg = params[4]
    mean_v = jnp.sum(v_r, axis=0) / T
    logits = mean_v / temp + bias
    p = jnp.exp(logits - jnp.max(logits)); p = p / jnp.sum(p)
    e = target_smoothed - p                                  # ascent error (J,)
    readout_seed = (w_r.T @ e)                               # (hidden=1,)
    rho = (1.0 / T) * (1.0 - alpha_m ** (T - tidx)) / (1.0 - alpha_m)  # (T,)

    sp = 1.0 / (1.0 + cfg.beta_s * jnp.abs(v_pre[:, 0] + cfg.gamma * h[:, 0] - cfg.v_th)) ** 2  # s'_s (T,)
    hp = 1.0 / (1.0 + cfg.beta_d * jnp.abs(matp[:, 0] - cfg.mu_th)) ** 2                        # s'_d (T,)

    Lv = (rho * readout_seed[0]) * sp                        # (T,) — single neuron
    P_hat0 = P_hat[:, 0, :]                                  # (T,3), frozen dendritic elig at t'
    # per-timestep hand contributions (raw summands, single neuron -> (T,3))
    hand_soma = Lv[:, None] * E_soma                         # Lv[t]·E_s[t]
    hand_dend = (Lv * cfg.gamma * hp)[:, None] * P_hat0      # (Lv·γ·s'_d)·P̂

    out = dict(mu=mu[:, 0], v_pre=v_pre[:, 0], h=h[:, 0], o=o[:, 0],
               v_r=np.asarray(v_r), mean_v=np.asarray(mean_v), probs=np.asarray(p),
               hand_soma=np.asarray(hand_soma), hand_dend=np.asarray(hand_dend),
               # individual chain-rule factors (for the dissection plots)
               delta_o=rho * readout_seed[0],   # ∂L/∂o  (raw hand scale; /τ for true scale)
               sps=sp,                           # ∂o/∂v_pre = s'_s
               spd=hp,                           # ∂h/∂μ_t' = s'_d
               E_soma=E_soma,                    # ∂v_pre/∂w_s = somatic eligibility
               P_hat=P_hat0,                     # ∂μ_t'/∂w_d = frozen dendritic eligibility
               gamma=cfg.gamma)
    return {k: (np.asarray(v) if not isinstance(v, np.ndarray) else v) for k, v in out.items()}


# ── exact autodiff per-timestep contributions ──────────────────────────

def auto_pertimestep(mod_auto, x, params, temp, bias, target_smoothed):
    """Differentiate the loss w.r.t. the per-step synaptic drive; multiply by the
    input to get the per-t contribution to each weight (sums to the exact grad)."""
    _tc_step, _init = mod_auto._tc_step, mod_auto._init_tc_state
    alpha_s, alpha_d, alpha_m, tp_list, config, alpha_w = params
    tp0 = tp_list[0]
    w_r = jnp.asarray(W_READ)
    T = x.shape[0]
    tidx = jnp.arange(T, dtype=jnp.int32)
    dend_in = x @ jnp.asarray([W_DEND]).T   # (T,1)
    soma_in = x @ jnp.asarray([W_SOMA]).T   # (T,1)

    def loss(din_seq, sin_seq):
        def step(state, xs):
            din, sin, t = xs
            ns, o = _tc_step(state, din, sin, t, alpha_s, alpha_d, tp0, config, alpha_w)
            return ns, o
        _, os = lax.scan(step, _init(1), (din_seq, sin_seq, tidx))   # (T,1)

        def rstep(carry, o):
            v_r, s = carry
            v_r = alpha_m * v_r + (o @ w_r.T)
            return (v_r, s + v_r), None
        (_, sumv), _ = lax.scan(rstep, (jnp.zeros(2), jnp.zeros(2)), os)
        mean_v = sumv / T
        logits = mean_v / temp + bias
        p = jnp.exp(logits - jnp.max(logits)); p = p / jnp.sum(p)
        return -jnp.sum(target_smoothed * jnp.log(p + 1e-8))

    g_din, g_sin = jax.grad(loss, argnums=(0, 1))(dend_in, soma_in)
    g_din, g_sin, x_np = np.asarray(g_din)[:, 0], np.asarray(g_sin)[:, 0], np.asarray(x)
    auto_dend = -(g_din[:, None] * x_np)    # (T,3) ascent
    auto_soma = -(g_sin[:, None] * x_np)
    return auto_soma, auto_dend


# ── plotting ────────────────────────────────────────────────────────────

def plot(x, fwd, hand_soma, hand_dend, auto_soma, auto_dend, cfg, out_path):
    T = x.shape[0]
    t = np.arange(T)
    plateau = fwd["h"] > 0.5
    v_th, gamma, mu_th = cfg["v_th"], cfg["gamma"], cfg["mu_th"]
    ch_col = ["#1b7837", "#762a83", "#b35806"]

    fig = plt.figure(figsize=(15, 13.5))
    gs = fig.add_gridspec(5, 3, height_ratios=[0.85, 1.0, 1.0, 1.15, 1.15],
                          hspace=0.55, wspace=0.28)

    def shade(ax):
        ax.fill_between(t, 0, 1, where=plateau, transform=ax.get_xaxis_transform(),
                        color="orange", alpha=0.10, step="mid", zorder=0)

    # --- forward: input spikes ---
    ax0 = fig.add_subplot(gs[0, :])
    for ch, times in SPIKES.items():
        for tt in times:
            if tt < T:
                ax0.vlines(tt, ch - 0.32, ch + 0.32, color=ch_col[ch], lw=3)
    shade(ax0)
    ax0.set_yticks([0, 1, 2]); ax0.set_yticklabels(
        [f"ch0 (w_d={W_DEND[0]})", f"ch1 (w_d={W_DEND[1]})", f"ch2 (w_d={W_DEND[2]})"])
    ax0.set_ylim(-0.6, 2.6); ax0.set_xlim(-0.5, T - 0.5)
    ax0.set_title("Forward pass (shared by both rules) — input spikes into both compartments", fontsize=12)

    # --- forward: dendrite ---
    ax1 = fig.add_subplot(gs[1, :])
    ax1.plot(t, fwd["mu"], color="#2ca02c", lw=1.6, label="μ (dendrite)")
    ax1.axhline(mu_th, color="#9467bd", ls=":", lw=1.0, label="μ_th")
    shade(ax1)
    ax1.set_ylabel("dendrite μ"); ax1.set_xlim(-0.5, T - 0.5)
    ax1.legend(fontsize=8, loc="upper right", framealpha=0.7)
    ax1.set_title("dendrite: channel-0 spike (w_d=1.1) crosses μ_th → plateau latches (shaded)", fontsize=10)

    # --- forward: soma ---
    ax2 = fig.add_subplot(gs[2, :])
    ax2.plot(t, fwd["v_pre"], color="#1f77b4", lw=1.6, label="v (soma)")
    ax2.plot(t, v_th - gamma * fwd["h"], color="#d62728", ls="--", lw=1.0, label="threshold v_th−γh")
    sp_t = t[fwd["o"] > 0.5]
    ytop = max(np.max(fwd["v_pre"]) * 1.05, v_th * 1.15)
    ax2.plot(sp_t, np.full_like(sp_t, ytop, dtype=float), "|", color="k", ms=12, mew=1.4, label="spike o")
    shade(ax2)
    ax2.set_ylabel("soma v"); ax2.set_xlabel("timestep"); ax2.set_xlim(-0.5, T - 0.5)
    ax2.legend(fontsize=8, loc="upper right", framealpha=0.7)
    ax2.set_title(f"soma: plateau lowers threshold by γ={gamma} → spikes.  "
                  f"mean_v={np.round(fwd['mean_v'],3)}  probs={np.round(fwd['probs'],3)}", fontsize=10)

    # --- backward: per-weight per-timestep contributions ---
    def lin(vals):
        m = np.max(np.abs(vals))
        return max(m * 0.02, 1e-4)

    all_vals = np.concatenate([auto_dend.ravel(), hand_dend.ravel(),
                               auto_soma.ravel(), hand_soma.ravel()])
    linthresh = lin(all_vals)
    ymax = float(np.max(np.abs(all_vals)))

    def bwd_axis(ax, auto_j, hand_j, title, ch):
        shade(ax)
        ax.axhline(0, color="k", lw=0.5)
        for tt in SPIKES[ch]:
            if tt < T:
                ax.axvline(tt, color=ch_col[ch], ls=":", lw=1.0, alpha=0.7)
        ax.plot(t, auto_j, color="#1f77b4", lw=1.5, marker="o", ms=3, label="autodiff (exact)")
        ax.plot(t, hand_j, color="#ff7f0e", lw=1.5, marker="s", ms=3, label="hand rule (norm)")
        ax.set_yscale("symlog", linthresh=linthresh)
        ax.set_ylim(-ymax * 1.6, ymax * 1.6)      # shared scale across all six panels
        ax.set_xlim(-0.5, T - 0.5)
        sa, sh = auto_j.sum(), hand_j.sum()
        ax.set_title(f"{title}\nΣ auto={sa:+.3g}   Σ hand={sh:+.3g}", fontsize=9)
        if max(np.abs(auto_j).max(), np.abs(hand_j).max()) < 1e-10:
            ax.text(0.5, 0.5, "both ≈ 0\ndendrite gated\nafter latch", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8.5, color="#555",
                    bbox=dict(boxstyle="round", fc="white", ec="#bbb", alpha=0.85))

    for j in range(3):
        axd = fig.add_subplot(gs[3, j])
        bwd_axis(axd, auto_dend[:, j], hand_dend[:, j], f"dL/dW_dend[0,{j}]  (channel {j})", j)
        if j == 0:
            axd.set_ylabel("per-t grad\n(symlog)"); axd.legend(fontsize=7.5, loc="upper left", framealpha=0.7)
        axs = fig.add_subplot(gs[4, j])
        bwd_axis(axs, auto_soma[:, j], hand_soma[:, j], f"dL/dW_soma[0,{j}]  (channel {j})", j)
        axs.set_xlabel("timestep")
        if j == 0:
            axs.set_ylabel("per-t grad\n(symlog)")

    fig.suptitle("Per-timestep credit assignment — one 2-compartment neuron, 3 synapses\n"
                 "autodiff (backward adjoint): credit sits at the event that used the synapse   ·   "
                 "hand rule (forward eligibility): credit spread over the trace", fontsize=12.5)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_chain(kind, fwd, temp, out_path):
    """Dissect the hand-rule weight gradient into its chain-rule factors, per weight.

    Columns = the 3 input channels (= the 3 weights of that compartment); rows = the
    chain-rule factors stacked, with the final product at the bottom. For a single
    neuron every factor except the eligibility is SHARED across synapses (it's a
    property of the neuron's output), so those rows are identical across columns.
    The bottom (product) row equals that weight's orange curve in
    single_neuron_credit.png.
    """
    T = fwd["h"].shape[0]
    t = np.arange(T)
    plateau = fwd["h"] > 0.5
    dLdo = fwd["delta_o"] / temp          # ∂L/∂o on the true-gradient scale (1/τ folded in)
    sps, spd = fwd["sps"], fwd["spd"]
    gamma = float(fwd["gamma"])
    E, P = fwd["E_soma"], fwd["P_hat"]
    ch_col = ["#1b7837", "#762a83", "#b35806"]

    if kind == "soma":
        rows = [
            ("∂L/∂o\n(ρ·Wᵣᵀe / τ)", lambda j: dLdo, True),
            ("∂o/∂v_pre\n= s'ₛ", lambda j: sps, True),
            ("∂v_pre/∂w_s\n= Eₛ  (eligibility)", lambda j: E[:, j], False),
            ("PRODUCT\n= ∂L/∂w_s", lambda j: dLdo * sps * E[:, j], False),
        ]
        wsym = "w_s"
        supt = ("SOMA weight gradient — chain-rule dissection (hand rule)\n"
                "∂L/∂w_s[0,j] = ∂L/∂o · ∂o/∂v_pre · ∂v_pre/∂w_s   "
                "(bottom row = orange curve in single_neuron_credit.png)")
    else:
        rows = [
            ("∂L/∂o\n(ρ·Wᵣᵀe / τ)", lambda j: dLdo, True),
            ("∂o/∂h\n= γ·s'ₛ", lambda j: gamma * sps, True),
            ("∂h/∂μ_t′\n= s'_d", lambda j: spd, True),
            ("∂μ_t′/∂w_d\n= P̂  (frozen elig.)", lambda j: P[:, j], False),
            ("PRODUCT\n= ∂L/∂w_d", lambda j: dLdo * gamma * sps * spd * P[:, j], False),
        ]
        wsym = "w_d"
        supt = ("DENDRITE weight gradient — chain-rule dissection (hand rule)\n"
                "∂L/∂w_d[0,j] = ∂L/∂o · ∂o/∂h · ∂h/∂μ_t′ · ∂μ_t′/∂w_d   "
                "(∂o/∂h = γ·s'ₛ carries the somatic surrogate)")

    nr = len(rows)
    fig, axes = plt.subplots(nr, 3, figsize=(14, 2.15 * nr), sharex=True, sharey="row")
    for r, (lbl, fn, shared) in enumerate(rows):
        for j in range(3):
            ax = axes[r, j]
            ax.fill_between(t, 0, 1, where=plateau, transform=ax.get_xaxis_transform(),
                            color="orange", alpha=0.08, step="mid", zorder=0)
            ax.axhline(0, color="k", lw=0.4)
            for tt in SPIKES[j]:
                if tt < T:
                    ax.axvline(tt, color=ch_col[j], ls=":", lw=0.9, alpha=0.6)
            col = "#555555" if shared else ch_col[j]
            ax.plot(t, fn(j), color=col, lw=1.6, marker=".", ms=3)
            ax.margins(y=0.18)
            if r == 0:
                ax.set_title(f"{wsym}[0,{j}]   (channel {j})", fontsize=10)
            if j == 0:
                ax.set_ylabel(lbl + ("  (shared)" if shared else ""), fontsize=8.5)
            if r == nr - 1:
                ax.set_xlabel("timestep")
    fig.suptitle(supt, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(_THIS, "out", "single_neuron_credit.png"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    mod_fix = cb.load_network_module(os.path.join(_ROOT, "BPTT_fix"), "net_fix")
    mod_auto = cb.load_network_module(os.path.join(_ROOT, "BPTT_fix_auto"), "net_auto")
    config = mod_fix.NeuronConfig()
    temp, bias = float(config.loss_temperature), float(config.loss_count_bias)

    from jax import random
    net_fix = mod_fix.Network(random.PRNGKey(a.seed), 3, [1], 2, config, dropout_rate=0.0)
    net_auto = mod_auto.Network(random.PRNGKey(a.seed), 3, [1], 2, config, dropout_rate=0.0)
    set_weights(net_fix); set_weights(net_auto)
    params = net_fix._params()
    tgt = net_fix._smooth_targets(TARGET)

    x = build_input(a.T)
    fwd = hand_forward(mod_fix, x, params, temp, bias, tgt)
    hand_soma, hand_dend = fwd["hand_soma"], fwd["hand_dend"]
    auto_soma, auto_dend = auto_pertimestep(mod_auto, x, params, temp, bias, tgt)

    # normalize hand to true-gradient scale (undo 1/temp and soma /8; here /8 cancels)
    hand_soma_n = hand_soma / temp
    hand_dend_n = hand_dend / temp

    # ── consistency checks: per-t series sum to the compiled totals ──
    gr_f, gs_f, gd_f, _, _ = cb.rule_grads(mod_fix, net_fix, x, TARGET, temp, bias)
    gr_a, gs_a, gd_a, _, _ = cb.auto_grads(mod_auto, net_auto, x, TARGET, temp, bias, prune=False)
    err_hand = max(np.max(np.abs(hand_soma.sum(0) / 8.0 - gs_f[0][0])),
                   np.max(np.abs(hand_dend.sum(0) - gd_f[0][0])))
    err_auto = max(np.max(np.abs(auto_soma.sum(0) - gs_a[0][0])),
                   np.max(np.abs(auto_dend.sum(0) - gd_a[0][0])))
    print(f"plateau steps: {int(fwd['h'].sum())}   somatic spikes: {int(fwd['o'].sum())}")
    print(f"sum(hand per-t) vs BPTT_fix total:      max abs diff = {err_hand:.2e} (expect ~0)")
    print(f"sum(auto per-t) vs BPTT_fix_auto total: max abs diff = {err_auto:.2e} (expect ~0)")
    print("totals (ascent, true scale):")
    for j in range(3):
        print(f"  W_d[0,{j}]  auto={auto_dend[:,j].sum():+.4e}  hand={hand_dend_n[:,j].sum():+.4e}")
    for j in range(3):
        print(f"  W_s[0,{j}]  auto={auto_soma[:,j].sum():+.4e}  hand={hand_soma_n[:,j].sum():+.4e}")

    cfgp = dict(v_th=float(config.v_th), gamma=float(config.gamma), mu_th=float(config.mu_th))
    plot(x, fwd, hand_soma_n, hand_dend_n, auto_soma, auto_dend, cfgp, a.out)
    print("wrote", a.out)

    # ── chain-rule dissection of the hand rule; verify product == orange curve ──
    g = float(fwd["gamma"])
    prod_soma = (fwd["delta_o"] / temp)[:, None] * fwd["sps"][:, None] * fwd["E_soma"]
    prod_dend = ((fwd["delta_o"] / temp) * g * fwd["sps"] * fwd["spd"])[:, None] * fwd["P_hat"]
    print(f"chain product == orange curve:  soma max diff = {np.max(np.abs(prod_soma - hand_soma_n)):.2e}, "
          f"dend max diff = {np.max(np.abs(prod_dend - hand_dend_n)):.2e} (expect ~0)")
    out_dir = os.path.dirname(a.out)
    for kind in ("soma", "dend"):
        p = os.path.join(out_dir, f"single_neuron_chain_{kind}.png")
        plot_chain(kind, fwd, temp, p)
        print("wrote", p)


if __name__ == "__main__":
    main()
