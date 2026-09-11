#!/usr/bin/env python3
"""Analyze a trained Soft_Reset two-compartment SHD model.

Produces, for a saved .npz model:
  1. Train + test accuracy.
  2. Per-digit accuracy + a confusion-matrix plot (test set).
  3. Mean readout ("prediction signal") + mean hidden firing rate.
  4. P(spike | in plateau)  vs  P(spike | outside plateau).
  5. A 4x4 figure: 4 random hidden neurons (one per column) showing, per row,
        row 1  forward  : mu (dendritic potential) and h (plateau flag)
        row 2  backward : dendritic learning signal per timestep (not accumulated)
        row 3  forward  : v (pre-reset soma) and v_eff (= v_th - gamma*h)
        row 4  backward : somatic learning signal per timestep (not accumulated)

Run (from the Soft_Reset/ directory, in the `dendroprop` conda env):
    python analyze_model.py [model.npz]

The model is trained at float32, so this script keeps jax_enable_x64 OFF to
reproduce the saved accuracies exactly.
"""
import os
import sys
import json
import argparse

# ── Precision: match training (float32). Must precede the jax.numpy import. ──
import jax
jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp
from jax import lax, vmap, jit, random
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
for _p in (_ROOT, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data.shd_binned import load_shd_binned
from network import Network, _TAU_ELIG_DEND_MS
from two_comp_neuron import TwoCompNeuron
from lif_neuron import LINeuron
from config import surrogate_sigma, surrogate_roof

# ── Colorblind-safe categorical palette (Okabe-Ito) ──
C = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "vermillion": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
    "yellow": "#F0E442", "black": "#111111", "grey": "#9AA0A6",
    "plateau": "#F7C79E",  # light orange band for plateau shading
}

# SHD: 0-9 English spoken digits, 10-19 German spoken digits.
SHD_NAMES = [
    "0 zero", "1 one", "2 two", "3 three", "4 four",
    "5 five", "6 six", "7 seven", "8 eight", "9 nine",
    "null", "eins", "zwei", "drei", "vier",
    "fünf", "sechs", "sieben", "acht", "neun",
]


# ══════════════════════════════════════════════════════════════════════
#  Instrumented forward pass — per-neuron spike / plateau counts.
#  Uses the SAME dynamics_step as training/inference (no dropout).
# ══════════════════════════════════════════════════════════════════════

def _counts_1s(x, w_dend, w_soma, alpha_s, alpha_d, T_p, config, alpha_w):
    """One sample -> (spikes_n, plateau_n, spike&plateau_n), each (N,)."""
    dend_in = x @ w_dend.T
    soma_in = x @ w_soma.T
    T = x.shape[0]
    n = w_dend.shape[0]
    tix = jnp.arange(T, dtype=jnp.int32)

    def step(carry, inp):
        state, spk, plat, spk_plat = carry
        d, s, t = inp
        state, o, _v_pre, h_new, _h_prev, _matp = TwoCompNeuron.dynamics_step(
            state, d, s, t, alpha_s, alpha_d, T_p, config, alpha_w,
        )
        of = o.astype(jnp.float32)
        hf = h_new.astype(jnp.float32)
        return (state, spk + of, plat + hf, spk_plat + of * hf), None

    h0 = (
        jnp.zeros(n), jnp.zeros(n),
        jnp.zeros(n, dtype=jnp.int32), jnp.zeros(n, dtype=jnp.int32),
        jnp.zeros(n), jnp.zeros(n),
    )
    init = (h0, jnp.zeros(n), jnp.zeros(n), jnp.zeros(n))
    (_, spk, plat, spk_plat), _ = lax.scan(step, init, (dend_in, soma_in, tix))
    return spk, plat, spk_plat


_counts_batch = jit(vmap(
    _counts_1s, in_axes=(0, None, None, None, None, None, None, None)
))


# ══════════════════════════════════════════════════════════════════════
#  Instrumented forward + backward for ONE sample — full per-timestep traces.
#  Mirrors network._forward_backward (pass 1 forward -> e; pass 2 backward),
#  but with dropout OFF and every per-step quantity recorded.
# ══════════════════════════════════════════════════════════════════════

def make_trace_fn(config):
    gamma = float(config.gamma)
    v_th = float(config.v_th)
    mu_th = float(config.mu_th)
    beta_s = float(config.beta_s)
    beta_s_dend = float(config.beta_s_dend)
    beta_d = float(config.beta_d)
    roof = bool(config.dend_surrogate_roof)
    a_elig = float(np.exp(-config.dt / _TAU_ELIG_DEND_MS))
    loss_temperature = float(config.loss_temperature)
    loss_count_bias = float(config.loss_count_bias)

    def trace_fn(x, w_dend, w_soma, w_readout, alpha_s, alpha_d, alpha_m,
                 T_p, alpha_w, target_smoothed):
        dend_inputs = x @ w_dend.T
        soma_inputs = x @ w_soma.T
        T = x.shape[0]
        n = w_dend.shape[0]
        k = w_dend.shape[1]
        j = w_readout.shape[0]
        tix = jnp.arange(T, dtype=jnp.int32)

        # ── Pass 1: forward -> readout voltage trace + mean voltage ──
        def step1(carry, inp):
            state, r_v, r_sum = carry
            d, s, t = inp
            state, o, _vp, _h, _hp, _matp = TwoCompNeuron.dynamics_step(
                state, d, s, t, alpha_s, alpha_d, T_p, config, alpha_w,
            )
            r_v, r_sum = LINeuron.integrate(r_v, r_sum, o.astype(jnp.float32),
                                            w_readout, alpha_m)
            return (state, r_v, r_sum), r_v

        h0 = (jnp.zeros(n), jnp.zeros(n), jnp.zeros(n, dtype=jnp.int32),
              jnp.zeros(n, dtype=jnp.int32), jnp.zeros(n), jnp.zeros(n))
        (_, _, r_sum), r_v_trace = lax.scan(
            step1, (h0, jnp.zeros(j), jnp.zeros(j)), (dend_inputs, soma_inputs, tix))
        mean_voltage = r_sum / T

        scaled = mean_voltage / loss_temperature + loss_count_bias
        probs = jnp.exp(scaled - jnp.max(scaled))
        probs = probs / jnp.sum(probs)
        prediction = jnp.argmax(mean_voltage)
        e = target_smoothed - probs
        readout_seed = w_readout.T @ e                       # (N,)
        rho = (1.0 / T) * (1.0 - alpha_m ** (T - tix)) / (1.0 - alpha_m)

        # ── Pass 2: re-run forward + eligibility -> per-step backward signals ──
        def step2(carry, inp):
            hc, = carry
            d, s, x_t, t, rho_t = inp
            hc, o, v_pre, h_new, h_prev, mu_at_tp = TwoCompNeuron.forward_step(
                hc, d, s, t, alpha_s, alpha_d, T_p, config, alpha_w,
            )
            mu_c, v_c, h_c, tp_c, matp_c, E_soma_c, dmu_c, dmu_atp_c, w_c = hc
            E_soma_new = TwoCompNeuron.update_somatic_eligibility(
                E_soma_c, x_t.astype(jnp.float32), alpha_s)
            dmu_new, dmu_atp_new = TwoCompNeuron.update_dendritic_eligibility(
                dmu_c, dmu_atp_c, x_t.astype(jnp.float32), h_prev, a_elig)
            hc = (mu_c, v_c, h_c, tp_c, matp_c, E_soma_new, dmu_new, dmu_atp_new, w_c)

            soma_dist = v_pre + gamma * h_new - v_th
            sp_hidden = surrogate_sigma(soma_dist, beta_s)
            hp_hidden = surrogate_sigma(mu_at_tp - mu_th, beta_d)
            delta = rho_t * readout_seed

            l_soma = delta * sp_hidden                        # somatic learning signal
            if roof:
                sp_hidden_d = surrogate_roof(v_pre, v_th - gamma, v_th, beta_s_dend)
                eta_d = delta * sp_hidden_d * hp_hidden
            else:
                sp_hidden_d = surrogate_sigma(soma_dist, beta_s_dend)
                eta_d = delta * sp_hidden_d * gamma * hp_hidden

            v_th_eff = v_th - gamma * h_new
            # per-step gradient-contribution norms (|LS| * ||eligibility||)
            g_soma_norm = jnp.abs(l_soma) * jnp.linalg.norm(E_soma_new)
            g_dend_norm = jnp.abs(eta_d) * jnp.linalg.norm(dmu_atp_new, axis=1)

            rec = dict(
                mu=mu_c, h=h_new.astype(jnp.float32), o=o.astype(jnp.float32),
                v_pre=v_pre, v_th_eff=v_th_eff,
                l_soma=l_soma, eta_d=eta_d,
                sp_hidden=sp_hidden, sp_hidden_d=sp_hidden_d, hp_hidden=hp_hidden,
                delta=delta, g_soma_norm=g_soma_norm, g_dend_norm=g_dend_norm,
            )
            return (hc,), rec

        h9 = (jnp.zeros(n), jnp.zeros(n), jnp.zeros(n, dtype=jnp.int32),
              jnp.zeros(n, dtype=jnp.int32), jnp.zeros(n),
              jnp.zeros(k), jnp.zeros((n, k)), jnp.zeros((n, k)), jnp.zeros(n))
        (_,), rec = lax.scan(
            step2, (h9,), (dend_inputs, soma_inputs, x, tix, rho))

        return mean_voltage, probs, prediction, r_v_trace, rec

    return jit(trace_fn)


# ══════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════

def batched_predict(net, X, batch=256):
    preds = []
    for s in range(0, len(X), batch):
        xb = jnp.asarray(X[s:s + batch])
        preds.append(np.asarray(net.batch_predict(xb)))
    return np.concatenate(preds) if preds else np.array([], dtype=int)


def collect_counts(net, X, batch=256):
    """Per-neuron spike/plateau/spike&plateau totals over dataset X."""
    N = net.n_hidden
    tot_spk = np.zeros(N)
    tot_plat = np.zeros(N)
    tot_spk_plat = np.zeros(N)
    for s in range(0, len(X), batch):
        xb = jnp.asarray(X[s:s + batch])
        spk, plat, spk_plat = _counts_batch(
            xb, net.hidden.w_dend, net.hidden.w_soma,
            net.hidden.alpha_s, net.hidden.alpha_d, net.hidden.T_p,
            net.config, net.hidden.alpha_w,
        )
        tot_spk += np.asarray(spk).sum(0)
        tot_plat += np.asarray(plat).sum(0)
        tot_spk_plat += np.asarray(spk_plat).sum(0)
    return tot_spk, tot_plat, tot_spk_plat


def shade_plateaus(ax, t_ms, h, label=None):
    """Shade the contiguous intervals where h == 1."""
    h = np.asarray(h) > 0.5
    if not h.any():
        return
    dt = t_ms[1] - t_ms[0] if len(t_ms) > 1 else 1.0
    start = None
    first = True
    for i, on in enumerate(h):
        if on and start is None:
            start = t_ms[i]
        if (not on or i == len(h) - 1) and start is not None:
            end = t_ms[i] + (dt if on else 0.0)
            ax.axvspan(start, end, color=C["plateau"], alpha=0.55, lw=0,
                       label=(label if first else None), zorder=0)
            first = False
            start = None


def mark_spikes(ax, t_ms, o, y=None, color=None):
    """Draw short ticks at the bottom of ax for each spike (o == 1)."""
    o = np.asarray(o) > 0.5
    if not o.any():
        return
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    y0 = ymin + 0.02 * span if y is None else y
    y1 = ymin + 0.10 * span if y is None else y + 0.08 * span
    for tt in t_ms[o]:
        ax.plot([tt, tt], [y0, y1], color=(color or C["vermillion"]),
                lw=0.8, alpha=0.9, zorder=5)


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?",
                    default=os.path.join(_SCRIPT_DIR, "models",
                                         "shd_seed12_20260909_150601.npz"),
                    help="Path to the .npz model.")
    ap.add_argument("--outdir", default=os.path.join(_SCRIPT_DIR, "analysis_out"))
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for picking the example sample + 4 neurons. "
                         "Change it to re-roll both.")
    ap.add_argument("--target", type=int, default=None,
                    help="Restrict the example sample to this true class (0-19; "
                         "see the per-digit list). Default: any correctly-classified "
                         "test sample.")
    ap.add_argument("--example", type=int, default=None,
                    help="Pin an exact test-sample index (overrides --target). Use "
                         "with different --seed to keep the sample but re-roll neurons.")
    ap.add_argument("--plateau_split", choices=["test", "train"], default="test",
                    help="Dataset used for firing-rate / plateau statistics.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ── Load model + provenance ──
    print(f"Loading model: {args.model}", flush=True)
    net = Network.load(args.model)
    meta = net.meta
    cfg = net.config
    a = meta["extra"].get("args", {})
    print(f"  arch: {net.n_inputs} -> {net.n_hidden} (2-comp) -> {net.n_outputs}")
    print(f"  gamma={cfg.gamma}  roof_surrogate={cfg.dend_surrogate_roof}  "
          f"dt={cfg.dt}ms  saved final/best acc = "
          f"{meta['extra'].get('final_acc'):.2f}/{meta['extra'].get('best_acc'):.2f}%")

    # ── Load data with the SAME preprocessing the model was trained with ──
    print("Loading SHD data...", flush=True)
    X_tr, y_tr, _, X_te, y_te, _ = load_shd_binned(
        bin_size_ms=a.get("bin_size_ms", 4.0),
        collapse_factor=a.get("collapse_factor", 5),
        max_duration_ms=a.get("max_duration_ms", 700.0),
        binarize=False, dtype=np.float32,
    )
    y_tr = np.asarray(y_tr).astype(int)
    y_te = np.asarray(y_te).astype(int)
    T = X_tr.shape[1]
    J = net.n_outputs
    dt = float(cfg.dt)
    t_ms = np.arange(T) * dt
    print(f"  train={len(y_tr)}  test={len(y_te)}  T={T}  n_inputs={X_tr.shape[2]}")

    summary = []

    def log(s):
        print(s, flush=True)
        summary.append(s)

    log("\n" + "=" * 64)
    log(f"MODEL: {os.path.basename(args.model)}")
    log("=" * 64)

    # ── 1 & 2. Accuracies (overall, per-digit) + confusion matrix ──
    log("\n[1/2] Accuracies ...")
    pred_tr = batched_predict(net, X_tr)
    pred_te = batched_predict(net, X_te)
    acc_tr = 100.0 * np.mean(pred_tr == y_tr)
    acc_te = 100.0 * np.mean(pred_te == y_te)
    log(f"  Train accuracy: {acc_tr:.2f}%   ({np.sum(pred_tr == y_tr)}/{len(y_tr)})")
    log(f"  Test  accuracy: {acc_te:.2f}%   ({np.sum(pred_te == y_te)}/{len(y_te)})")

    # Confusion matrix (test), row = true class
    cm = np.zeros((J, J), dtype=int)
    for yt, yp in zip(y_te, pred_te):
        cm[yt, yp] += 1
    per_digit_te = np.array([100.0 * cm[i, i] / max(cm[i].sum(), 1) for i in range(J)])
    # per-digit train
    cm_tr = np.zeros((J, J), dtype=int)
    for yt, yp in zip(y_tr, pred_tr):
        cm_tr[yt, yp] += 1
    per_digit_tr = np.array([100.0 * cm_tr[i, i] / max(cm_tr[i].sum(), 1) for i in range(J)])

    log("\n  Per-digit accuracy (test):")
    for i in range(J):
        log(f"    {SHD_NAMES[i]:>10s}: {per_digit_te[i]:5.1f}%  (n={cm[i].sum()})")

    # --- Confusion-matrix plot (row-normalized recall, counts annotated) ---
    cm_norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(J)); ax.set_yticks(range(J))
    ax.set_xticklabels(SHD_NAMES, rotation=90, fontsize=8)
    ax.set_yticklabels(SHD_NAMES, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"SHD confusion matrix (test, n={len(y_te)}) — overall {acc_te:.2f}%")
    for i in range(J):
        for jj in range(J):
            c = cm[i, jj]
            if c:
                ax.text(jj, i, str(c), ha="center", va="center", fontsize=6,
                        color="white" if cm_norm[i, jj] > 0.5 else "#333333")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized (recall)")
    fig.tight_layout()
    p_cm = os.path.join(args.outdir, "confusion_matrix.png")
    fig.savefig(p_cm, dpi=150); plt.close(fig)
    log(f"  -> {p_cm}")

    # --- Per-digit accuracy bar (train vs test) ---
    fig, ax = plt.subplots(figsize=(13, 5))
    xpos = np.arange(J)
    ax.bar(xpos - 0.2, per_digit_tr, width=0.4, color=C["sky"], label="train")
    ax.bar(xpos + 0.2, per_digit_te, width=0.4, color=C["blue"], label="test")
    ax.axhline(acc_te, color=C["vermillion"], ls="--", lw=1,
               label=f"test overall {acc_te:.1f}%")
    ax.set_xticks(xpos); ax.set_xticklabels(SHD_NAMES, rotation=90, fontsize=8)
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 105)
    ax.set_title("Per-digit accuracy")
    ax.grid(axis="y", color=C["grey"], alpha=0.25)
    ax.legend()
    fig.tight_layout()
    p_pd = os.path.join(args.outdir, "per_digit_accuracy.png")
    fig.savefig(p_pd, dpi=150); plt.close(fig)
    log(f"  -> {p_pd}")

    # ── 3. Mean readout (prediction signal) + mean hidden firing rate ──
    log("\n[3]  Mean readout + hidden firing rate ...")
    X_stat = X_te if args.plateau_split == "test" else X_tr
    y_stat = y_te if args.plateau_split == "test" else y_tr
    tot_spk, tot_plat, tot_spk_plat = collect_counts(net, X_stat)
    n_stat = len(X_stat)
    rate_per_neuron = tot_spk / (n_stat * T)          # spike prob per step, per neuron
    mean_rate = float(rate_per_neuron.mean())
    log(f"  Mean hidden firing rate ({args.plateau_split}): {mean_rate:.4f} "
        f"(spikes per neuron per timestep)")
    log(f"    per-neuron rate: min={rate_per_neuron.min():.4f}  "
        f"median={np.median(rate_per_neuron):.4f}  max={rate_per_neuron.max():.4f}  "
        f"dead(<1e-4)={int(np.sum(rate_per_neuron < 1e-4))}/{net.n_hidden}")

    # mean readout voltage per class = the logit source (prediction signal)
    trace_fn = make_trace_fn(cfg)

    def smooth_target(label):
        oh = np.eye(J)[label]
        return jnp.asarray(oh * (1 - cfg.loss_label_smoothing)
                           + cfg.loss_label_smoothing / J)

    # readout voltage traces for a representative example. Selection priority:
    #   --example (exact index)  >  --target (a chosen class)  >  any correct sample.
    correct_te = np.where(pred_te == y_te)[0]
    if args.example is not None:
        ex = int(args.example)
    elif args.target is not None:
        cand = np.where((pred_te == y_te) & (y_te == args.target))[0]
        if len(cand) == 0:  # no correct one for this class -> fall back to any of it
            cand = np.where(y_te == args.target)[0]
        ex = int(rng.choice(cand)) if len(cand) else 0
    else:
        ex = int(rng.choice(correct_te)) if len(correct_te) else 0
    ex_x = jnp.asarray(X_te[ex]); ex_y = int(y_te[ex])
    mean_v, probs, pred, r_v_trace, rec = trace_fn(
        ex_x, net.hidden.w_dend, net.hidden.w_soma, net.readout.w,
        net.hidden.alpha_s, net.hidden.alpha_d, net.readout.alpha_m,
        net.hidden.T_p, net.hidden.alpha_w, smooth_target(ex_y),
    )
    mean_v = np.asarray(mean_v); r_v_trace = np.asarray(r_v_trace)
    log(f"  Example test sample #{ex}: true={SHD_NAMES[ex_y]}  "
        f"pred={SHD_NAMES[int(pred)]}  p(true)={float(probs[ex_y]):.3f}")
    log(f"  Mean readout voltage (this example): true-class={mean_v[ex_y]:.3f}  "
        f"argmax={mean_v[int(pred)]:.3f}  mean|logit|={np.mean(np.abs(mean_v)):.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax = axes[0]
    for c in range(J):
        if c == ex_y:
            continue
        ax.plot(t_ms, r_v_trace[:, c], color=C["grey"], lw=0.8, alpha=0.5)
    ax.plot(t_ms, r_v_trace[:, ex_y], color=C["blue"], lw=2.2,
            label=f"true class ({SHD_NAMES[ex_y]})")
    if int(pred) != ex_y:
        ax.plot(t_ms, r_v_trace[:, int(pred)], color=C["vermillion"], lw=2.0,
                label=f"pred class ({SHD_NAMES[int(pred)]})")
    ax.set_xlabel("time (ms)"); ax.set_ylabel("readout membrane voltage")
    ax.set_title(f"Readout / prediction signal — test #{ex}")
    ax.grid(color=C["grey"], alpha=0.2); ax.legend(fontsize=8)

    ax = axes[1]
    ax.hist(rate_per_neuron, bins=40, color=C["blue"], alpha=0.85)
    ax.axvline(mean_rate, color=C["vermillion"], ls="--", lw=1.5,
               label=f"mean = {mean_rate:.4f}")
    ax.set_xlabel("per-neuron firing rate (spikes / step)")
    ax.set_ylabel("# hidden neurons")
    ax.set_title(f"Hidden firing-rate distribution ({args.plateau_split})")
    ax.grid(axis="y", color=C["grey"], alpha=0.2); ax.legend()
    fig.tight_layout()
    p_ro = os.path.join(args.outdir, "readout_and_firing_rate.png")
    fig.savefig(p_ro, dpi=150); plt.close(fig)
    log(f"  -> {p_ro}")

    # ── 4. P(spike | plateau) vs P(spike | no plateau) ──
    log("\n[4]  Spike probability inside vs outside a plateau ...")
    total_steps = n_stat * T * net.n_hidden
    tot_spk_all = tot_spk.sum()
    tot_plat_all = tot_plat.sum()
    tot_spk_plat_all = tot_spk_plat.sum()
    tot_spk_nonplat_all = tot_spk_all - tot_spk_plat_all
    tot_nonplat_all = total_steps - tot_plat_all
    p_spk_plat = tot_spk_plat_all / max(tot_plat_all, 1)
    p_spk_nonplat = tot_spk_nonplat_all / max(tot_nonplat_all, 1)
    plateau_frac = tot_plat_all / total_steps
    log(f"  Fraction of (neuron,step) in a plateau: {plateau_frac:.4f}")
    log(f"  P(spike | in plateau)     = {p_spk_plat:.4f}   "
        f"({int(tot_spk_plat_all)}/{int(tot_plat_all)})")
    log(f"  P(spike | outside plateau)= {p_spk_nonplat:.4f}   "
        f"({int(tot_spk_nonplat_all)}/{int(tot_nonplat_all)})")
    ratio = p_spk_plat / max(p_spk_nonplat, 1e-12)
    log(f"  Ratio (plateau / outside) = {ratio:.2f}x")

    fig, ax = plt.subplots(figsize=(5.5, 5))
    bars = ax.bar([0, 1], [p_spk_nonplat, p_spk_plat],
                  color=[C["grey"], C["orange"]], width=0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["outside plateau\n(h=0)", "in plateau\n(h=1)"])
    ax.set_ylabel("P(spike | state)")
    ax.set_title(f"Spike probability — plateau vs not  ({ratio:.1f}x)")
    for b, v in zip(bars, [p_spk_nonplat, p_spk_plat]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                ha="center", va="bottom", fontsize=10)
    ax.grid(axis="y", color=C["grey"], alpha=0.25)
    fig.tight_layout()
    p_pl = os.path.join(args.outdir, "plateau_spike_prob.png")
    fig.savefig(p_pl, dpi=150); plt.close(fig)
    log(f"  -> {p_pl}")

    # ── 4b. Per-neuron gain modulation: mean ± std P(spike) out vs in plateau ──
    #  Same quantity as [4], but resolved per neuron so we have a real spread to
    #  draw: each neuron contributes P(spike|h=0) and P(spike|h=1); we plot the
    #  population mean at each state with std error bars and a line connecting the
    #  two means (the "gain" the plateau applies).
    log("\n[4b] Per-neuron gain modulation (mean ± std across neurons) ...")
    min_plat_steps = 50  # need enough plateau samples for a stable per-neuron rate
    denom_plat = tot_plat
    denom_nonplat = n_stat * T - tot_plat
    valid = (denom_plat >= min_plat_steps) & (denom_nonplat > 0)
    p_out_n = (tot_spk[valid] - tot_spk_plat[valid]) / denom_nonplat[valid]
    p_in_n = tot_spk_plat[valid] / denom_plat[valid]
    n_valid = int(valid.sum())
    mu_out, sd_out = float(p_out_n.mean()), float(p_out_n.std())
    mu_in, sd_in = float(p_in_n.mean()), float(p_in_n.std())
    gain_pop = mu_in / max(mu_out, 1e-12)
    per_neuron_gain = float(np.median(p_in_n / np.maximum(p_out_n, 1e-9)))
    log(f"  Neurons with >= {min_plat_steps} plateau steps: {n_valid}/{net.n_hidden}")
    log(f"  P(spike | outside): mean={mu_out:.4f}  std={sd_out:.4f}")
    log(f"  P(spike | in):      mean={mu_in:.4f}  std={sd_in:.4f}")
    log(f"  Population gain (mean_in/mean_out) = {gain_pop:.2f}x   "
        f"median per-neuron gain = {per_neuron_gain:.2f}x")

    fig, ax = plt.subplots(figsize=(5.5, 5))
    xk = np.array([0.0, 1.0])
    # population mean ± std, with the connecting line
    ax.plot(xk, [mu_out, mu_in], color=C["orange"], lw=2.6, zorder=3,
            marker="o", ms=10, mfc=C["orange"], mec=C["black"], mew=1.4,
            label="population mean")
    ax.errorbar(xk, [mu_out, mu_in], yerr=[sd_out, sd_in], fmt="none",
                ecolor=C["black"], elinewidth=1.8, capsize=9, capthick=1.8,
                zorder=4, label="± 1 std across neurons")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["outside plateau\n(h=0)", "in plateau\n(h=1)"])
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("P(spike | state)  per neuron")
    ax.set_title(f"Plateau gain modulation — {gain_pop:.1f}x\n"
                 f"mean ± std across {n_valid} hidden neurons")
    for xi, m in zip(xk, [mu_out, mu_in]):
        ax.text(xi + 0.07, m, f"{m:.3f}", va="center", ha="left", fontsize=11,
                fontweight="bold")
    ax.grid(axis="y", color=C["grey"], alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    p_gm = os.path.join(args.outdir, "plateau_gain_modulation.png")
    fig.savefig(p_gm, dpi=150); plt.close(fig)
    log(f"  -> {p_gm}")

    # ── 5. 4x4 forward/backward traces for 4 random hidden neurons ──
    log("\n[5]  Forward/backward traces for 4 random hidden neurons ...")
    mu = np.asarray(rec["mu"]); h = np.asarray(rec["h"]); o = np.asarray(rec["o"])
    v_pre = np.asarray(rec["v_pre"]); v_eff = np.asarray(rec["v_th_eff"])
    eta_d = np.asarray(rec["eta_d"]); l_soma = np.asarray(rec["l_soma"])

    # prefer neurons that are actually active on this example (>=1 spike),
    # else fall back to plateau-active, else any.
    active = np.where(o.sum(0) > 0)[0]
    pool = active if len(active) >= 4 else np.where(h.sum(0) > 0)[0]
    if len(pool) < 4:
        pool = np.arange(net.n_hidden)
    neurons = sorted(rng.choice(pool, size=4, replace=False).tolist())
    log(f"  Example test sample #{ex} (true={SHD_NAMES[ex_y]}, "
        f"pred={SHD_NAMES[int(pred)]}); neurons={neurons}")

    fig, axes = plt.subplots(4, 4, figsize=(18, 12), sharex=True)
    row_titles = [
        "forward: μ (dendrite) & plateau h",
        "backward: dendritic learning signal / step",
        "forward: v (soma) & v_eff",
        "backward: somatic learning signal / step",
    ]
    for col, ndx in enumerate(neurons):
        hh = h[:, ndx]
        # Row 1 — mu & plateau
        ax = axes[0, col]
        shade_plateaus(ax, t_ms, hh, label="plateau (h=1)")
        ax.plot(t_ms, mu[:, ndx], color=C["blue"], lw=1.6, label="μ")
        ax.axhline(float(cfg.mu_th), color=C["grey"], ls=":", lw=1, label="μ_th")
        ax.set_title(f"hidden neuron #{ndx}  "
                     f"(T_p={int(net.hidden.T_p[ndx])} steps, "
                     f"{o[:, ndx].sum():.0f} spikes)", fontsize=10)
        if col == 0:
            ax.set_ylabel(row_titles[0], fontsize=9)
            ax.legend(fontsize=7, loc="upper right")

        # Row 2 — dendritic backward (per timestep)
        ax = axes[1, col]
        shade_plateaus(ax, t_ms, hh)
        ax.axhline(0, color=C["grey"], lw=0.8)
        ax.plot(t_ms, eta_d[:, ndx], color=C["green"], lw=1.4, label="η_d")
        if col == 0:
            ax.set_ylabel(row_titles[1], fontsize=9)
            ax.legend(fontsize=7, loc="upper right")

        # Row 3 — soma v & v_eff
        ax = axes[2, col]
        shade_plateaus(ax, t_ms, hh)
        ax.plot(t_ms, v_pre[:, ndx], color=C["blue"], lw=1.4, label="v (pre-reset)")
        ax.plot(t_ms, v_eff[:, ndx], color=C["vermillion"], lw=1.2, ls="--",
                label="v_eff")
        mark_spikes(ax, t_ms, o[:, ndx], color=C["black"])
        if col == 0:
            ax.set_ylabel(row_titles[2], fontsize=9)
            ax.legend(fontsize=7, loc="upper right")

        # Row 4 — somatic backward (per timestep)
        ax = axes[3, col]
        shade_plateaus(ax, t_ms, hh)
        ax.axhline(0, color=C["grey"], lw=0.8)
        ax.plot(t_ms, l_soma[:, ndx], color=C["purple"], lw=1.4, label="δ·s'_s")
        ax.set_xlabel("time (ms)")
        if col == 0:
            ax.set_ylabel(row_titles[3], fontsize=9)
            ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        f"Forward & backward per timestep — {os.path.basename(args.model)} — "
        f"test #{ex} (true={SHD_NAMES[ex_y]}, pred={SHD_NAMES[int(pred)]})",
        fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    p_tr = os.path.join(args.outdir, "neuron_forward_backward_traces.png")
    fig.savefig(p_tr, dpi=150); plt.close(fig)
    log(f"  -> {p_tr}")

    # ── Write text summary ──
    p_txt = os.path.join(args.outdir, "summary.txt")
    with open(p_txt, "w") as f:
        f.write("\n".join(summary) + "\n")
    log(f"\nWrote summary -> {p_txt}")
    log(f"All outputs in: {args.outdir}")


if __name__ == "__main__":
    main()
