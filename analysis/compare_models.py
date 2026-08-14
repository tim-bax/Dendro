#!/usr/bin/env python3
"""Compare two trained SHD models — BPTT (frozen, scalar adaptation) vs
BPTT_adapt (per-neuron trained a_adapt/b_adapt) — on their internal dynamics.

Reports (side-by-side, per hidden layer):
  * hidden firing rate
  * plateau timing (occupancy + mean event duration)
  * spike probability inside vs outside plateau
  * readout mean voltage (predicted class + margin)
  * sparsity / dead-neuron stats
and draws, per model & layer, 5 random neurons' somatic (voltage + dynamic
threshold + spikes) and dendritic (mu + plateau state) traces on one
correctly-classified test sample.

The two checkpoints have DIFFERENT input widths (BPTT collapse_factor=6 -> 117
channels; BPTT_adapt collapse_factor=5 -> 140) and DIFFERENT configs (v_th,
adaptation). Each model is therefore fed its own test set, rebuilt from the
preprocessing args stored inside the checkpoint. This is a side-by-side
characterization, not a controlled ablation.

The dynamics here are copied faithfully from
BPTT_adapt/two_comp_neuron.py::TwoCompNeuron.forward_step and
BPTT_adapt/lif_neuron.py::LINeuron.forward_step. We load raw arrays straight
from the .npz (not Network.load) so a single script can drive both model
variants without importing their two incompatible network.py modules.
"""
import os
import sys
import json
import argparse

import numpy as np

# Repo root on the path so `data.shd_binned` imports regardless of CWD.
_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import jax
jax.config.update("jax_enable_x64", True)  # forward casts to float64 internally
import jax.numpy as jnp
from jax import lax, vmap, jit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.shd_binned import load_shd_binned


# Defaults: the highest-final-accuracy recent checkpoint of each model.
DEFAULT_BPTT = os.path.join(_ROOT, "BPTT", "models", "shd_seed12_20260806_155834.npz")
DEFAULT_ADAPT = os.path.join(_ROOT, "BPTT_adapt", "models", "shd_seed12_20260810_152443.npz")


# ══════════════════════════════════════════════════════════════════════
#  Loading
# ══════════════════════════════════════════════════════════════════════

def load_model(path):
    """Read raw arrays + meta from a checkpoint into a flat params dict.

    a_adapt/b_adapt: use the saved per-neuron vectors if present (BPTT_adapt),
    else fill from the scalar config values (BPTT).
    """
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    arch = meta["arch"]
    cfg = meta["config"]
    args = meta.get("extra", {}).get("args", {})
    L = len(arch["hidden_sizes"])

    w_dend = [jnp.asarray(d[f"w_dend_{l}"], dtype=jnp.float64) for l in range(L)]
    w_soma = [jnp.asarray(d[f"w_soma_{l}"], dtype=jnp.float64) for l in range(L)]
    T_p = [jnp.asarray(d[f"T_p_{l}"]).astype(jnp.int32) for l in range(L)]

    a_adapt, b_adapt = [], []
    for l in range(L):
        n = w_dend[l].shape[0]
        if f"a_adapt_{l}" in d.files:
            a_adapt.append(jnp.asarray(d[f"a_adapt_{l}"], dtype=jnp.float64))
            b_adapt.append(jnp.asarray(d[f"b_adapt_{l}"], dtype=jnp.float64))
        else:
            a_adapt.append(jnp.full((n,), float(cfg["a_adapt"]), dtype=jnp.float64))
            b_adapt.append(jnp.full((n,), float(cfg["b_adapt"]), dtype=jnp.float64))

    p = dict(
        L=L,
        w_dend=w_dend, w_soma=w_soma, T_p=T_p,
        w_readout=jnp.asarray(d["w_readout"], dtype=jnp.float64),
        a_adapt=a_adapt, b_adapt=b_adapt,
        alpha_s=float(d["alpha_s"]), alpha_d=float(d["alpha_d"]),
        alpha_w=float(d["alpha_w"]), alpha_m=float(d["alpha_m"]),
        mu_th=float(cfg["mu_th"]), v_th=float(cfg["v_th"]), gamma=float(cfg["gamma"]),
        hidden_sizes=list(arch["hidden_sizes"]),
        n_outputs=int(arch["n_outputs"]), n_inputs=int(arch["n_inputs"]),
        per_neuron_adapt=(f"a_adapt_0" in d.files),
    )
    return p, meta, args


def load_test_set(args, dtype=np.float32):
    """Rebuild the model's own SHD test split from its stored preprocessing args."""
    X_tr, y_tr, _, X_te, y_te, _ = load_shd_binned(
        bin_size_ms=float(args.get("bin_size_ms", 4.0)),
        collapse_factor=int(args.get("collapse_factor", 5)),
        max_duration_ms=float(args.get("max_duration_ms", 700.0)),
        binarize=False,
        dtype=dtype,
    )
    return np.asarray(X_te), np.asarray(y_te).astype(np.int64)


# ══════════════════════════════════════════════════════════════════════
#  Forward dynamics (faithful copy of forward_step, forward-only)
# ══════════════════════════════════════════════════════════════════════

def _init_hidden(n):
    return (
        jnp.zeros(n),                    # mu
        jnp.zeros(n),                    # v (post-reset)
        jnp.zeros(n, dtype=jnp.int32),   # h  (plateau)
        jnp.zeros(n, dtype=jnp.int32),   # t_prime
        jnp.zeros(n),                    # mu_at_tprime
        jnp.zeros(n),                    # w  (adaptation current)
    )


def make_runners(p):
    """Build jitted stats (batched) and single-sample trace runners for model p."""
    L = p["L"]
    w_dend, w_soma = p["w_dend"], p["w_soma"]
    w_readout = p["w_readout"]
    T_p, a_adapt, b_adapt = p["T_p"], p["a_adapt"], p["b_adapt"]
    alpha_s, alpha_d, alpha_w, alpha_m = (
        p["alpha_s"], p["alpha_d"], p["alpha_w"], p["alpha_m"])
    mu_th, v_th, gamma = p["mu_th"], p["v_th"], p["gamma"]
    hidden_sizes = p["hidden_sizes"]
    J = p["n_outputs"]

    def hidden_step(carry_l, dend_in, soma_in, t, l):
        mu_p, v_p, h_p, tp_p, matp_p, w_p = carry_l
        t_prime = jnp.where(t == 0, 0, jnp.where(h_p == 1, tp_p, t))
        mu = jnp.where(t > 0, alpha_d * mu_p + (1 - h_p) * dend_in, dend_in)
        mu_at_tp = jnp.where(h_p == 0, mu, matp_p)
        plat_dur = t - t_prime
        h = jnp.where(
            (mu_at_tp >= mu_th) & (plat_dur <= T_p[l]) & (plat_dur >= 0),
            1, 0,
        ).astype(jnp.int32)
        v_pre = jnp.where(t > 0, alpha_s * v_p + soma_in - w_p, soma_in)
        o = jnp.where(v_pre >= v_th - gamma * h, 1, 0).astype(jnp.int32)
        v = v_pre * (1 - o)
        w = alpha_w * w_p + (1 - alpha_w) * a_adapt[l] * v_pre + b_adapt[l] * o
        new = (mu, v, h, t_prime, mu_at_tp, w)
        return new, o, v_pre, h, mu

    # ── Aggregate statistics over one sample (accumulating scan) ──
    def run_sample_stats(dend0, soma0):
        T = dend0.shape[0]
        tidx = jnp.arange(T, dtype=jnp.int32)
        h_states = [_init_hidden(n) for n in hidden_sizes]
        zeros = [jnp.zeros(n) for n in hidden_sizes]
        init = (
            h_states, jnp.zeros(J), jnp.zeros(J),      # readout v, sum_v
            list(zeros), list(zeros), list(zeros), list(zeros),  # spk, plat, sip, edge
        )

        def step(carry, xs):
            h_st, r_v, r_sumv, spk, plat, sip, edge = carry
            dend0_t, soma0_t, t = xs
            new_h, new_spk, new_plat, new_sip, new_edge = [], [], [], [], []
            o_prev = None
            for l in range(L):
                if l == 0:
                    din, sin = dend0_t, soma0_t
                else:
                    din = o_prev @ w_dend[l].T
                    sin = o_prev @ w_soma[l].T
                h_prev_plateau = h_st[l][2]
                nc, o, v_pre, h, mu = hidden_step(h_st[l], din, sin, t, l)
                of = o.astype(jnp.float64)
                hf = h.astype(jnp.float64)
                new_h.append(nc)
                new_spk.append(spk[l] + of)
                new_plat.append(plat[l] + hf)
                new_sip.append(sip[l] + of * hf)
                onset = ((h == 1) & (h_prev_plateau == 0)).astype(jnp.float64)
                new_edge.append(edge[l] + onset)
                o_prev = of
            r_v = alpha_m * r_v + o_prev @ w_readout.T
            r_sumv = r_sumv + r_v
            return (new_h, r_v, r_sumv, new_spk, new_plat, new_sip, new_edge), None

        (_, _, r_sumv, spk, plat, sip, edge), _ = lax.scan(
            step, init, (dend0, soma0, tidx))
        mean_v = r_sumv / T
        return mean_v, spk, plat, sip, edge

    stats_batch = jit(vmap(run_sample_stats))

    # ── Full per-timestep traces for one sample (for plotting) ──
    def run_sample_trace(dend0, soma0):
        T = dend0.shape[0]
        tidx = jnp.arange(T, dtype=jnp.int32)
        h_states = [_init_hidden(n) for n in hidden_sizes]

        def step(carry, xs):
            h_st = carry
            dend0_t, soma0_t, t = xs
            new_h, v_out, h_out, mu_out, o_out = [], [], [], [], []
            o_prev = None
            for l in range(L):
                if l == 0:
                    din, sin = dend0_t, soma0_t
                else:
                    din = o_prev @ w_dend[l].T
                    sin = o_prev @ w_soma[l].T
                nc, o, v_pre, h, mu = hidden_step(h_st[l], din, sin, t, l)
                new_h.append(nc)
                v_out.append(v_pre)
                h_out.append(h.astype(jnp.float64))
                mu_out.append(mu)
                o_out.append(o.astype(jnp.float64))
                o_prev = o.astype(jnp.float64)
            return new_h, (v_out, h_out, mu_out, o_out)

        _, (v_out, h_out, mu_out, o_out) = lax.scan(
            step, h_states, (dend0, soma0, tidx))
        # Each leaf is a length-L list of (T, N_l) arrays.
        return v_out, h_out, mu_out, o_out

    trace_one = jit(run_sample_trace)
    return stats_batch, trace_one


def layer0_drive(X, p):
    """Precompute layer-0 synaptic drive: (B,T,in) @ (N0,in)^T -> (B,T,N0)."""
    dend0 = X @ p["w_dend"][0].T
    soma0 = X @ p["w_soma"][0].T
    return dend0, soma0


# ══════════════════════════════════════════════════════════════════════
#  Metric reduction
# ══════════════════════════════════════════════════════════════════════

def reduce_stats(mean_v, spk, plat, sip, edge, y):
    """Turn per-sample accumulators into per-layer summary dicts."""
    L = len(spk)
    preds = np.asarray(jnp.argmax(mean_v, axis=1))
    y = np.asarray(y)
    acc = 100.0 * float(np.mean(preds == y))

    # Readout margin: predicted-class mean-v minus runner-up, averaged.
    sorted_mv = np.sort(np.asarray(mean_v), axis=1)
    pred_v = sorted_mv[:, -1]
    margin = sorted_mv[:, -1] - sorted_mv[:, -2]

    per_layer = []
    for l in range(L):
        per_layer.append(dict(
            N=int(np.asarray(spk[l]).shape[1]),
            spk=np.asarray(spk[l]),    # (B, N) spikes summed over T
            plat=np.asarray(plat[l]),  # (B, N) plateau steps summed over T
            sip=np.asarray(sip[l]),    # (B, N) spikes-in-plateau summed over T
            edge=np.asarray(edge[l]),  # (B, N) plateau onsets summed over T
        ))
    return dict(acc=acc, preds=preds, pred_v=pred_v, margin=margin,
                per_layer=per_layer)


def summarize(model_name, red, T, bin_size_ms, tp_list):
    """Build a human-readable per-layer metrics dict for one model."""
    B = red["preds"].shape[0]
    out = dict(name=model_name, acc=red["acc"],
               pred_v_mean=float(np.mean(red["pred_v"])),
               margin_mean=float(np.mean(red["margin"])),
               layers=[])
    total_bin_s = bin_size_ms / 1000.0
    for l, pl in enumerate(red["per_layer"]):
        N = pl["N"]
        tot_spk = pl["spk"].sum()
        tot_plat = pl["plat"].sum()
        tot_sip = pl["sip"].sum()
        tot_edge = pl["edge"].sum()
        denom = B * T * N

        rate_per_step = tot_spk / denom
        rate_hz = rate_per_step / total_bin_s

        occ = tot_plat / denom
        mean_dur_steps = (tot_plat / tot_edge) if tot_edge > 0 else 0.0
        mean_dur_ms = mean_dur_steps * bin_size_ms

        p_spk_in = (tot_sip / tot_plat) if tot_plat > 0 else 0.0
        out_steps = denom - tot_plat
        p_spk_out = ((tot_spk - tot_sip) / out_steps) if out_steps > 0 else 0.0

        # Sparsity: per-neuron total spikes across the whole batch.
        neuron_tot = pl["spk"].sum(axis=0)  # (N,)
        dead_frac = float(np.mean(neuron_tot == 0))
        spikes_per_neuron_per_seq = pl["spk"].mean(axis=0)  # avg over batch, (N,)
        q = np.percentile(spikes_per_neuron_per_seq, [0, 50, 100])

        out["layers"].append(dict(
            N=N,
            rate_per_step=float(rate_per_step), rate_hz=float(rate_hz),
            occupancy=float(occ),
            mean_dur_steps=float(mean_dur_steps), mean_dur_ms=float(mean_dur_ms),
            p_spk_in=float(p_spk_in), p_spk_out=float(p_spk_out),
            dead_frac=dead_frac,
            spk_seq_min=float(q[0]), spk_seq_med=float(q[1]), spk_seq_max=float(q[2]),
            Tp_min=int(np.asarray(tp_list[l]).min()),
            Tp_max=int(np.asarray(tp_list[l]).max()),
        ))
    return out


# ══════════════════════════════════════════════════════════════════════
#  Reporting
# ══════════════════════════════════════════════════════════════════════

def _fmt_table(sa, sb, T, bin_size_ms):
    """Side-by-side text table for two summarized models."""
    a, b = sa["name"], sb["name"]
    lines = []
    w = 26
    def row(label, va, vb):
        lines.append(f"{label:<34}{va:>{w}}{vb:>{w}}")
    row("METRIC", a, b)
    lines.append("-" * (34 + 2 * w))
    row("test accuracy (this batch)", f"{sa['acc']:.2f}%", f"{sb['acc']:.2f}%")
    row("readout v_mean (pred class)", f"{sa['pred_v_mean']:.4f}", f"{sb['pred_v_mean']:.4f}")
    row("readout margin (top1-top2)", f"{sa['margin_mean']:.4f}", f"{sb['margin_mean']:.4f}")
    lines.append("")
    nl = len(sa["layers"])
    for l in range(nl):
        la, lb = sa["layers"][l], sb["layers"][l]
        lines.append(f"--- hidden layer {l} (N={la['N']}) ---")
        row("  firing rate (spk/neuron/step)", f"{la['rate_per_step']:.4f}", f"{lb['rate_per_step']:.4f}")
        row("  firing rate (Hz)", f"{la['rate_hz']:.1f}", f"{lb['rate_hz']:.1f}")
        row("  plateau occupancy", f"{la['occupancy']:.4f}", f"{lb['occupancy']:.4f}")
        row("  mean plateau dur (steps)", f"{la['mean_dur_steps']:.2f}", f"{lb['mean_dur_steps']:.2f}")
        row("  mean plateau dur (ms)", f"{la['mean_dur_ms']:.1f}", f"{lb['mean_dur_ms']:.1f}")
        row("  P(spike | plateau)", f"{la['p_spk_in']:.4f}", f"{lb['p_spk_in']:.4f}")
        row("  P(spike | no plateau)", f"{la['p_spk_out']:.4f}", f"{lb['p_spk_out']:.4f}")
        row("  plateau spike boost (x)",
            f"{(la['p_spk_in']/la['p_spk_out']):.2f}" if la['p_spk_out'] > 0 else "inf",
            f"{(lb['p_spk_in']/lb['p_spk_out']):.2f}" if lb['p_spk_out'] > 0 else "inf")
        row("  dead-neuron fraction", f"{la['dead_frac']:.3f}", f"{lb['dead_frac']:.3f}")
        row("  spikes/neuron/seq (med)", f"{la['spk_seq_med']:.2f}", f"{lb['spk_seq_med']:.2f}")
        row("  spikes/neuron/seq (max)", f"{la['spk_seq_max']:.1f}", f"{lb['spk_seq_max']:.1f}")
        row("  T_p range (steps, ref)", f"{la['Tp_min']}-{la['Tp_max']}", f"{lb['Tp_min']}-{lb['Tp_max']}")
        lines.append("")
    return "\n".join(lines)


def write_summary_md(path, sa, sb, T, bin_size_a, bin_size_b, meta_a, meta_b):
    with open(path, "w") as f:
        f.write("# BPTT vs BPTT_adapt — dynamics comparison\n\n")
        f.write(f"- **{sa['name']}**: {meta_a['n_inputs']} input ch, "
                f"v_th={meta_a['v_th']}, adaptation={'per-neuron trained' if meta_a['per_neuron_adapt'] else 'fixed scalar'}, "
                f"bin={bin_size_a} ms, T={T}\n")
        f.write(f"- **{sb['name']}**: {meta_b['n_inputs']} input ch, "
                f"v_th={meta_b['v_th']}, adaptation={'per-neuron trained' if meta_b['per_neuron_adapt'] else 'fixed scalar'}, "
                f"bin={bin_size_b} ms, T={T}\n\n")
        f.write("> Side-by-side characterization, not a controlled ablation: the two models "
                "differ in input width, v_th, and adaptation style.\n\n")
        f.write("```\n")
        f.write(_fmt_table(sa, sb, T, bin_size_a))
        f.write("\n```\n")


# ══════════════════════════════════════════════════════════════════════
#  Plotting
# ══════════════════════════════════════════════════════════════════════

def plot_traces(model_name, p, traces, neurons_per_layer, sample_label,
                out_dir, rng_seed=0):
    """One figure per hidden layer: `neurons_per_layer` random neurons, each a
    column with a somatic panel (top) and a dendritic panel (bottom)."""
    v_out, h_out, mu_out, o_out = traces  # lists of (T, N) per layer
    L = p["L"]
    v_th, gamma, mu_th = p["v_th"], p["gamma"], p["mu_th"]
    rng = np.random.default_rng(rng_seed)
    paths = []
    for l in range(L):
        V = np.asarray(v_out[l]); H = np.asarray(h_out[l])
        MU = np.asarray(mu_out[l]); O = np.asarray(o_out[l])
        T, N = V.shape
        k = min(neurons_per_layer, N)
        # Draw random neurons from those that actually spike on this sample, so
        # every panel shows real spike/threshold dynamics (excludes silent or
        # runaway-subthreshold neurons that would just blow up the y-axis).
        active = np.where(O.sum(axis=0) > 0)[0]
        pool = active if active.size >= k else np.arange(N)
        idx = np.sort(rng.choice(pool, k, replace=False))
        t = np.arange(T)

        fig, axes = plt.subplots(
            2, k, figsize=(3.2 * k, 5.0), sharex=True,
            gridspec_kw=dict(height_ratios=[1.0, 1.0]))
        if k == 1:
            axes = axes.reshape(2, 1)
        for j, ni in enumerate(idx):
            ax_s = axes[0, j]
            ax_d = axes[1, j]

            # ── Somatic panel: v_pre_reset + dynamic threshold + spikes ──
            vv = V[:, ni]
            ax_s.plot(t, vv, color="#1f77b4", lw=0.9, label="v (soma)")
            dyn_th = v_th - gamma * H[:, ni]
            ax_s.plot(t, dyn_th, color="#d62728", lw=0.9, ls="--",
                      label="dyn. threshold")
            # Robust y-limits so a large sub-threshold excursion can't hide the
            # spiking dynamics near threshold.
            lo, hi = np.percentile(vv, [1, 99])
            span = max(hi - lo, 1e-6)
            lo -= 0.12 * span
            hi = max(hi + 0.12 * span, v_th + 0.15 * span)
            ax_s.set_ylim(lo, hi)
            sp_t = t[O[:, ni] > 0.5]
            ax_s.plot(sp_t, np.full_like(sp_t, hi, dtype=float), "|",
                      color="k", markersize=8, markeredgewidth=1.0, label="spike")
            ax_s.set_title(f"neuron {ni}", fontsize=9)
            if j == 0:
                ax_s.set_ylabel("somatic V")
                ax_s.legend(fontsize=6, loc="upper right", framealpha=0.6)

            # ── Dendritic panel: mu + mu_th + plateau shading ──
            ax_d.plot(t, MU[:, ni], color="#2ca02c", lw=0.9, label="mu (dend)")
            ax_d.axhline(mu_th, color="#9467bd", lw=0.8, ls=":", label="mu_th")
            ymn = float(np.min(MU[:, ni])); ymx = float(np.max(MU[:, ni]))
            if ymx <= ymn:
                ymx = ymn + 1.0
            ax_d.fill_between(t, ymn, ymx, where=H[:, ni] > 0.5, color="orange",
                              alpha=0.25, step="mid", label="plateau (h=1)")
            ax_d.set_ylim(ymn, ymx)
            if j == 0:
                ax_d.set_ylabel("dendritic mu")
                ax_d.legend(fontsize=6, loc="upper right", framealpha=0.6)
            ax_d.set_xlabel("timestep")

        fig.suptitle(f"{model_name} — hidden layer {l} — "
                     f"sample (true class {sample_label})", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out = os.path.join(out_dir, f"{model_name}_layer{l}_traces.png")
        fig.savefig(out, dpi=130)
        plt.close(fig)
        paths.append(out)
    return paths


# ══════════════════════════════════════════════════════════════════════
#  Driver
# ══════════════════════════════════════════════════════════════════════

def run_model(path, batch, neurons, rng_seed, out_dir):
    p, meta, args = load_model(path)
    name = "BPTT_adapt" if p["per_neuron_adapt"] else "BPTT"
    print(f"[{name}] loading test set "
          f"(collapse_factor={args.get('collapse_factor')}, "
          f"n_inputs={p['n_inputs']}) ...", flush=True)
    X_te, y_te = load_test_set(args)
    bin_size_ms = float(args.get("bin_size_ms", 4.0))

    # The SHD test split is ordered (not class-balanced along the index), so
    # shuffle with a fixed seed before subsampling for representative stats.
    perm = np.random.default_rng(rng_seed).permutation(X_te.shape[0])
    X_te, y_te = X_te[perm], y_te[perm]

    B = min(batch, X_te.shape[0])
    Xb = jnp.asarray(X_te[:B], dtype=jnp.float64)
    yb = y_te[:B]
    T = int(Xb.shape[1])
    print(f"[{name}] batch={B}  T={T}  n_inputs={Xb.shape[2]}", flush=True)

    stats_batch, trace_one = make_runners(p)

    dend0, soma0 = layer0_drive(Xb, p)
    mean_v, spk, plat, sip, edge = stats_batch(dend0, soma0)

    red = reduce_stats(mean_v, spk, plat, sip, edge, yb)
    summ = summarize(name, red, T, bin_size_ms, p["T_p"])
    summ_meta = dict(n_inputs=p["n_inputs"], v_th=p["v_th"],
                     per_neuron_adapt=p["per_neuron_adapt"])

    # Pick a correctly-classified sample for the trace plots.
    correct = np.where(red["preds"] == yb)[0]
    si = int(correct[0]) if correct.size else 0
    d0s, s0s = layer0_drive(jnp.asarray(X_te[si:si+1], dtype=jnp.float64), p)
    traces = trace_one(d0s[0], s0s[0])
    fig_paths = plot_traces(name, p, traces, neurons, int(y_te[si]),
                            out_dir, rng_seed=rng_seed)

    print(f"[{name}] accuracy (batch of {B}) = {red['acc']:.2f}%  "
          f"(stored final_acc = {meta['extra'].get('final_acc'):.2f}%)", flush=True)
    for fp in fig_paths:
        print(f"[{name}] wrote {fp}", flush=True)
    return summ, summ_meta, T, bin_size_ms, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bptt", default=DEFAULT_BPTT, help="BPTT checkpoint .npz")
    ap.add_argument("--adapt", default=DEFAULT_ADAPT, help="BPTT_adapt checkpoint .npz")
    ap.add_argument("--batch", type=int, default=512,
                    help="Number of (shuffled) test samples for aggregate stats.")
    ap.add_argument("--neurons", type=int, default=5,
                    help="Neurons per layer in the trace plots.")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for the random neuron selection.")
    ap.add_argument("--out_dir", default=os.path.join(_THIS, "out"),
                    help="Directory for figures and summary.md.")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)

    sb_a, meta_a, T_a, bin_a, raw_a = run_model(
        a.bptt, a.batch, a.neurons, a.seed, a.out_dir)
    sb_b, meta_b, T_b, bin_b, raw_b = run_model(
        a.adapt, a.batch, a.neurons, a.seed, a.out_dir)

    table = _fmt_table(sb_a, sb_b, T_a, bin_a)
    print("\n" + "=" * 86)
    print(table)
    print("=" * 86)

    md_path = os.path.join(a.out_dir, "summary.md")
    write_summary_md(md_path, sb_a, sb_b, T_a, bin_a, bin_b, meta_a, meta_b)
    print(f"\nWrote {md_path}")


if __name__ == "__main__":
    main()
