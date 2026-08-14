#!/usr/bin/env python3
"""Compare the BACKWARD pass of BPTT_fix (hand-derived, pruned, e-prop) vs
BPTT_fix_auto (exact -grad(loss) via jax.grad) on a small, controlled network
with an ARBITRARY number of hidden layers (default two).

Both folders share byte-identical config.py / two_comp_neuron.py / lif_neuron.py,
so the FORWARD pass is identical — only network.py differs. We therefore build ONE
set of weights and hand it to both rules, feed a burst input strong enough to drive
the dendrite past mu_th and latch a plateau (h=1), and lay the gradients side by
side per hidden layer.

THREE sources of difference, decomposed so each can be seen in isolation:

  #1 INTER-LAYER dendrite edge (only exists with >=2 hidden layers). Exact BPTT
     credits o^{l-1} back into layer l through BOTH the same-time soma edge AND the
     dendrite edge (o^{l-1}[t'] -> W_d -> mu -> h -> o^l). The hand rule PRUNES the
     dendrite edge from the inter-layer Jacobian: error flows layer->layer only
     through W_soma^T. To isolate JUST this effect we add a third gradient source,
     `auto_prune`: exact autodiff, but with the inter-layer dendrite input
     stop_gradient'd (dend_in_l = stop_gradient(presyn) @ W_d^l.T for l>=1). That
     keeps W_d^l's own gradient while blocking error transport to o^{l-1} through
     the dendrite — structurally identical to the hand rule's prune, with the exact
     within-layer time adjoint otherwise intact.

  #2 WITHIN-LAYER time credit. Exact BPTT backprops through the alpha_s/alpha_d
     recurrences, the v-reset and adaptation; the hand rule replaces that with
     forward eligibility traces (e-prop). This is what makes the exact gradient
     explode through the plateau while the hand rule stays bounded.

  #3 RESCALINGS. The hand rule drops the softmax 1/loss_temperature factor
     everywhere and divides the SOMA gradient by _SOMA_GRAD_SCALE(=8). Undoing
     those (the "norm" gradients) isolates the purely STRUCTURAL differences.

The clean decomposition, reported per layer:
  auto_full vs auto_prune ->  difference #1 alone (the pruned inter-layer dendrite).
  auto_prune vs fix_norm  ->  difference #2 alone (e-prop time truncation).
  auto_full vs fix        ->  the total.
plus, per inter-layer boundary, the EXACT adjoint ratio ||dendrite route|| /
||soma route|| = the relative magnitude of what the hand rule discards.

KEY RESULT (explosion panel): exact surrogate-BPTT runs the surrogate slope back
through the whole plateau/soma recurrence, whose Jacobian has spectral radius > 1,
so the hidden-layer gradient GROWS EXPONENTIALLY with plateau length. `auto_prune`
still explodes (the prune is not what bounds it); only the e-prop hand rule stays
bounded (~O(1)) at every sequence length.
"""
import os
import sys
import argparse
import importlib.util

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)

import jax
jax.config.update("jax_enable_x64", True)  # forward casts to float64 internally
import jax.numpy as jnp
from jax import random, lax

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════════
#  Load both network.py modules under distinct names. Their internal
#  `from config import ...` resolve against sys.path; the three shared modules
#  are byte-identical so a single cached copy serves both.
# ══════════════════════════════════════════════════════════════════════

def load_network_module(folder, alias):
    if folder not in sys.path:
        sys.path.insert(0, folder)
    spec = importlib.util.spec_from_file_location(alias, os.path.join(folder, "network.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


def make_input(T, burst, amp, n_in):
    """(T, n_in) count-like input. All channels fire during `burst`, then the
    network goes silent so the latched plateau is clearly visible. Channels are
    scaled differently so the weight columns get distinct drive. The first three
    scales reproduce the historical single-layer diagnostic."""
    base = np.array([1.0, 0.6, 1.3])
    scales = np.array([base[c % 3] * (1.0 + 0.15 * (c // 3)) for c in range(n_in)])
    x = np.zeros((T, n_in), dtype=np.float64)
    t0, t1 = burst[0], min(burst[1], T)
    x[t0:t1, :] = amp * scales[None, :]
    return jnp.asarray(x)


# ══════════════════════════════════════════════════════════════════════
#  Forward trace (identical for both rules) — mu, v, h, o per layer + readout.
# ══════════════════════════════════════════════════════════════════════

def _zero_h(n, k):
    return (
        jnp.zeros(n), jnp.zeros(n),
        jnp.zeros(n, dtype=jnp.int32), jnp.zeros(n, dtype=jnp.int32),
        jnp.zeros(n), jnp.zeros(k),
        jnp.zeros((n, k)), jnp.zeros((n, k)), jnp.zeros(n),
    )


def forward_trace(mod, x_input, weights, params):
    """Returns per-layer lists (mu, v_pre, h, o), each element (T, N_l), plus the
    readout trace v_r (T, J) and the mean voltage (J,)."""
    TwoComp, LI = mod.TwoCompNeuron, mod.LINeuron
    alpha_s, alpha_d, alpha_m, tp_list, config, alpha_w = params
    w_dend, w_soma, w_readout = weights["dend"], weights["soma"], weights["readout"]
    L = len(w_dend)
    T, J = x_input.shape[0], w_readout.shape[0]
    tidx = jnp.arange(T, dtype=jnp.int32)

    h0 = [_zero_h(w_dend[l].shape[0], w_dend[l].shape[1]) for l in range(L)]
    r0 = (jnp.zeros(J), jnp.zeros(J), jnp.zeros(w_dend[-1].shape[0]))

    def step(carry, xs):
        hs, rc = carry
        x_t, t = xs
        x_t = x_t.astype(jnp.float64)
        new_hs = list(hs)
        mus, vps, hsp, os = [], [], [], []
        o_prev = None
        for l in range(L):
            presyn = x_t if l == 0 else o_prev
            din = presyn @ w_dend[l].T
            sin = presyn @ w_soma[l].T
            new_hc, o, v_pre, h, _hp, _matp = TwoComp.forward_step(
                hs[l], din, sin, t, alpha_s, alpha_d, tp_list[l], config, alpha_w)
            new_hs[l] = new_hc
            of = o.astype(jnp.float64)
            mus.append(new_hc[0]); vps.append(v_pre)
            hsp.append(h.astype(jnp.float64)); os.append(of)
            o_prev = of
        rc, v_r, _ = LI.forward_step(rc, o_prev, w_readout, alpha_m)
        return (new_hs, rc), (mus, vps, hsp, os, v_r)

    _, (mu, v_pre, h, o, v_r) = lax.scan(step, (h0, r0), (x_input, tidx))
    mean_v = jnp.sum(v_r, axis=0) / T
    to_np = lambda lst: [np.asarray(a) for a in lst]
    return (to_np(mu), to_np(v_pre), to_np(h), to_np(o),
            np.asarray(v_r), np.asarray(mean_v))


# ══════════════════════════════════════════════════════════════════════
#  Gradient sources.
#
#  fix       — the hand-derived, pruned, e-prop rule (BPTT_fix).
#  auto_full — exact reverse-mode surrogate BPTT (BPTT_fix_auto), nothing pruned.
#  auto_prune— exact autodiff with the INTER-LAYER dendrite edge stop_gradient'd,
#              so it prunes exactly what the hand rule prunes (#1) but keeps the
#              exact within-layer time adjoint. Isolates difference #1.
# ══════════════════════════════════════════════════════════════════════

def rule_grads(mod, net, x_input, target, temp, count_bias):
    """Call a rule's compiled forward_backward. Returns per-layer grad lists."""
    hc, rc = net._h_carry(), net._r_carry()
    gs0, gd0, Er0 = net._acc_zeros()
    tgt = net._smooth_targets(target)
    common = (x_input, net._weights(), *net._params(), hc, rc, gs0, gd0, Er0,
              tgt, temp, count_bias, random.PRNGKey(0), 0.0)
    mv, loss, pred, gr, gs_list, gd_list = mod._fb_single(*common)
    return (np.asarray(gr), [np.asarray(g) for g in gs_list],
            [np.asarray(g) for g in gd_list], np.asarray(mv), float(loss))


def _make_loss(mod_auto, prune):
    """Build a differentiable loss(weights, ...) mirroring BPTT_fix_auto's
    _loss_and_meanv at dropout=0, reusing its _tc_step/_init_tc_state. When
    `prune`, the inter-layer dendrite input (l>=1) is stop_gradient'd so error
    cannot flow to o^{l-1} through the dendrite — the hand rule's structural prune,
    with the exact within-layer time adjoint left intact."""
    _tc_step = mod_auto._tc_step
    _init = mod_auto._init_tc_state

    def loss_fn(weights, x_input, alpha_s, alpha_d, alpha_m, tp_list, config,
                alpha_w, target_smoothed, temp, bias):
        w_dend, w_soma, w_readout = weights["dend"], weights["soma"], weights["readout"]
        L = len(w_dend)
        T = x_input.shape[0]
        tidx = jnp.arange(T, dtype=jnp.int32)
        h0 = [_init(w_dend[l].shape[0]) for l in range(L)]
        n_out = w_readout.shape[0]

        def step(carry, xs):
            hs, r_v, r_sum = carry
            x_t, t = xs
            x_t = x_t.astype(jnp.float64)
            new_hs = list(hs)
            o_prev = None
            for l in range(L):
                presyn = x_t if l == 0 else o_prev
                pre_d = jax.lax.stop_gradient(presyn) if (prune and l >= 1) else presyn
                dend_in_l = pre_d @ w_dend[l].T
                soma_in_l = presyn @ w_soma[l].T
                new_hs[l], o_l = _tc_step(
                    hs[l], dend_in_l, soma_in_l, t,
                    alpha_s, alpha_d, tp_list[l], config, alpha_w)
                o_prev = o_l
            r_v = alpha_m * r_v + o_prev @ w_readout.T
            r_sum = r_sum + r_v
            return (new_hs, r_v, r_sum), None

        (_, _, r_sum_f), _ = lax.scan(
            step, (h0, jnp.zeros(n_out), jnp.zeros(n_out)), (x_input, tidx))
        mean_v = r_sum_f / T
        logits = mean_v / temp + bias
        p = jnp.exp(logits - jnp.max(logits))
        p = p / jnp.sum(p)
        loss = -jnp.sum(target_smoothed * jnp.log(p + 1e-8))
        return loss, mean_v

    return loss_fn


def auto_grads(mod_auto, net, x_input, target, temp, count_bias, prune):
    """Exact ascent gradient g = -dL/dw via autodiff; `prune` toggles the
    inter-layer dendrite stop_gradient. Returns per-layer grad lists."""
    loss_fn = _make_loss(mod_auto, prune)
    tgt = net._smooth_targets(target)
    (loss, mv), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        net._weights(), x_input, *net._params(), tgt, temp, count_bias)
    gr = -np.asarray(grads["readout"])
    gs = [-np.asarray(g) for g in grads["soma"]]
    gd = [-np.asarray(g) for g in grads["dend"]]
    return gr, gs, gd, np.asarray(mv), float(loss)


def _make_boundary_loss(mod_auto, weights, params, b, freeze_dend_first):
    """loss(O) for a partial forward that starts by feeding an INJECTED spike
    sequence O (T, N_{b-1}) into layer b and runs layers b..L-1 + readout. With
    all-zero initial states (matching the full forward), layers b..L reproduce
    their spikes exactly, so dL/dO is the EXACT adjoint at that inter-layer
    boundary. `freeze_dend_first` stop_gradients the dendrite input of layer b
    only, isolating O's soma-route contribution; full - soma = dendrite route."""
    _tc_step = mod_auto._tc_step
    _init = mod_auto._init_tc_state
    w_dend, w_soma, w_readout = weights["dend"], weights["soma"], weights["readout"]
    alpha_s, alpha_d, alpha_m, tp_list, config, alpha_w = params
    L = len(w_dend)

    def loss_fn(O, target_smoothed, temp, bias):
        T = O.shape[0]
        tidx = jnp.arange(T, dtype=jnp.int32)
        h0 = [_init(w_dend[l].shape[0]) for l in range(b, L)]
        n_out = w_readout.shape[0]

        def step(carry, xs):
            hs, r_v, r_sum = carry
            o_in, t = xs
            new_hs = list(hs)
            presyn = o_in
            for i, l in enumerate(range(b, L)):
                if l == b and freeze_dend_first:
                    pre_d = jax.lax.stop_gradient(presyn)
                else:
                    pre_d = presyn
                dend_in_l = pre_d @ w_dend[l].T
                soma_in_l = presyn @ w_soma[l].T
                new_hs[i], o_l = _tc_step(
                    hs[i], dend_in_l, soma_in_l, t,
                    alpha_s, alpha_d, tp_list[l], config, alpha_w)
                presyn = o_l
            r_v = alpha_m * r_v + presyn @ w_readout.T
            r_sum = r_sum + r_v
            return (new_hs, r_v, r_sum), None

        (_, _, r_sum_f), _ = lax.scan(
            step, (h0, jnp.zeros(n_out), jnp.zeros(n_out)), (O, tidx))
        mean_v = r_sum_f / T
        logits = mean_v / temp + bias
        p = jnp.exp(logits - jnp.max(logits))
        p = p / jnp.sum(p)
        return -jnp.sum(target_smoothed * jnp.log(p + 1e-8))

    return loss_fn


def boundary_adjoint_ratio(mod_auto, net, o_layers, target, temp, count_bias, b):
    """Exact ||dendrite route|| / ||soma route|| for the inter-layer boundary into
    layer b (b>=1), i.e. the relative magnitude of the edge the hand rule prunes."""
    weights, params = net._weights(), net._params()
    O = jnp.asarray(o_layers[b - 1])
    tgt = net._smooth_targets(target)
    g_full = jax.grad(_make_boundary_loss(mod_auto, weights, params, b, False))(
        O, tgt, temp, count_bias)
    g_soma = jax.grad(_make_boundary_loss(mod_auto, weights, params, b, True))(
        O, tgt, temp, count_bias)
    g_full, g_soma = np.asarray(g_full), np.asarray(g_soma)
    g_dend = g_full - g_soma
    ns, nd = np.linalg.norm(g_soma), np.linalg.norm(g_dend)
    return (nd / ns) if ns > 0 else float("nan"), ns, nd


# ══════════════════════════════════════════════════════════════════════
#  Plotting
# ══════════════════════════════════════════════════════════════════════

def plot_forward(x_input, mu, v, h, o, v_r, mean_v, probs, target, cfg, out_path,
                 layer_idx, max_neurons=4):
    """Forward dynamics for one hidden layer (first `max_neurons` neurons)."""
    T, H = v.shape
    H = min(H, max_neurons)
    t = np.arange(T)
    v_th, gamma, mu_th = cfg["v_th"], cfg["gamma"], cfg["mu_th"]
    fig = plt.figure(figsize=(3.4 * H, 8.6))
    gs = fig.add_gridspec(3, H, height_ratios=[1.0, 1.0, 0.9], hspace=0.38, wspace=0.3)
    for n in range(H):
        ax_s = fig.add_subplot(gs[0, n])
        ax_s.plot(t, v[:, n], color="#1f77b4", lw=1.0, label="v (soma)")
        ax_s.plot(t, v_th - gamma * h[:, n], color="#d62728", lw=1.0, ls="--", label="threshold")
        sp = t[o[:, n] > 0.5]
        ytop = max(np.max(v[:, n]) * 1.05, v_th * 1.2)
        ax_s.plot(sp, np.full_like(sp, ytop, dtype=float), "|", color="k", ms=9, mew=1.1, label="spike o")
        ax_s.set_title(f"L{layer_idx} neuron {n}", fontsize=10)
        if n == 0:
            ax_s.set_ylabel("soma  v"); ax_s.legend(fontsize=6, loc="upper right", framealpha=0.6)
        ax_d = fig.add_subplot(gs[1, n])
        ax_d.plot(t, mu[:, n], color="#2ca02c", lw=1.0, label="mu (dend)")
        ax_d.axhline(mu_th, color="#9467bd", lw=0.9, ls=":", label="mu_th")
        ymn, ymx = float(np.min(mu[:, n])), float(np.max(mu[:, n]))
        if ymx <= ymn:
            ymx = ymn + 1.0
        ax_d.fill_between(t, ymn, ymx, where=h[:, n] > 0.5, color="orange", alpha=0.25,
                          step="mid", label="plateau h=1")
        ax_d.set_ylim(ymn, ymx)
        if n == 0:
            ax_d.set_ylabel("dend  mu"); ax_d.legend(fontsize=6, loc="upper right", framealpha=0.6)
        ax_d.set_xlabel("timestep")
    ax_in = fig.add_subplot(gs[2, : max(1, H // 2)])
    for c in range(min(x_input.shape[1], 6)):
        ax_in.plot(t, np.asarray(x_input)[:, c], lw=1.0, label=f"in ch {c}")
    ax_in.set_title("input channels", fontsize=10)
    ax_in.set_xlabel("timestep"); ax_in.set_ylabel("amplitude")
    ax_in.legend(fontsize=6, loc="upper right", framealpha=0.6)
    ax_r = fig.add_subplot(gs[2, max(1, H // 2):])
    for j in range(v_r.shape[1]):
        ax_r.plot(t, v_r[:, j], lw=1.1,
                  label=f"readout {j}" + ("  (target)" if j == target else ""))
    ax_r.set_title(f"readout v | mean_v={np.round(mean_v,3)} probs={np.round(probs,3)}", fontsize=9)
    ax_r.set_xlabel("timestep"); ax_r.set_ylabel("readout v")
    ax_r.legend(fontsize=6, loc="upper right", framealpha=0.6)
    fig.suptitle(f"Forward dynamics — hidden layer {layer_idx} "
                 "(identical for BPTT_fix and BPTT_fix_auto)", fontsize=12)
    fig.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close(fig)


def _labels(shape, sym):
    r, c = shape
    return [f"{sym}[{i},{j}]" for i in range(r) for j in range(c)]


def _panel3(ax, labels, ga, gb, gc, title, na, nb, nc, symlog=True):
    x = np.arange(len(labels)); w = 0.27
    ax.bar(x - w, ga, w, color="#ff7f0e", label=na)
    ax.bar(x,     gb, w, color="#2ca02c", label=nb)
    ax.bar(x + w, gc, w, color="#1f77b4", label=nc)
    ax.axhline(0, color="k", lw=0.6)
    if symlog:
        ax.set_yscale("symlog", linthresh=1e-2)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_title(title, fontsize=10); ax.legend(fontsize=7, framealpha=0.7)


def plot_backward_layer(l, gs_fn, gd_fn, gs_pr, gd_pr, gs_fu, gd_fu, T, out_path):
    ls, ld = _labels(gs_fu.shape, "s"), _labels(gd_fu.shape, "d")
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    _panel3(ax[0], ld, gd_fn.ravel(), gd_pr.ravel(), gd_fu.ravel(),
            f"L{l}  dL/dW_dend", "fix (norm)", "auto_prune", "auto_full")
    _panel3(ax[1], ls, gs_fn.ravel(), gs_pr.ravel(), gs_fu.ravel(),
            f"L{l}  dL/dW_soma", "fix (norm)", "auto_prune", "auto_full")
    fig.suptitle(f"Backward — hidden layer {l} (T={T}, symlog y). "
                 "fix(norm)≈auto_prune ⇒ the gap to auto_full is the pruned "
                 "inter-layer dendrite edge (#1)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(out_path, dpi=130); plt.close(fig)


def plot_readout(gr_fn, gr_fu, out_path):
    lr = _labels(gr_fu.shape, "r")
    x = np.arange(len(lr)); w = 0.4
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - w / 2, gr_fn.ravel(), w, color="#ff7f0e", label="fix (norm)")
    ax.bar(x + w / 2, gr_fu.ravel(), w, color="#1f77b4", label="auto_full")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(lr, rotation=90, fontsize=6)
    ax.set_title("dL/dW_readout — exact in both (no hidden recurrence); overlays", fontsize=10)
    ax.legend(fontsize=8, framealpha=0.7)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


def plot_explosion(Ts, mag, clip, out_path):
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    styles = {
        "fix_soma":   ("#ff7f0e", "-",  "o", "fix  max|dL/dW_soma|"),
        "fix_dend":   ("#ffbb78", "--", "s", "fix  max|dL/dW_dend|"),
        "prune_soma": ("#2ca02c", "-",  "o", "auto_prune  max|dL/dW_soma|"),
        "prune_dend": ("#98df8a", "--", "s", "auto_prune  max|dL/dW_dend|"),
        "full_soma":  ("#1f77b4", "-",  "o", "auto_full  max|dL/dW_soma|"),
        "full_dend":  ("#aec7e8", "--", "s", "auto_full  max|dL/dW_dend|"),
    }
    for k, (c, l, m, lab) in styles.items():
        ax.plot(Ts, mag[k], color=c, ls=l, marker=m, ms=4, lw=1.6, label=lab)
    ax.axhline(clip, color="k", ls=":", lw=1.0, label=f"clip_value={clip:g}")
    ax.set_yscale("log"); ax.set_xlabel("sequence length T  (longer = longer plateau)")
    ax.set_ylabel("max |dL/dW| over all hidden layers  (log)")
    ax.set_title("Hidden-layer gradient magnitude vs plateau length\n"
                 "exact (full AND pruned) explode through the plateau; "
                 "e-prop hand rule stays bounded", fontsize=11)
    ax.legend(fontsize=7.5, framealpha=0.8, ncol=1); ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
#  Numeric report
# ══════════════════════════════════════════════════════════════════════

def cos(a, b):
    a, b = a.ravel(), b.ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def rel(a, b):
    """||a - b|| / ||b||."""
    nb = np.linalg.norm(b.ravel())
    return float(np.linalg.norm((a - b).ravel()) / nb) if nb > 0 else float("nan")


def med_ratio(a, b):
    a, b = a.ravel(), b.ravel(); m = np.abs(b) > 1e-12
    return float(np.median(a[m] / b[m])) if m.any() else float("nan")


def report(G, temp, soma_scale, T, header, Ts, mag, bratios, L, growth):
    gr_fix, gs_fix, gd_fix = G["fix"]
    gr_fn, gs_fn, gd_fn = G["fix_norm"]
    gr_fu, gs_fu, gd_fu = G["full"]
    gr_pr, gs_pr, gd_pr = G["prune"]

    lines = ["# Backward-pass comparison: BPTT_fix vs BPTT_fix_auto", "", header, ""]
    lines += [
        "Three gradient sources:",
        "  fix        — hand-derived, pruned, e-prop rule (BPTT_fix, as returned)",
        "  auto_full  — exact reverse-mode surrogate BPTT (nothing pruned)",
        "  auto_prune — exact autodiff, inter-layer dendrite edge stop_gradient'd",
        "               (prunes exactly #1; keeps the exact within-layer time adjoint)",
        "  fix_norm   — fix with its rescalings undone: readout,dend x temp; soma x temp/scale",
        f"               (temp={temp:g}, _SOMA_GRAD_SCALE={soma_scale:g})",
        "",
        "Decomposition:  auto_full↔auto_prune = #1 inter-layer dendrite prune;",
        "                auto_prune↔fix_norm  = #2 e-prop time truncation;  fix↔auto_full = total.",
        "",
        "## Readout gradient (exact in all; sanity check)",
        f"  cos(fix_norm, auto_full) = {cos(gr_fn, gr_fu):+.6f}   (expect ~1.0)",
        f"  med ratio fix/auto_full  = {med_ratio(gr_fix, gr_fu):+.4f}   (expect temp={temp:g})",
        "",
        f"## Per-hidden-layer gradients at T={T}",
    ]
    for l in range(L):
        lines += [
            f"\n### hidden layer {l}",
            f"  ||W_soma grad||:  fix_norm={np.linalg.norm(gs_fn[l]):.4e}  "
            f"prune={np.linalg.norm(gs_pr[l]):.4e}  full={np.linalg.norm(gs_fu[l]):.4e}",
            f"  ||W_dend grad||:  fix_norm={np.linalg.norm(gd_fn[l]):.4e}  "
            f"prune={np.linalg.norm(gd_pr[l]):.4e}  full={np.linalg.norm(gd_fu[l]):.4e}",
            "  --- #1 inter-layer dendrite prune (auto_full vs auto_prune) ---",
            f"    soma  ||full-prune||/||full|| = {rel(gs_fu[l], gs_pr[l]):.4f}   "
            f"cos = {cos(gs_fu[l], gs_pr[l]):+.4f}",
            f"    dend  ||full-prune||/||full|| = {rel(gd_fu[l], gd_pr[l]):.4f}   "
            f"cos = {cos(gd_fu[l], gd_pr[l]):+.4f}",
            "  --- #2 e-prop time truncation (auto_prune vs fix_norm) ---",
            f"    soma  cos = {cos(gs_pr[l], gs_fn[l]):+.4f}   "
            f"med ratio prune/fix_norm = {med_ratio(gs_pr[l], gs_fn[l]):+.3f}",
            f"    dend  cos = {cos(gd_pr[l], gd_fn[l]):+.4f}   "
            f"med ratio prune/fix_norm = {med_ratio(gd_pr[l], gd_fn[l]):+.3f}",
            "  --- total (fix_norm vs auto_full) ---",
            f"    soma  cos = {cos(gs_fn[l], gs_fu[l]):+.4f}    dend  cos = {cos(gd_fn[l], gd_fu[l]):+.4f}",
        ]
        if l >= 1 and (l in bratios):
            r, ns, nd = bratios[l]
            lines.append(
                f"  --- exact inter-layer adjoint into L{l}: "
                f"||dendrite route|| / ||soma route|| = {r:.4f}  "
                f"(soma={ns:.3e}, dend={nd:.3e}) ---")

    lines += ["", "## Gradient magnitude vs sequence length (the key difference)",
              f"{'T':>6}{'fix':>14}{'auto_prune':>14}{'auto_full':>14}{'full/fix':>12}"]
    for i, Tv in enumerate(Ts):
        r = mag["full_soma"][i] / mag["fix_soma"][i] if mag["fix_soma"][i] > 0 else float("nan")
        lines.append(f"{Tv:>6}{mag['fix_soma'][i]:>14.3e}{mag['prune_soma'][i]:>14.3e}"
                     f"{mag['full_soma'][i]:>14.3e}{r:>12.2e}")
    lines.append(f"\nper-step growth factor (max|dW_soma|):  auto_full ~ {growth['full']:.3f}/step,  "
                 f"auto_prune ~ {growth['prune']:.3f}/step,  fix ~ {growth['fix']:.3f}/step "
                 "(>1 ⇒ exponential blow-up; e-prop hand rule ≈ flat).")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
#  Driver
# ══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hidden", type=int, nargs="+", default=[6, 6],
                    help="hidden layer sizes, e.g. --hidden 6 6 for two layers")
    ap.add_argument("--inputs", type=int, default=3, help="number of input channels")
    ap.add_argument("--outputs", type=int, default=2)
    ap.add_argument("--target", type=int, default=0)
    ap.add_argument("--seed", type=int, default=16,
                    help="weight-init seed (default chosen so BOTH layers spike + plateau)")
    ap.add_argument("--T", type=int, default=60, help="primary timesteps (forward + per-weight grads)")
    ap.add_argument("--amp", type=float, default=4.0, help="burst amplitude (counts)")
    ap.add_argument("--burst", type=int, nargs=2, default=(5, 45))
    ap.add_argument("--sweep", type=int, nargs="+", default=[20, 30, 40, 60, 80, 120, 175],
                    help="sequence lengths for the explosion diagnostic")
    ap.add_argument("--clip", type=float, default=1.0, help="reference clip_value line")
    ap.add_argument("--out_dir", default=os.path.join(_THIS, "out"))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    L = len(a.hidden)

    mod_fix = load_network_module(os.path.join(_ROOT, "BPTT_fix"), "net_fix")
    mod_auto = load_network_module(os.path.join(_ROOT, "BPTT_fix_auto"), "net_auto")
    config = mod_fix.NeuronConfig()  # default: a_adapt=b_adapt=0 (no adaptation recurrence)
    temp = float(config.loss_temperature)
    cbias = float(config.loss_count_bias)
    soma_scale = float(mod_fix._SOMA_GRAD_SCALE)

    def build_net(mod):
        return mod.Network(random.PRNGKey(a.seed), a.inputs, list(a.hidden), a.outputs,
                           config, dropout_rate=0.0)

    net_fix, net_auto = build_net(mod_fix), build_net(mod_auto)
    weights, params = net_fix._weights(), net_fix._params()

    # ── forward (primary T) ──
    x = make_input(a.T, tuple(a.burst), a.amp, a.inputs)
    mu, v, h, o, v_r, mean_v = forward_trace(mod_fix, x, weights, params)
    logit = mean_v / temp + cbias
    p = np.exp(logit - logit.max()); probs = p / p.sum()
    plat_desc, spk_desc = [], []
    for l in range(L):
        plat = np.where(h[l].sum(0) > 0)[0]
        spk = np.where(o[l].sum(0) > 0)[0]
        plat_desc.append(f"L{l}: plateau neurons {plat.tolist()} "
                         f"(steps {h[l].sum(0).astype(int).tolist()})")
        spk_desc.append(f"L{l}: spiking neurons {spk.tolist()} "
                        f"(spikes {o[l].sum(0).astype(int).tolist()})")
    header = (f"- arch: {a.inputs} inputs -> {' -> '.join(str(n) for n in a.hidden)} "
              f"two-comp hidden -> {a.outputs} readout, target class {a.target}, "
              f"burst amp {a.amp} on t={a.burst[0]}..{a.burst[1]}\n- "
              + "\n- ".join(plat_desc) + "\n- " + "\n- ".join(spk_desc))
    print(header)
    if all(h[l].sum() == 0 for l in range(L)):
        print("!! NO PLATEAU anywhere — raise --amp / widen --burst.")

    cfgp = dict(v_th=float(config.v_th), gamma=float(config.gamma), mu_th=float(config.mu_th))
    for l in range(L):
        fwd = os.path.join(a.out_dir, f"backward_forward_L{l}.png")
        plot_forward(x, mu[l], v[l], h[l], o[l], v_r, mean_v, probs, a.target, cfgp, fwd, l)
        print("wrote", fwd)

    # ── backward (primary T): three gradient sources ──
    gr_fix, gs_fix, gd_fix, mv_fix, loss_fix = rule_grads(mod_fix, net_fix, x, a.target, temp, cbias)
    gr_fu, gs_fu, gd_fu, mv_fu, loss_fu = auto_grads(mod_auto, net_auto, x, a.target, temp, cbias, prune=False)
    gr_pr, gs_pr, gd_pr, mv_pr, loss_pr = auto_grads(mod_auto, net_auto, x, a.target, temp, cbias, prune=True)
    print(f"forward agree: max|mean_v_fix - mean_v_auto| = {np.max(np.abs(mv_fix - mv_fu)):.2e} "
          f"(loss_fix={loss_fix:.5f}, loss_auto_full={loss_fu:.5f}, loss_auto_prune={loss_pr:.5f})")

    # cross-check: the compiled BPTT_fix_auto rule agrees with our local auto_full
    gr_ref, gs_ref, gd_ref, _, _ = rule_grads(mod_auto, net_auto, x, a.target, temp, cbias)
    ref_err = max([np.max(np.abs(gr_ref - gr_fu))]
                  + [np.max(np.abs(gs_ref[l] - gs_fu[l])) for l in range(L)]
                  + [np.max(np.abs(gd_ref[l] - gd_fu[l])) for l in range(L)])
    print(f"local auto_full vs BPTT_fix_auto._fb_single: max abs diff = {ref_err:.2e} (expect ~0)")

    # fix_norm: undo the hand rule's rescalings to expose the structural difference
    gr_fn = gr_fix / temp
    gs_fn = [g * soma_scale / temp for g in gs_fix]
    gd_fn = [g / temp for g in gd_fix]

    G = {"fix": (gr_fix, gs_fix, gd_fix), "fix_norm": (gr_fn, gs_fn, gd_fn),
         "full": (gr_fu, gs_fu, gd_fu), "prune": (gr_pr, gs_pr, gd_pr)}

    plot_readout(gr_fn, gr_fu, os.path.join(a.out_dir, "backward_grad_readout.png"))
    for l in range(L):
        out = os.path.join(a.out_dir, f"backward_grad_L{l}.png")
        plot_backward_layer(l, gs_fn[l], gd_fn[l], gs_pr[l], gd_pr[l], gs_fu[l], gd_fu[l], a.T, out)
        print("wrote", out)

    # ── exact inter-layer adjoint ratios (magnitude of the pruned dendrite edge) ──
    bratios = {}
    for l in range(1, L):
        bratios[l] = boundary_adjoint_ratio(mod_auto, net_auto, o, a.target, temp, cbias, l)

    # ── explosion sweep (fixed burst, growing plateau tail) ──
    Ts = sorted(set(a.sweep))
    keys = ("fix_soma", "fix_dend", "prune_soma", "prune_dend", "full_soma", "full_dend")
    mag = {k: [] for k in keys}
    for Tv in Ts:
        xv = make_input(Tv, tuple(a.burst), a.amp, a.inputs)
        _, gsf, gdf, _, _ = rule_grads(mod_fix, net_fix, xv, a.target, temp, cbias)
        _, gsu, gdu, _, _ = auto_grads(mod_auto, net_auto, xv, a.target, temp, cbias, prune=False)
        _, gsp, gdp, _, _ = auto_grads(mod_auto, net_auto, xv, a.target, temp, cbias, prune=True)
        mmax = lambda lst: max(float(np.abs(g).max()) for g in lst)
        mag["fix_soma"].append(mmax(gsf)); mag["fix_dend"].append(mmax(gdf))
        mag["prune_soma"].append(mmax(gsp)); mag["prune_dend"].append(mmax(gdp))
        mag["full_soma"].append(mmax(gsu)); mag["full_dend"].append(mmax(gdu))
    expl = os.path.join(a.out_dir, "backward_explosion.png")
    plot_explosion(Ts, mag, a.clip, expl)
    print("wrote", expl)

    def growth_of(key):
        if len(Ts) < 2 or mag[key][0] <= 0 or mag[key][-1] <= 0:
            return float("nan")
        return float(np.exp((np.log(mag[key][-1]) - np.log(mag[key][0])) / (Ts[-1] - Ts[0])))
    growth = {"full": growth_of("full_soma"), "prune": growth_of("prune_soma"),
              "fix": growth_of("fix_soma")}

    txt = report(G, temp, soma_scale, a.T, header, Ts, mag, bratios, L, growth)
    md = os.path.join(a.out_dir, "backward_summary.md")
    with open(md, "w") as f:
        f.write(txt + "\n")
    print("\n" + "=" * 80 + "\n" + txt + "\n" + "=" * 80 + "\nwrote " + md)


if __name__ == "__main__":
    main()
