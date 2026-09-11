#!/usr/bin/env python3
"""Membrane-potential diagnostics for the Soft_Reset two-compartment model.

Replays the forward pass over a batch of SHD test samples and reports the range
and overshoot of the hidden somatic (v) and dendritic (mu) potentials, plus the
pre-reset somatic voltage V_pre that actually drives spiking.

Answers:
  * how far above threshold do neurons overshoot at spike time (V_pre / v_th_eff)?
  * what are the min / max / percentiles of v and mu?
  * how large is the raw per-step input drive (soma_in, dend_in)?

Runs the SAME dynamics_step the trained model uses, so the numbers are exact
(soft reset included). Use a saved model to see the *trained* operating point;
a fresh net only shows the init.

Usage (in the dendroprop env, from repo root):
    python Soft_Reset/probe_membrane.py --model Soft_Reset/models/shd_seed42_XXXX.npz
    python Soft_Reset/probe_membrane.py            # fresh untrained net (sanity only)
"""
import argparse
import os
import sys

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax, vmap, random
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
for _p in (_ROOT, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data.shd_binned import load_shd_binned
from config import NeuronConfig
from network import Network
from two_comp_neuron import TwoCompNeuron


def _trace_forward(x_input, w_dend, w_soma, alpha_s, alpha_d, T_p, config, alpha_w):
    """One sample: per-step (V_pre, v, mu, o, h, soma_in, dend_in), each (T, n)."""
    dend_inputs = x_input @ w_dend.T
    soma_inputs = x_input @ w_soma.T
    T = x_input.shape[0]
    n = w_dend.shape[0]
    time_indices = jnp.arange(T, dtype=jnp.int32)

    def step(state, inp):
        dend_in, soma_in, t = inp
        new_state, o, v_pre, h, _h_prev, _mu_atp = TwoCompNeuron.dynamics_step(
            state, dend_in, soma_in, t, alpha_s, alpha_d, T_p, config, alpha_w,
        )
        mu, v, _h_new, _tp, _matp, _w = new_state
        return new_state, (v_pre, v, mu, o, h, soma_in, dend_in)

    h_zeros = (
        jnp.zeros(n), jnp.zeros(n),
        jnp.zeros(n, dtype=jnp.int32), jnp.zeros(n, dtype=jnp.int32),
        jnp.zeros(n), jnp.zeros(n),
    )
    _, traces = lax.scan(step, h_zeros, (dend_inputs, soma_inputs, time_indices))
    return traces


_AXES = (0, None, None, None, None, None, None, None)
_trace_batch = jax.jit(vmap(_trace_forward, in_axes=_AXES))


def _stats(name, a):
    a = np.asarray(a).ravel()
    ps = np.percentile(a, [0, 1, 50, 99, 100])
    print(f"  {name:8s} min={ps[0]:10.3f}  p1={ps[1]:9.3f}  median={ps[2]:8.3f}  "
          f"p99={ps[3]:9.3f}  max={ps[4]:10.3f}  |mean|={np.mean(np.abs(a)):.3f}")


def main():
    p = argparse.ArgumentParser(description="Membrane-potential probe for Soft_Reset.")
    p.add_argument("--model", type=str, default="", help="Path to a saved .npz; empty = fresh net.")
    p.add_argument("--n_samples", type=int, default=128)
    p.add_argument("--bin_size_ms", type=float, default=4.0)
    p.add_argument("--collapse_factor", type=int, default=5)
    p.add_argument("--max_duration_ms", type=float, default=1400.0)
    p.add_argument("--n_hidden", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    bin_size, collapse, maxdur = args.bin_size_ms, args.collapse_factor, args.max_duration_ms
    net = None
    if args.model:
        net = Network.load(args.model)
        margs = net.meta.get("extra", {}).get("args", {}) if hasattr(net, "meta") else {}
        bin_size = float(margs.get("bin_size_ms", bin_size))
        collapse = int(margs.get("collapse_factor", collapse))
        maxdur = float(margs.get("max_duration_ms", maxdur))
        print(f"Loaded {args.model}  (bin={bin_size} collapse={collapse} maxdur={maxdur})")

    _, _, _, X_te, y_te, _ = load_shd_binned(
        bin_size_ms=bin_size, collapse_factor=collapse, max_duration_ms=maxdur,
        binarize=False, dtype=np.float64,
    )
    n_inputs = X_te.shape[2]

    if net is None:
        config = NeuronConfig(dt=bin_size)
        net = Network(random.PRNGKey(args.seed), n_inputs, args.n_hidden, 20, config)
        print("Fresh (untrained) net — ranges reflect init only.")

    N = min(args.n_samples, X_te.shape[0])
    x_batch = jnp.asarray(X_te[:N])

    Vpre, v, mu, o, h, soma_in, dend_in = _trace_batch(
        x_batch, net.hidden.w_dend, net.hidden.w_soma,
        net.hidden.alpha_s, net.hidden.alpha_d, net.hidden.T_p,
        net.config, net.hidden.alpha_w,
    )

    v_th = float(net.config.v_th); gamma = float(net.config.gamma); mu_th = float(net.config.mu_th)
    o_np = np.asarray(o).astype(bool)
    h_np = np.asarray(h)
    Vpre_np = np.asarray(Vpre)
    v_th_eff = v_th - gamma * h_np                 # effective threshold at each step
    spikes = Vpre_np[o_np]                         # V_pre at spike times
    denom = v_th_eff[o_np]
    over_ratio = spikes / denom                    # how many thresholds V_pre reached
    over_excess = spikes - denom                   # absolute excess above threshold

    print(f"\nProbed {N} samples, T={x_batch.shape[1]}, n_hidden={net.n_hidden}  "
          f"(v_th={v_th}, gamma={gamma}, mu_th={mu_th})")
    print(f"firing rate (soma o) = {o_np.mean():.4f}   plateau rate (h) = {h_np.mean():.4f}\n")

    print("Hidden potentials over all (sample x time x neuron):")
    _stats("V_pre", Vpre)          # pre-reset somatic drive (triggers spikes)
    _stats("v", v)                 # post-(soft)-reset somatic potential
    _stats("mu", mu)               # dendritic potential
    _stats("soma_in", soma_in)     # per-step somatic input drive
    _stats("dend_in", dend_in)     # per-step dendritic input drive

    print("\nSomatic overshoot at spike times (V_pre / v_th_eff = thresholds reached):")
    if over_ratio.size:
        qs = np.percentile(over_ratio, [50, 90, 99, 100])
        print(f"  median={qs[0]:.2f}x  p90={qs[1]:.2f}x  p99={qs[2]:.2f}x  max={qs[3]:.2f}x")
        # Raw V_pre at spike time = overshoot vs the BASE threshold (v_th, ignoring the
        # gamma plateau reduction). Separates "high voltage" from "low plateau threshold".
        sq = np.percentile(spikes, [50, 90, 99, 100])
        print(f"  raw V_pre at spike (= x base v_th): median={sq[0]/v_th:.2f}  p90={sq[1]/v_th:.2f}  "
              f"p99={sq[2]/v_th:.2f}  max={sq[3]/v_th:.2f}")
        print(f"  ...of those spikes, fraction during a plateau (v_th_eff=v_th-gamma): "
              f"{float(h_np[o_np].mean()):.4f}")
        for k in (2, 5, 10, 20):
            print(f"  fraction of spikes overshooting > {k:2d}x threshold: "
                  f"{float(np.mean(over_ratio > k)):.4f}")
        print(f"  excess (V_pre - v_th_eff): median={np.median(over_excess):.3f}  "
              f"max={over_excess.max():.3f}")
    else:
        print("  (no spikes fired)")

    # ── Per-neuron concentration: is the tail a few units or the whole layer? ──
    n_hidden = Vpre_np.shape[2]
    vp_flat = Vpre_np.reshape(-1, n_hidden)          # (B*T, n)
    o_flat = o_np.reshape(-1, n_hidden)
    per_max = vp_flat.max(axis=0)                    # per-neuron max V_pre
    per_p99 = np.percentile(vp_flat, 99, axis=0)
    per_fire = o_flat.mean(axis=0)

    print("\nPer-neuron overshoot concentration (over hidden units):")
    print(f"  per-neuron max V_pre:  median={np.median(per_max):.2f}  "
          f"p90={np.percentile(per_max, 90):.2f}  max={per_max.max():.2f}")
    print(f"  per-neuron p99 V_pre:  median={np.median(per_p99):.2f}  "
          f"p90={np.percentile(per_p99, 90):.2f}  max={per_p99.max():.2f}")
    for thr in (10, 20, 50):
        cnt = int((per_max > thr).sum())
        print(f"  units with max V_pre > {thr:2d}: {cnt}/{n_hidden} ({100 * cnt / n_hidden:.1f}%)")

    # Concentration of the big-overshoot spikes (raw V_pre > 5): what share sits in
    # the top-10 units, and how few units account for 50% of them?
    big = (vp_flat > 5.0) & o_flat
    per_big = big.sum(axis=0)
    total_big = int(per_big.sum())
    if total_big > 0:
        order = np.argsort(per_big)[::-1]
        top10_frac = per_big[order[:10]].sum() / total_big
        cum = np.cumsum(per_big[order]) / total_big
        n50 = int(np.searchsorted(cum, 0.5) + 1)
        n_active = int((per_big > 0).sum())
        print(f"  spikes with raw V_pre>5: {total_big} total, from {n_active}/{n_hidden} units; "
              f"top-10 units hold {100 * top10_frac:.1f}%; 50% come from {n50} unit(s)")
        b0 = int(order[0])
        print(f"  busiest unit idx={b0}: {int(per_big[b0])} big spikes, "
              f"max V_pre={per_max[b0]:.1f}, firing={per_fire[b0]:.3f}")
    else:
        print("  (no spikes with raw V_pre > 5)")


if __name__ == "__main__":
    main()
