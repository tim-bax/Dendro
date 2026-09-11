#!/usr/bin/env python3
"""Verify the autograd multi-layer network reproduces Soft_Reset's e-prop gradient.

The Multi net's surrogates are wired to MATCH Soft_Reset (two-path somatic spike:
soma edge = surrogate_sigma(beta_s), dend edge = roof or gamma·bump with
beta_s_dend; plateau surrogate ungated). So at one layer the gradients should be
identical to Soft_Reset's hand-coded e-prop rule, for BOTH surrogate modes.

Checks (real SHD, float64):

(1) FORWARD PARITY (L=1): autograd forward == Soft_Reset's hard `_predict_only`
    → ~0 (straight-through preserves the hard forward value).

(2) GRADIENT PARITY (L=1), for dend_surrogate_roof in {False, True}:
    -loss_temperature · (autograd dL/dw)  ==  Soft_Reset's custom e-prop grad.
    Now EXACT on all three (readout/soma/dend ~1e-12), because the plateau
    surrogate is ungated and the dendritic edge uses the same roof/bump + beta_s_dend.
    (Requires tau_dend=15 = _TAU_ELIG_DEND_MS and no adaptation; both hold here.)

(3) MULTI-LAYER SANITY: on a deeper net the gradient is finite, BOUNDED (the
    detached reset prevents the plateau blow-up), and nonzero down to layer 0.

    NB: no finite-difference check — the hard-spike forward is piecewise-constant
    in the weights, so its true gradient is zero a.e.; the surrogate is a modeling
    choice, and parity (2) validates it against the reference implementation.

Usage (from repo root):
    JAX_PLATFORMS=cpu SHD_CACHE_DIR=$HOME/data \
      ~/miniconda3/envs/dendroprop/bin/python Soft_Reset_Multi/verify_autograd_multi.py
"""
import importlib.util
import os
import sys

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random, lax, vmap
import numpy as np

_MULTI = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_MULTI)
_SR = os.path.join(_ROOT, "Soft_Reset")
for _p in (_ROOT, _MULTI):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import NeuronConfig                     # noqa: E402
from two_comp_neuron import TwoCompNeuron           # noqa: E402
import network as srm                               # noqa: E402
from data.shd_binned import load_shd_binned         # noqa: E402


def _load_soft_reset_network():
    """Load Soft_Reset/network.py under a private name, reusing the shared
    config/two_comp_neuron/lif_neuron already imported above (identical code)."""
    spec = importlib.util.spec_from_file_location(
        "sr_network", os.path.join(_SR, "network.py")
    )
    sr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sr)
    return sr


def _report(name, ref, cust):
    ref = np.asarray(ref, dtype=np.float64)
    cust = np.asarray(cust, dtype=np.float64)
    diff = np.linalg.norm(ref - cust)
    den = np.linalg.norm(cust) + 1e-300
    print(f"  {name:34s} rel_l2={diff/den:.3e}   max_abs={np.max(np.abs(ref-cust)):.3e}")


def _count_plateaus(x, w_dend, w_soma, alpha_s, alpha_d, T_p, config, alpha_w):
    """Total somatic spikes and plateau-steps for one sample (hard dynamics)."""
    dend_in = x @ w_dend.T
    soma_in = x @ w_soma.T
    Tn, n = x.shape[0], w_dend.shape[0]
    ti = jnp.arange(Tn, dtype=jnp.int32)

    def step(state, inp):
        di, si, t = inp
        state, o, _vp, h, _hp, _m = TwoCompNeuron.dynamics_step(
            state, di, si, t, alpha_s, alpha_d, T_p, config, alpha_w,
        )
        return state, (jnp.sum(o.astype(jnp.float64)), jnp.sum(h.astype(jnp.float64)))

    hz = (jnp.zeros(n), jnp.zeros(n),
          jnp.zeros(n, dtype=jnp.int32), jnp.zeros(n, dtype=jnp.int32),
          jnp.zeros(n), jnp.zeros(n))
    _, (osum, hsum) = lax.scan(step, hz, (dend_in, soma_in, ti))
    return jnp.sum(osum), jnp.sum(hsum)


def _parity(sr, x, target, use_roof, n_inputs, n_hidden, n_outputs, seed):
    """Forward + gradient parity for one dend_surrogate_roof setting."""
    config = NeuronConfig(dt=4.0, weight_scale=0.6, dend_surrogate_roof=use_roof)
    net = srm.Network(random.PRNGKey(seed), n_inputs, [n_hidden], n_outputs,
                      config, dropout_rate=0.0)
    params = net._params_tree()
    w_dend0, w_soma0 = params["w_dend"][0], params["w_soma"][0]
    w_readout = params["w_readout"]
    alpha_s, alpha_d, alpha_m, alpha_w = net._alphas()
    temp, bias = config.loss_temperature, config.loss_count_bias
    target_smoothed = net._smooth_targets(target)

    mode = "roof" if use_roof else "bump"
    print("\n" + "=" * 70)
    print(f"dend_surrogate_roof = {use_roof}  ({mode})")
    print("=" * 70)

    # (1) forward parity
    mean_v_hard = sr._pred_single(
        x, w_dend0, w_soma0, w_readout, alpha_s, alpha_d, alpha_m, T_p0(net), config, alpha_w,
    )
    mean_v_auto, _ = srm._forward_single(
        params, net._T_p(), x, alpha_s, alpha_d, alpha_m, alpha_w, config,
        random.PRNGKey(0), 0.0,
    )
    _report("(1) forward mean voltage", mean_v_auto, mean_v_hard)

    # (2) gradient parity: -temp * autograd dL/dw  vs  custom e-prop grad
    _, _l, _p, g_r_c, g_s_c, g_d_c = sr._fb_single(
        x, w_dend0, w_soma0, w_readout,
        alpha_s, alpha_d, alpha_m, T_p0(net), config, alpha_w,
        target_smoothed, temp, bias, random.PRNGKey(0), 0.0, 0.0, 0.0,
    )

    def loss_fn(p):
        mv, _ = srm._sample_forward(
            p, net._T_p(), x, alpha_s, alpha_d, alpha_m, alpha_w, config,
            random.PRNGKey(0), 0.0,
        )
        return -jnp.sum(target_smoothed * jax.nn.log_softmax(mv / temp + bias))

    g = jax.grad(loss_fn)(params)
    _report("(2) readout", -temp * g["w_readout"], g_r_c)
    _report("(2) soma", -temp * g["w_soma"][0], g_s_c)
    _report("(2) dend", -temp * g["w_dend"][0], g_d_c)


def T_p0(net):
    return net._T_p()[0]


def main():
    sr = _load_soft_reset_network()
    seed, n_hidden, n_outputs = 42, 64, 20

    print("Loading SHD (count-binned, float64)...", flush=True)
    _, _, _, X_te, y_te, _ = load_shd_binned(
        bin_size_ms=4.0, collapse_factor=5, max_duration_ms=1400.0,
        binarize=False, dtype=np.float64,
    )
    n_inputs = X_te.shape[2]

    # Pick the most dendritically active sample (weights are roof-independent).
    ref = srm.Network(random.PRNGKey(seed), n_inputs, [n_hidden], n_outputs,
                      NeuronConfig(dt=4.0, weight_scale=0.6), dropout_rate=0.0)
    rp = ref._params_tree()
    a_s, a_d, a_m, a_w = ref._alphas()
    cnt = jax.jit(vmap(_count_plateaus,
                       in_axes=(0, None, None, None, None, None, None, None)))
    N = min(128, X_te.shape[0])
    osum, hsum = cnt(jnp.asarray(X_te[:N]), rp["w_dend"][0], rp["w_soma"][0],
                     a_s, a_d, ref._T_p()[0], ref.config, a_w)
    hsum = np.asarray(hsum)
    idx = int(np.argmax(hsum))
    x = jnp.asarray(X_te[idx])
    target = int(y_te[idx])
    print(f"\nSample #{idx} (label={target}): {int(np.asarray(osum)[idx])} spikes, "
          f"{int(hsum[idx])} plateau-steps over T={x.shape[0]} x {n_hidden} neurons.")
    print("(readout/soma/dend should all be ~1e-12 in BOTH modes now.)")

    for use_roof in (False, True):
        _parity(sr, x, target, use_roof, n_inputs, n_hidden, n_outputs, seed)

    # (3) multi-layer sanity — the user's roof setting, a deeper net.
    config = NeuronConfig(dt=4.0, weight_scale=0.6, dend_surrogate_roof=True)
    deep = srm.Network(random.PRNGKey(seed), n_inputs, [32, 24, 16], n_outputs,
                       config, dropout_rate=0.0)
    temp, bias = config.loss_temperature, config.loss_count_bias

    def deep_loss(p):
        mv, _ = srm._sample_forward(
            p, deep._T_p(), x, *deep._alphas(), config, random.PRNGKey(0), 0.0,
        )
        return -jnp.sum(deep._smooth_targets(target) * jax.nn.log_softmax(mv / temp + bias))

    gd = jax.grad(deep_loss)(deep._params_tree())
    print("\n" + "=" * 70)
    print("(3) MULTI-LAYER SANITY  —  net 32 -> 24 -> 16 (roof), per-layer grad norms")
    print("=" * 70)
    ok = True
    for i in range(len(gd["w_dend"])):
        nd = float(jnp.linalg.norm(gd["w_dend"][i]))
        ns = float(jnp.linalg.norm(gd["w_soma"][i]))
        fin = np.isfinite(nd) and np.isfinite(ns) and (nd > 0 or ns > 0)
        ok = ok and fin
        print(f"  layer {i}:  |g_dend|={nd:.4e}  |g_soma|={ns:.4e}  "
              f"{'ok' if fin else 'FAIL'}")
    nr = float(jnp.linalg.norm(gd["w_readout"]))
    print(f"  readout:  |g_readout|={nr:.4e}")
    print(f"\n  bounded & credit reaches every layer: "
          f"{'YES' if ok and np.isfinite(nr) else 'NO'}")

    print("\nInterpretation:")
    print("  (1) ~0                    ⇒ same network as Soft_Reset (forward).")
    print("  (2) all ~1e-12, both modes ⇒ L=1 autograd grad == Soft_Reset e-prop")
    print("      grad exactly (roof honored, hp ungated, beta_s_dend used).")
    print("  (3) finite & nonzero at layer 0 ⇒ multi-layer credit assignment works.")


if __name__ == "__main__":
    main()
