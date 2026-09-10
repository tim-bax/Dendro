#!/usr/bin/env python3
"""Does Soft_Reset's custom e-prop gradient equal autograd with a detached reset?

This script answers that question numerically, to floating-point precision, on a
real SHD sample. It builds a *fully differentiable replica* of the Soft_Reset
forward pass and compares `jax`'s reverse-mode gradient of it against the
hand-rolled e-prop gradients from `network._forward_backward`.

The differentiable replica is constructed to match the hand-coded gradient's
assumptions exactly:

  * somatic spike and dendritic plateau use `config.surrogate_sigma` as their VJP
    (straight-through: forward value is the hard 0/1, backward is the surrogate);
  * the somatic soft reset `o * v_th_eff` is wrapped in `stop_gradient`
    (this is the "detached reset");
  * integer state that acts as a gate — the plateau flag `h_prev` that gates
    dendritic integration and the `mu_at_tprime` latch, and the plateau
    duration window — is carried as a hard int, so no gradient flows through it,
    exactly as the eligibility recursions treat it.

Two readings of the dendritic path are tested, differing only in whether the
plateau surrogate `hp` is gated by the duration window:

    hp UNMASKED  -> what network.py's g_dend actually computes (hp has no
                    duration_mask; see _forward_and_accum, hp_hidden line).
    hp MASKED    -> the mathematically faithful autograd of
                    h = duration_mask * H(mu_atp - mu_th).

Expected result (see the printed verdict):
    readout, soma                     : match to ~1e-12  (exact)
    dend, hp UNMASKED                 : match to ~1e-12  (custom == detached autograd)
    dend, hp MASKED                   : small nonzero     (the one genuine approx:
                                        e-prop keeps hp at plateau-offset steps
                                        where the faithful gradient is gated off)

Also shown: the full loss-gradient obeys  g_custom = -loss_temperature * dL/dw,
i.e. the hand-coded update drops the 1/temperature factor (folds into the lr).

Usage (dendroprop env, from repo root):
    python analysis/verify_detached_reset_grad.py
    python analysis/verify_detached_reset_grad.py --model Soft_Reset/models/shd_seedXX.npz
"""
import argparse
import os
import sys
from functools import partial

import jax
jax.config.update("jax_enable_x64", True)  # exact comparison needs float64
import jax.numpy as jnp
from jax import lax, vmap, random
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
_SOFT = os.path.join(_ROOT, "Soft_Reset")
for _p in (_ROOT, _SOFT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data.shd_binned import load_shd_binned          # noqa: E402
from config import NeuronConfig, surrogate_sigma      # noqa: E402
from two_comp_neuron import TwoCompNeuron              # noqa: E402
from network import Network, _fb_single  # noqa: E402


# ── Surrogate spike: forward = Heaviside, backward = config.surrogate_sigma ──

@partial(jax.custom_vjp, nondiff_argnums=(1,))
def sg_spike(x, beta):
    """H(x) with a surrogate derivative surrogate_sigma(x, beta) w.r.t. x."""
    return (x >= 0).astype(x.dtype)


def _sg_fwd(x, beta):
    return sg_spike(x, beta), x


def _sg_bwd(beta, x, g):
    return (g * surrogate_sigma(x, beta),)


sg_spike.defvjp(_sg_fwd, _sg_bwd)


def _straight_through(hard, soft):
    """Value of `hard`, gradient of `soft`."""
    return jax.lax.stop_gradient(hard - soft) + soft


# ── Differentiable replica of the forward pass → readout mean voltage ──

def diff_mean_voltage(
    w_dend, w_soma, w_readout, x_input,
    alpha_s, alpha_d, alpha_m, alpha_w, T_p, config,
    plateau_mask: bool,
):
    """Mean readout voltage (J,) as a differentiable function of the weights.

    Same numbers as network._predict_only / _forward_and_accum (dropout off),
    but wired so reverse-mode autodiff reproduces the *intended* gradient:
    surrogate spikes, detached soft reset, hard integer gates.

    plateau_mask: if True, the plateau surrogate is gated by the duration
    window (faithful autograd of h = mask * H(mu_atp - mu_th)); if False, the
    surrogate is ungated, matching network.py's hp_hidden.
    """
    dend_inputs = x_input @ w_dend.T   # (T, n)
    soma_inputs = x_input @ w_soma.T   # (T, n)
    T = x_input.shape[0]
    n = w_dend.shape[0]
    J = w_readout.shape[0]
    time_indices = jnp.arange(T, dtype=jnp.int32)

    vth, gamma, mu_th = config.v_th, config.gamma, config.mu_th
    beta_s, beta_d = config.beta_s, config.beta_d
    a_adapt, b_adapt = config.a_adapt, config.b_adapt

    def step(carry, inp):
        mu_p, v_p, h_p, tprime_p, matp_p, w_p, rv, rsum = carry
        dend_in, soma_in, t = inp
        f = mu_p.dtype

        # ── dendrite (h_p, tprime_p are hard ints → gates carry no gradient) ──
        tprime = jnp.where(t == 0, 0, jnp.where(h_p == 1, tprime_p, t))
        mu = jnp.where(t > 0, alpha_d * mu_p + (1 - h_p) * dend_in, dend_in)
        matp = jnp.where(h_p == 0, mu, matp_p)               # latch (differentiable)
        plateau_dur = t - tprime
        dur_mask = ((plateau_dur <= T_p) & (plateau_dur >= 0)).astype(f)

        h_hard = jnp.where(
            (matp >= mu_th) & (plateau_dur <= T_p) & (plateau_dur >= 0), 1, 0
        ).astype(jnp.int32)
        h_grad = sg_spike(matp - mu_th, beta_d)              # d/dmatp = surrogate
        if plateau_mask:
            h_grad = h_grad * dur_mask
        h_soft = _straight_through(h_hard.astype(f), h_grad)  # value=h_hard

        # ── soma ──
        v_th_eff = vth - gamma * h_soft
        v_pre = jnp.where(t > 0, alpha_s * v_p + soma_in - w_p, soma_in)
        o_hard = (v_pre >= (vth - gamma * h_hard.astype(f))).astype(jnp.int32)
        o_grad = sg_spike(v_pre - v_th_eff, beta_s)          # grad → v_pre and h_soft
        o_soft = _straight_through(o_hard.astype(f), o_grad)

        # Soft reset, detached: gradient does NOT flow back through the reset.
        v = v_pre - jax.lax.stop_gradient(o_soft * v_th_eff)
        w = alpha_w * w_p + (1 - alpha_w) * a_adapt * v_pre + b_adapt * o_soft

        # ── LI readout (linear) ──
        rv = alpha_m * rv + o_soft @ w_readout.T
        rsum = rsum + rv

        return (mu, v, h_hard, tprime, matp, w, rv, rsum), None

    init = (
        jnp.zeros(n), jnp.zeros(n),
        jnp.zeros(n, dtype=jnp.int32), jnp.zeros(n, dtype=jnp.int32),
        jnp.zeros(n), jnp.zeros(n),
        jnp.zeros(J), jnp.zeros(J),
    )
    final_carry, _ = lax.scan(step, init, (dend_inputs, soma_inputs, time_indices))
    rsum = final_carry[-1]   # readout voltage sum over time
    return rsum / T


# ── Plateau/spike counter, to pick a sample that actually exercises the dendrite ──

def _count(x, w_dend, w_soma, alpha_s, alpha_d, T_p, config, alpha_w):
    dend_in = x @ w_dend.T
    soma_in = x @ w_soma.T
    Tn = x.shape[0]
    n = w_dend.shape[0]
    ti = jnp.arange(Tn, dtype=jnp.int32)

    def step(state, inp):
        di, si, t = inp
        state, o, _vp, h, _hp, _matp = TwoCompNeuron.dynamics_step(
            state, di, si, t, alpha_s, alpha_d, T_p, config, alpha_w,
        )
        return state, (jnp.sum(o.astype(jnp.float64)), jnp.sum(h.astype(jnp.float64)))

    hz = (
        jnp.zeros(n), jnp.zeros(n),
        jnp.zeros(n, dtype=jnp.int32), jnp.zeros(n, dtype=jnp.int32),
        jnp.zeros(n), jnp.zeros(n),
    )
    _, (osum, hsum) = lax.scan(step, hz, (dend_in, soma_in, ti))
    return jnp.sum(osum), jnp.sum(hsum)


_count_batch = jax.jit(vmap(_count, in_axes=(0, None, None, None, None, None, None, None)))


def _report(name, ref, cust):
    ref = np.asarray(ref, dtype=np.float64)
    cust = np.asarray(cust, dtype=np.float64)
    diff = np.linalg.norm(ref - cust)
    den = np.linalg.norm(cust) + 1e-300
    print(f"  {name:46s} rel_l2={diff / den:.3e}   max_abs={np.max(np.abs(ref - cust)):.3e}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=str, default="", help="Saved .npz; empty = fresh net.")
    p.add_argument("--bin_size_ms", type=float, default=4.0)
    p.add_argument("--collapse_factor", type=int, default=5)
    p.add_argument("--max_duration_ms", type=float, default=1400.0)
    p.add_argument("--n_hidden", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scan_samples", type=int, default=128,
                   help="How many test samples to scan when picking the most active one.")
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
        # Weight scale bumped so a fresh net actually fires plateaus (else the
        # dendritic path is all-zero and the comparison is vacuous there).
        config = NeuronConfig(dt=bin_size, weight_scale=0.6)
        net = Network(random.PRNGKey(args.seed), n_inputs, args.n_hidden, 20, config,
                      dropout_rate=0.0)
        print(f"Fresh net (seed={args.seed}, weight_scale={config.weight_scale}).")

    w_dend, w_soma, w_readout = net._weights()
    alpha_s, alpha_d, alpha_m, T_p, config, alpha_w = net._params()

    # Pick the most dendritically active sample so the dend comparison is real.
    N = min(args.scan_samples, X_te.shape[0])
    xs = jnp.asarray(X_te[:N])
    osum, hsum = _count_batch(xs, w_dend, w_soma, alpha_s, alpha_d, T_p, config, alpha_w)
    hsum = np.asarray(hsum)
    idx = int(np.argmax(hsum))
    x = jnp.asarray(X_te[idx])
    target = int(y_te[idx])
    T = x.shape[0]
    print(f"\nUsing test sample #{idx} (label={target}): "
          f"{int(np.asarray(osum)[idx])} somatic spikes, "
          f"{int(hsum[idx])} plateau-steps over T={T} x {net.n_hidden} neurons.")
    if hsum[idx] == 0:
        print("  WARNING: no plateaus fire — the dendritic gradient is identically "
              "zero here, so its comparison is vacuous. Try --model or raise "
              "weight_scale / lower mu_th.")

    # ── Hand-coded (e-prop) gradients from the training path (dropout off) ──
    target_smoothed = net._smooth_targets(target)
    mean_v_c, loss, pred, g_r_c, g_s_c, g_d_c = _fb_single(
        x, w_dend, w_soma, w_readout,
        alpha_s, alpha_d, alpha_m, T_p, config, alpha_w,
        target_smoothed, config.loss_temperature, config.loss_count_bias,
        random.PRNGKey(0), 0.0,
    )

    # global_error is the cotangent the hand-coded path contracts A_* with.
    scaled = mean_v_c / config.loss_temperature + config.loss_count_bias
    probs = jax.nn.softmax(scaled)
    global_error = target_smoothed - probs

    # ── Reference: reverse-mode autodiff of the detached-reset replica ──
    def mv(wd, ws, wr, mask):
        return diff_mean_voltage(
            wd, ws, wr, x, alpha_s, alpha_d, alpha_m, alpha_w, T_p, config, mask,
        )

    # Sanity: the replica reproduces the forward mean voltage exactly.
    mean_v_ref = mv(w_dend, w_soma, w_readout, True)
    fwd_match = float(np.max(np.abs(np.asarray(mean_v_ref) - np.asarray(mean_v_c))))
    print(f"\nForward mean-voltage replica vs training path: max_abs={fwd_match:.3e} "
          f"(should be ~0)")

    # vjp contracted with global_error == Σ_j ge[j] ∂mean_v[j]/∂w == the custom grads.
    _, vjp_masked = jax.vjp(lambda wd, ws, wr: mv(wd, ws, wr, True),
                            w_dend, w_soma, w_readout)
    _, vjp_unmask = jax.vjp(lambda wd, ws, wr: mv(wd, ws, wr, False),
                            w_dend, w_soma, w_readout)
    gd_m, gs_m, gr_m = vjp_masked(global_error)
    gd_u, gs_u, gr_u = vjp_unmask(global_error)

    print("\n" + "=" * 74)
    print("JACOBIAN CHECK  —  custom e-prop grad   vs   detached-reset autograd")
    print("  (both = Σ_j global_error[j] · ∂mean_voltage[j]/∂w ; identical ⇒ rel_l2≈1e-12)")
    print("=" * 74)
    _report("readout", gr_u, g_r_c)
    _report("soma", gs_u, g_s_c)
    _report("dend  [autograd hp UNMASKED  = network.py's formula]", gd_u, g_d_c)
    _report("dend  [autograd hp MASKED by duration window]", gd_m, g_d_c)

    print("\n" + "=" * 74)
    print("FULL LOSS-GRAD RELATIONSHIP  —  custom g   vs   -temperature · dL/dw")
    print("  (the hand-coded update uses global_error directly, dropping 1/temp)")
    print("=" * 74)

    def loss_fn(wd, ws, wr, mask):
        mvv = mv(wd, ws, wr, mask)
        sc = mvv / config.loss_temperature + config.loss_count_bias
        return -jnp.sum(target_smoothed * jax.nn.log_softmax(sc))

    gL = jax.grad(loss_fn, argnums=(0, 1, 2))(w_dend, w_soma, w_readout, False)
    temp = config.loss_temperature
    _report("readout: -temp·dL/dw  vs  custom g", -temp * gL[2], g_r_c)
    _report("soma:    -temp·dL/dw  vs  custom g", -temp * gL[1], g_s_c)
    _report("dend:    -temp·dL/dw  vs  custom g (hp UNMASKED)", -temp * gL[0], g_d_c)

    print("\nInterpretation:")
    print("  • readout & soma matching ~1e-12  ⇒ for those paths the custom gradient")
    print("    IS reverse-mode autograd with a detached reset (exact, not approximate).")
    print("  • dend matches under 'hp UNMASKED' but not 'hp MASKED'  ⇒ the one genuine")
    print("    approximation: network.py applies the plateau surrogate at plateau-offset")
    print("    steps where the faithful detached gradient is gated to zero.")
    print("  • loss-grad block matching ⇒ custom g = -temperature · dL/dw (const rescale).")


if __name__ == "__main__":
    main()
