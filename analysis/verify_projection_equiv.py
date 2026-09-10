#!/usr/bin/env python3
"""Does the new projected two-pass e-prop equal the old single-pass (J,N,K) rule?

`Soft_Reset/network.py` was refactored to speed up training: instead of building
the error-independent sensitivity tensors A_soma / A_dend of shape
(J, N, K) = (n_outputs, n_hidden, n_inputs) and contracting the class dimension J
with the error only at the end, it now projects the error to the hidden layer
first (readout_seed = w_readout^T e) and accumulates only (N, K) gradients in a
second forward pass.

That transformation is a pure linear reassociation — the LI readout is linear, so
its per-output surrogate is identically 1, which makes the time-summed gradient
core independent of J. So every gradient should be the identical real number, up
to floating-point summation order (~1e-13 relative).

This script proves it numerically. It reproduces the ORIGINAL single-pass rule
verbatim in `_ref_forward_backward` and diffs it against the shipped
`network._forward_backward` (via `_fb_single` / `_fb_batch`), for the single and
batched paths, with dropout off and on (shared keys). Self-contained: synthetic
spiking input, no dataset needed.

Usage (dendroprop env, from repo root):
    python analysis/verify_projection_equiv.py
    python analysis/verify_projection_equiv.py --n_hidden 32 --weight_scale 2.5
"""
import argparse
import os
import sys

import jax
jax.config.update("jax_enable_x64", True)  # exact comparison needs float64
import jax.numpy as jnp
from jax import lax, jit, vmap, random
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
_SOFT = os.path.join(_ROOT, "Soft_Reset")
for _p in (_ROOT, _SOFT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import NeuronConfig, surrogate_sigma          # noqa: E402
from two_comp_neuron import TwoCompNeuron                  # noqa: E402
from lif_neuron import LINeuron                            # noqa: E402
from network import Network, _fb_single, _fb_batch, _FB_AXES  # noqa: E402


def _ref_forward_backward(
    x_input, w_dend, w_soma, w_readout,
    alpha_s, alpha_d, alpha_m, T_p, config, alpha_w,
    target_smoothed, loss_temperature, loss_count_bias,
    rng_key, dropout_rate,
):
    """The ORIGINAL single-pass (J,N,K) accumulator + loss/grads, reproduced
    verbatim. Same signature/return as the new `_forward_backward` so they diff
    directly. Do not 'optimize' this — it is the ground truth."""
    dend_inputs = x_input @ w_dend.T
    soma_inputs = x_input @ w_soma.T
    T = x_input.shape[0]
    n_hidden = w_dend.shape[0]
    n_inputs = w_dend.shape[1]
    n_outputs = w_readout.shape[0]
    time_indices = jnp.arange(T, dtype=jnp.int32)
    dropout_keys = random.split(rng_key, T)
    dropout_scale = 1.0 / (1.0 - dropout_rate)

    h_zeros = (
        jnp.zeros(n_hidden), jnp.zeros(n_hidden),
        jnp.zeros(n_hidden, dtype=jnp.int32), jnp.zeros(n_hidden, dtype=jnp.int32),
        jnp.zeros(n_hidden), jnp.zeros(n_inputs),
        jnp.zeros((n_hidden, n_inputs)), jnp.zeros((n_hidden, n_inputs)),
        jnp.zeros(n_hidden),
    )
    r_init = (jnp.zeros(n_outputs), jnp.zeros(n_outputs), jnp.zeros(n_hidden))
    A_d_init = jnp.zeros((n_outputs, n_hidden, n_inputs))
    F_d_init = jnp.zeros_like(A_d_init)

    def step(carry, inputs):
        h_carry, r_carry, A_d, F_d = carry
        dend_in, soma_in, x_t, t, drop_key = inputs

        h_carry, h_o, h_v_pre, h_h, h_h_prev, h_mu_at_tp = TwoCompNeuron.forward_step(
            h_carry, dend_in, soma_in, t, alpha_s, alpha_d, T_p, config, alpha_w,
        )
        hidden_o_float = h_o.astype(jnp.float64)
        mask = random.bernoulli(drop_key, 1.0 - dropout_rate, (n_hidden,)).astype(jnp.float64)
        hidden_o_float = hidden_o_float * mask * dropout_scale
        r_carry, r_v, r_E = LINeuron.forward_step(
            r_carry, hidden_o_float, w_readout, alpha_m,
        )

        mu_c, v_c, h_c, tp_c, matp_c, E_soma_c, dmu_c, dmu_atp_c, w_c = h_carry
        E_soma_new = TwoCompNeuron.update_somatic_eligibility(
            E_soma_c, x_t.astype(jnp.float64), alpha_s,
        )
        dmu_new, dmu_atp_new = TwoCompNeuron.update_dendritic_eligibility(
            dmu_c, dmu_atp_c, x_t.astype(jnp.float64), h_h_prev, alpha_d,
        )
        h_carry = (mu_c, v_c, h_c, tp_c, matp_c, E_soma_new, dmu_new, dmu_atp_new, w_c)

        sp_readout = jnp.ones(n_outputs)
        soma_dist = h_v_pre + config.gamma * h_h - config.v_th
        sp_hidden = surrogate_sigma(soma_dist, config.beta_s)
        sp_hidden_d = surrogate_sigma(soma_dist, config.beta_s_dend)
        hp_hidden = surrogate_sigma(h_mu_at_tp - config.mu_th, config.beta_d)

        eta = sp_readout[:, None] * w_readout * sp_hidden_d[None, :]
        eta_d = eta * (hp_hidden * config.gamma)[None, :]
        g_d = jnp.einsum("ji,ik->jik", eta_d, dmu_atp_new)
        F_d = alpha_m * F_d + g_d
        A_d = A_d + F_d

        return (h_carry, r_carry, A_d, F_d), (sp_readout, sp_hidden, r_E, E_soma_new)

    init_carry = (h_zeros, r_init, A_d_init, F_d_init)
    scan_inputs = (dend_inputs, soma_inputs, x_input, time_indices, dropout_keys)
    final_carry, per_step_all = lax.scan(step, init_carry, scan_inputs)
    sp_r, sp_h, E_r, E_s = per_step_all
    _, r_carry_f, A_d_f, _ = final_carry
    mean_voltage = r_carry_f[1] / T

    B = (1.0 - alpha_m ** (T - time_indices)) / (1.0 - alpha_m)
    A_readout = jnp.einsum("ti,tj->ij", sp_r, E_r)
    C_soma = jnp.einsum("tj,ti,tk->jik", sp_r, sp_h, E_s * B[:, None])
    A_soma = w_readout[:, :, None] * C_soma

    scaled_logits = mean_voltage / loss_temperature + loss_count_bias
    probs = jnp.exp(scaled_logits - jnp.max(scaled_logits))
    probs = probs / jnp.sum(probs)
    prediction = jnp.argmax(mean_voltage)
    loss = -jnp.sum(target_smoothed * jnp.log(probs + 1e-8))
    global_error = target_smoothed - probs

    grad_readout = (global_error[:, None] * A_readout) / T
    grad_soma = jnp.einsum("j,jik->ik", global_error, A_soma) / T
    grad_dend = jnp.einsum("j,jik->ik", global_error, A_d_f) / T
    return mean_voltage, loss, prediction, grad_readout, grad_soma, grad_dend


_ref_single = jit(_ref_forward_backward)
_ref_batch = jit(vmap(_ref_forward_backward, in_axes=_FB_AXES))


def _rel_l2(new, ref):
    new = np.asarray(new, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    denom = np.linalg.norm(ref.ravel())
    if denom == 0.0:
        return float(np.linalg.norm((new - ref).ravel()))
    return float(np.linalg.norm((new - ref).ravel()) / denom)


def _activity(net, x, cfg):
    """Somatic spikes + plateau-steps for one sample (to confirm non-vacuous)."""
    w_dend, w_soma, _ = net._weights()
    alpha_s, alpha_d, alpha_m, T_p, config, alpha_w = net._params()
    di = x @ w_dend.T
    si = x @ w_soma.T
    T = x.shape[0]
    n = w_dend.shape[0]
    ti = jnp.arange(T, dtype=jnp.int32)
    hz = (jnp.zeros(n), jnp.zeros(n), jnp.zeros(n, jnp.int32), jnp.zeros(n, jnp.int32),
          jnp.zeros(n), jnp.zeros(n))

    def step(c, inp):
        hs, os_, hs_sum = c
        d, s, t = inp
        hs, o, _, h, *_ = TwoCompNeuron.dynamics_step(
            hs, d, s, t, alpha_s, alpha_d, T_p, config, alpha_w)
        return (hs, os_ + jnp.sum(o), hs_sum + jnp.sum(h)), None
    (_, o_sum, h_sum), _ = lax.scan(step, (hz, 0.0, 0.0), (di, si, ti))
    return int(o_sum), int(h_sum)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_inputs", type=int, default=60)
    ap.add_argument("--n_hidden", type=int, default=24)
    ap.add_argument("--n_outputs", type=int, default=20)
    ap.add_argument("--T", type=int, default=50)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--rate", type=float, default=0.2, help="input spike prob per (t,k)")
    ap.add_argument("--weight_scale", type=float, default=2.5)
    ap.add_argument("--tol", type=float, default=1e-11, help="rel_l2 pass threshold")
    args = ap.parse_args()

    cfg = NeuronConfig(dt=4.0, weight_scale=args.weight_scale)
    net = Network(random.PRNGKey(args.seed), args.n_inputs, args.n_hidden,
                  args.n_outputs, cfg)
    weights = net._weights()
    params = net._params()

    kx = random.PRNGKey(args.seed + 100)
    x1 = (random.uniform(kx, (args.T, args.n_inputs)) < args.rate).astype(jnp.float64)
    kb = random.split(random.PRNGKey(args.seed + 200), args.batch)
    xb = jnp.stack([
        (random.uniform(k, (args.T, args.n_inputs)) < args.rate).astype(jnp.float64)
        for k in kb
    ])
    targets = jnp.arange(args.batch) % args.n_outputs
    ts1 = net._smooth_targets(int(targets[0]))
    tsb = net._smooth_targets(targets)
    tmp, bias = cfg.loss_temperature, cfg.loss_count_bias

    o_sp, h_sp = _activity(net, x1, cfg)
    print(f"single sample: {o_sp} somatic spikes, {h_sp} plateau-steps "
          f"over T={args.T} x {args.n_hidden} neurons "
          f"({'OK — all grad paths exercised' if o_sp and h_sp else 'WARNING: a path is vacuous'})")

    names = ["mean_voltage", "loss", "grad_readout", "grad_soma", "grad_dend"]
    all_ok = True

    for label, dp, key in [("single, dropout=0.0", 0.0, random.PRNGKey(1)),
                           ("single, dropout=0.3", 0.3, random.PRNGKey(1))]:
        new = _fb_single(x1, *weights, *params, ts1, tmp, bias, key, dp)
        ref = _ref_single(x1, *weights, *params, ts1, tmp, bias, key, dp)
        mv_n, ls_n, pr_n, gr_n, gs_n, gd_n = new
        mv_r, ls_r, pr_r, gr_r, gs_r, gd_r = ref
        diffs = [_rel_l2(mv_n, mv_r), abs(float(ls_n) - float(ls_r)),
                 _rel_l2(gr_n, gr_r), _rel_l2(gs_n, gs_r), _rel_l2(gd_n, gd_r)]
        pred_ok = int(pr_n) == int(pr_r)
        ok = all(d < args.tol for d in diffs) and pred_ok
        all_ok &= ok
        print(f"\n[{label}]  pred new={int(pr_n)} ref={int(pr_r)} "
              f"({'match' if pred_ok else 'MISMATCH'})")
        for nm, d in zip(names, diffs):
            print(f"    {nm:14s} rel_l2/abs = {d:.3e}")
        print(f"    => {'PASS' if ok else 'FAIL'}")

    # Batched path (per-sample dropout keys, shared between new and ref).
    for label, dp in [("batch, dropout=0.0", 0.0), ("batch, dropout=0.3", 0.3)]:
        bkeys = random.split(random.PRNGKey(2), args.batch)
        new = _fb_batch(xb, *weights, *params, tsb, tmp, bias, bkeys, dp)
        ref = _ref_batch(xb, *weights, *params, tsb, tmp, bias, bkeys, dp)
        mv_n, ls_n, pr_n, gr_n, gs_n, gd_n = new
        mv_r, ls_r, pr_r, gr_r, gs_r, gd_r = ref
        # Average grads over batch, as batch_train_step does.
        diffs = [_rel_l2(mv_n, mv_r), _rel_l2(ls_n, ls_r),
                 _rel_l2(jnp.mean(gr_n, 0), jnp.mean(gr_r, 0)),
                 _rel_l2(jnp.mean(gs_n, 0), jnp.mean(gs_r, 0)),
                 _rel_l2(jnp.mean(gd_n, 0), jnp.mean(gd_r, 0))]
        pred_ok = bool(jnp.all(pr_n == pr_r))
        ok = all(d < args.tol for d in diffs) and pred_ok
        all_ok &= ok
        print(f"\n[{label}]  preds {'match' if pred_ok else 'MISMATCH'}")
        for nm, d in zip(names, diffs):
            print(f"    {nm:14s} rel_l2 = {d:.3e}")
        print(f"    => {'PASS' if ok else 'FAIL'}")

    print("\n" + "=" * 60)
    print(f"OVERALL: {'PASS — new == old to float precision' if all_ok else 'FAIL'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
