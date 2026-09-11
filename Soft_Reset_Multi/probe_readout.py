#!/usr/bin/env python3
"""Readout-logit diagnostics for the Soft_Reset model — for choosing loss_temperature.

The readout produces a per-class mean voltage (network._predict_only). The loss
turns it into logits via

    scaled_logits = mean_voltage / loss_temperature + loss_count_bias      (network.py:178)

and a softmax. Because softmax is shift-invariant, only the *gaps between
classes* matter, divided by loss_temperature. This script replays the forward
pass on SHD test samples and reports:

  * the absolute spread of mean_voltage, and the correct-vs-rest gaps
    (what temperature actually trades against);
  * a "matched" temperature implied by your label smoothing — the temperature at
    which the softmax fixed point equals the smoothed target, so the gradient
    (target - probs) neither saturates to 0 nor stays diffuse;
  * a sweep: at each candidate temperature, the mean cross-entropy and the mean
    confidence (correct-class prob), so you can see saturation directly.

Note: argmax accuracy does NOT depend on temperature (it rescales logits only),
so accuracy is constant across the sweep — temperature is a calibration / gradient
-scale knob, not a prediction knob.

Point --model at a *saved* net for the trained operating point; an early-epoch
checkpoint shows the early operating point (the gap grows over training, so both
matter). A fresh net shows init only.

Usage (in the dendroprop env, from repo root):
    python Soft_Reset/probe_readout.py --model Soft_Reset/models/shd_seed12_20260908_123728.npz
"""
import argparse
import os
import sys

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
for _p in (_ROOT, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data.shd_binned import load_shd_binned
from config import NeuronConfig
from network import Network, _pred_batch


def _pct(name, a, unit=""):
    a = np.asarray(a).ravel()
    ps = np.percentile(a, [0, 1, 25, 50, 75, 99, 100])
    print(f"  {name:22s} min={ps[0]:8.3f}  p1={ps[1]:8.3f}  p25={ps[2]:8.3f}  "
          f"median={ps[3]:8.3f}  p75={ps[4]:8.3f}  p99={ps[5]:8.3f}  max={ps[6]:8.3f}{unit}")


def _softmax_rows(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def main():
    p = argparse.ArgumentParser(description="Readout-logit probe for choosing loss_temperature.")
    p.add_argument("--model", type=str, default="", help="Path to a saved .npz; empty = fresh net.")
    p.add_argument("--n_samples", type=int, default=256)
    p.add_argument("--bin_size_ms", type=float, default=4.0)
    p.add_argument("--collapse_factor", type=int, default=5)
    p.add_argument("--max_duration_ms", type=float, default=1400.0)
    p.add_argument("--n_hidden", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temps", type=float, nargs="*", default=None,
                   help="Custom temperature grid for the sweep (default: a spread around the model's).")
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
    y = np.asarray(y_te[:N]).astype(int)

    # Raw per-class readout mean voltage — the pre-temperature "output" (N, C).
    mean_voltage = _pred_batch(x_batch, *net._weights(), *net._params())
    mv = np.asarray(mean_voltage)
    C = mv.shape[1]

    pred = mv.argmax(axis=1)
    correct = pred == y
    acc = float(correct.mean())

    cfg = net.config
    eps = float(cfg.loss_label_smoothing)
    temperature = float(cfg.loss_temperature)
    bias = float(cfg.loss_count_bias)

    # Label-smoothing fixed point: probs that make (target - probs) == 0.
    p_correct = 1.0 - eps + eps / C
    p_other = eps / C
    logodds = np.log(p_correct / p_other)   # required correct-vs-other logit gap

    # Per-sample gaps in the RAW mean voltage (what temperature divides).
    correct_v = mv[np.arange(N), y]
    mv_other = mv.copy(); mv_other[np.arange(N), y] = -np.inf
    runnerup_v = mv_other.max(axis=1)
    gap_runnerup = correct_v - runnerup_v            # >0 iff classified correctly
    mv_nan = mv.copy(); mv_nan[np.arange(N), y] = np.nan
    gap_mean_others = correct_v - np.nanmean(mv_nan, axis=1)

    # Winner margin, independent of the label (top1 - top2 of each row).
    srt = np.sort(mv, axis=1)
    winner_margin = srt[:, -1] - srt[:, -2]

    print(f"\nProbed {N} samples, C={C} classes, argmax accuracy={acc:.3f}")
    print(f"config: loss_temperature={temperature}  label_smoothing={eps}  "
          f"loss_count_bias={bias} (scalar → shift-invariant → no-op in softmax)")

    print("\nRaw readout mean_voltage (over all sample x class):")
    _pct("mean_voltage", mv)

    print("\nCorrect-vs-rest gaps in mean_voltage (absolute units):")
    _pct("correct - runner-up", gap_runnerup)
    _pct("correct - mean(others)", gap_mean_others)
    _pct("winner margin (top1-top2)", winner_margin)

    # Matched temperature: gap / required-logit-gap, over CORRECT samples (those
    # define the separation the net actually achieves).
    med_gap = float(np.median(gap_runnerup[correct])) if correct.any() else float("nan")
    med_gap_mean = float(np.median(gap_mean_others[correct])) if correct.any() else float("nan")
    matched_runnerup = med_gap / logodds
    matched_mean = med_gap_mean / logodds

    print(f"\nLabel-smoothing target: correct-class prob = {p_correct:.4f}, "
          f"each other = {p_other:.4f}")
    print(f"  required correct-vs-other logit gap = log(p_c/p_o) = {logodds:.3f}")
    print(f"  median correct-vs-runner-up gap (correct samples)   = {med_gap:.3f}"
          f"  → matched temperature ≈ {matched_runnerup:.2f}")
    print(f"  median correct-vs-mean(others) gap (correct samples) = {med_gap_mean:.3f}"
          f"  → matched temperature ≈ {matched_mean:.2f}")

    # ── Temperature sweep: CE loss + confidence at the fixed operating point ──
    one_hot = np.eye(C)[y]
    target = one_hot * (1 - eps) + eps / C          # matches Network._smooth_targets

    if args.temps:
        grid = sorted(set(args.temps) | {temperature})
    else:
        base = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
        grid = sorted(set(base) | {round(temperature, 3),
                                   round(matched_runnerup, 3)})

    print("\nTemperature sweep (argmax accuracy is temperature-independent = "
          f"{acc:.3f}):")
    print(f"  {'temp':>7}   {'mean CE':>8}   {'mean p(correct)':>15}   "
          f"{'mean max-prob':>13}   {'|target p_c':>11}")
    for T in grid:
        logits = mv / T + bias
        probs = _softmax_rows(logits)
        ce = float(np.mean(-np.sum(target * np.log(probs + 1e-8), axis=1)))
        p_c_mean = float(np.mean(probs[np.arange(N), y]))
        maxp_mean = float(np.mean(probs.max(axis=1)))
        tag = "  <- config" if abs(T - temperature) < 1e-6 else (
            "  <- matched" if abs(T - matched_runnerup) < 1e-3 else "")
        print(f"  {T:7.3f}   {ce:8.4f}   {p_c_mean:15.4f}   {maxp_mean:13.4f}   "
              f"{p_correct:11.4f}{tag}")

    print("\nReading it: temperature ≈ matched keeps mean p(correct) near the target "
          f"p_c={p_correct:.3f}.\n  Too small → p(correct)→1, softmax saturates, "
          "(target-probs) gradient starves.\n  Too large → probs→uniform (~1/C="
          f"{1.0 / C:.3f}), weak separation, high CE floor.")


if __name__ == "__main__":
    main()
