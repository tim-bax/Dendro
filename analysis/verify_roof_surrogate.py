#!/usr/bin/env python3
"""Does the dendritic-path 'roof' surrogate have the intended shape?

Soft_Reset's dendrite can only affect the soma via a binary plateau gate h that
lowers the somatic threshold to v_th_eff = v_th - gamma*h. The dendritic e-prop
path therefore wants a surrogate that credits a neuron whenever its voltage sits
anywhere in the band [v_th-gamma, v_th] a plateau could flip -- a flat "roof" --
rather than a single bump at one threshold (config.surrogate_roof).

This script checks that shape numerically and saves a plot next to it:

    - the roof peaks INSIDE the band [v_th-gamma, v_th];
    - it decays to ~0 well below v_th-gamma and well above v_th (where the plateau
      can't change the spike);
    - the top gets flatter as beta_s_dend*gamma grows (edge sharpness);
    - the exact finite difference H(v-(v_th-gamma)) - H(v-v_th) it approximates is
      a boxcar of height 1 over the band.

Usage (dendroprop env, from repo root):
    python analysis/verify_roof_surrogate.py
"""
import os
import sys

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
_SOFT = os.path.join(_ROOT, "Soft_Reset")
for _p in (_ROOT, _SOFT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import surrogate_sigma, surrogate_roof  # noqa: E402


def main():
    v_th, gamma = 1.0, 0.5
    lo, hi = v_th - gamma, v_th
    betas = [1.0, 4.0, 10.0]

    center = 0.5 * (lo + hi)
    dv = 4e-4
    v = jnp.arange(v_th - 3 * gamma, v_th + 2 * gamma, dv)
    vnp = np.asarray(v)
    in_band = (vnp >= lo) & (vnp <= hi)
    boxcar = in_band.astype(float)                  # exact target the roof matches
    d_out = 0.6 * gamma                             # a fixed probe distance outside the band

    print(f"band = [{lo:.3g}, {hi:.3g}]  center={center:.3g}  gamma={gamma}\n")
    ok = True
    tail_vals = []
    for beta in betas:
        roof = np.asarray(surrogate_roof(v, lo, hi, beta))
        # 1) flat top: exactly 1 across the whole band
        flat_err = float(np.max(np.abs(roof[in_band] - 1.0)))
        # 2) symmetric about the band center (mirror v across center)
        mirror = np.asarray(surrogate_roof(2 * center - v, lo, hi, beta))
        sym_err = float(np.max(np.abs(roof - mirror)))
        # 3) monotonic decay moving outward from each edge
        left = roof[vnp <= lo]                       # increasing v -> should rise to 1
        right = roof[vnp >= hi]                      # increasing v -> should fall from 1
        mono = bool(np.all(np.diff(left) >= -1e-12) and np.all(np.diff(right) <= 1e-12))
        # 4) value at a fixed distance outside the band (for the beta=decay check)
        tail = float(surrogate_roof(jnp.array(lo - d_out), lo, hi, beta))

        flat_ok = flat_err < 1e-9
        sym_ok = sym_err < 1e-9
        ok = ok and flat_ok and sym_ok and mono
        tail_vals.append(tail)
        print(
            f"beta={beta:5.1f}: flat_top|roof-1|={flat_err:.1e} {'ok' if flat_ok else 'BAD!'}  "
            f"sym_err={sym_err:.1e} {'ok' if sym_ok else 'BAD!'}  "
            f"monotonic_decay={mono}  roof(edge-{d_out:g})={tail:.3f}"
        )

    # 5) exact degradation to the single bump as the band collapses (lo -> hi)
    degrade_err = float(np.max(np.abs(
        np.asarray(surrogate_roof(v, v_th, v_th, 4.0))
        - np.asarray(surrogate_sigma(v - v_th, 4.0))
    )))
    degrades = degrade_err < 1e-12
    # 6) beta sets the OUT-OF-BAND decay rate: larger beta -> smaller tail
    decay_monotone = all(tail_vals[i] > tail_vals[i + 1] for i in range(len(tail_vals) - 1))

    print(f"\nflat top == 1 across band, symmetric, monotonic decay: {ok}")
    print(f"reduces exactly to surrogate_sigma as band->0 (err={degrade_err:.1e}): {degrades}")
    print(f"beta = out-of-band decay rate (tail shrinks with beta): {decay_monotone}  "
          f"{[round(t,3) for t in tail_vals]}")
    print(f"\nSHAPE CHECK: {'PASS' if (ok and degrades and decay_monotone) else 'FAIL'}")

    # ── Plot ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.fill_between(np.asarray(v), 0, boxcar, color="0.85",
                        label="exact H(v-lo)-H(v-hi) (boxcar)")
        for beta in betas:
            ax.plot(np.asarray(v), np.asarray(surrogate_roof(v, lo, hi, beta)),
                    label=f"roof beta_s_dend={beta:g}")
        # old single bump for reference (centered at static v_th, h=0 case)
        ax.plot(np.asarray(v), np.asarray(surrogate_sigma(v - v_th, 1.0)),
                "k--", alpha=0.6, label="old bump surrogate_sigma(v-v_th, 1)")
        ax.axvline(lo, color="r", ls=":", alpha=0.5)
        ax.axvline(hi, color="r", ls=":", alpha=0.5)
        ax.set_xlabel("v_soma (pre-reset)")
        ax.set_ylabel("surrogate value")
        ax.set_title("Dendritic-path roof surrogate over [v_th-gamma, v_th]")
        ax.legend(fontsize=8)
        out = os.path.join(_SCRIPT_DIR, "roof_surrogate_shape.png")
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        print(f"\nSaved plot -> {out}")
    except Exception as e:  # noqa: BLE001
        print(f"\n(plot skipped: {e})")


if __name__ == "__main__":
    main()
