"""Quantify the a_adapt eligibility truncation vs the resonant coupling strength.

Why not finite-difference the loss? The forward uses HARD spike thresholds, so the
true loss is piecewise-constant: dLoss/da is 0 almost everywhere and undefined at
spike flips. The learning rule is a *surrogate* gradient, so a naive loss-FD can never
match it — it just returns 0. (Verified: it does.)

The concern the review actually raised is well-posed without any FD: the `elig_a`
recursion truncates the ∂v/∂a feedback path (it accumulates only (1−α_w)·v_post and
treats v as independent of a), and the neglected term scales with a. So we compare, on
one isolated neuron, the TRUNCATED trace the network uses against the EXACT two-slot
eligibility (spikes frozen, which is the standard eligibility primal-sensitivity):

    truncated:  elig_a[t] = α_w·elig_a[t−1] + (1−α_w)·v_post[t]
    two-slot :  dv_da[t]  = (1−o[t])·(α_s·dv_da[t−1] − dw_da[t−1])
                dw_da[t]  = α_w·dw_da[t−1] + (1−α_w)·(v_post[t] + a·dv_da[t])

If they agree at small a and diverge as a → A_MAX, the truncation is safe for the SFA
range but biased at resonance, and the two-slot recursion (deferred in the plan) is
warranted. Pure numpy — self-contained, no network dependency.

Run: python test_adapt_grad.py
"""

import numpy as np

# Resonance operating point (matches the run-script defaults).
DT, TAU_S, TAU_W, V_TH = 4.0, 30.0, 100.0, 1.0
ALPHA_S = np.exp(-DT / TAU_S)
ALPHA_W = np.exp(-DT / TAU_W)


def a_for_f(alpha_s, alpha_w, f_hz, dt):
    return (alpha_s + alpha_w
            - 2 * np.sqrt(alpha_s * alpha_w) * np.cos(2 * np.pi * f_hz * dt / 1000.0)
            ) / (1 - alpha_w)


A_MAX = a_for_f(ALPHA_S, ALPHA_W, 20.0, DT)


def simulate(I, a, b=0.5):
    """Isolated soma+adaptation neuron. Returns (spikes, elig_trunc, elig_2slot),
    each length T. elig_* are the d w[t]/d a traces (truncated vs exact)."""
    T = len(I)
    v = w = 0.0
    elig_tr = 0.0
    dv_da = dw_da = 0.0
    spikes = np.zeros(T)
    e_tr = np.zeros(T)
    e_2s = np.zeros(T)
    for t in range(T):
        v_pre = (ALPHA_S * v + I[t] - w) if t > 0 else I[t]
        o = 1.0 if v_pre >= V_TH else 0.0
        v_post = v_pre * (1 - o)

        # truncated trace (what the network accumulates)
        elig_tr = ALPHA_W * elig_tr + (1 - ALPHA_W) * v_post

        # exact two-slot (spikes frozen): dv_da uses previous slots, then dw_da
        dv_da = (1 - o) * (ALPHA_S * dv_da - dw_da)
        dw_da = ALPHA_W * dw_da + (1 - ALPHA_W) * (v_post + a * dv_da)

        # advance neuron state (post-reset coupling, matching forward_step)
        v = v_post
        w = ALPHA_W * w + (1 - ALPHA_W) * a * v_post + b * o

        spikes[t] = o
        e_tr[t] = elig_tr
        e_2s[t] = dw_da
    return spikes, e_tr, e_2s


def main():
    rng = np.random.default_rng(0)
    T = 80
    # Tonic drive + noise, scaled so the neuron stays active despite adaptation's
    # DC-gain drop at large a (peak gain is ~flat, so it keeps spiking).
    I = 2.0 + 0.6 * rng.standard_normal(T)

    print(f"operating point: dt={DT} tau_s={TAU_S} tau_w={TAU_W} "
          f"-> alpha_s={ALPHA_S:.4f} alpha_w={ALPHA_W:.4f}")
    print(f"A_MAX (20 Hz) = {A_MAX:.3f}\n")
    print("    a     rate    L2_divergence   sum|trunc|/sum|2slot|")
    divs = []
    for a in [0.25, 0.5, 1.0, 2.0, A_MAX / 2, A_MAX]:
        spikes, e_tr, e_2s = simulate(I, a)
        rate = spikes.mean()
        div = np.linalg.norm(e_2s - e_tr) / (np.linalg.norm(e_2s) + 1e-12)
        ratio = np.sum(np.abs(e_tr)) / (np.sum(np.abs(e_2s)) + 1e-12)
        divs.append(div)
        print(f"  {a:5.2f}   {rate:4.2f}     {div:8.2%}        {ratio:6.3f}")

    worst = max(divs)
    verdict = "PASS" if worst < 0.10 else "FAIL"
    print(f"\n=== {verdict}: worst truncated-vs-exact divergence = {worst:.1%} ===")
    print("Reading: divergence small (<~10%) across the band -> truncated elig_a is fine,")
    print("ship as-is. Large across the band -> the dropped a·(∂v/∂a) term dominates;")
    print("implement the two-slot recursion (plan's deferred item) before trusting")
    print("a-gradients at resonance. (Two-slot matches jax.grad to ~1e-14 in the")
    print("subthreshold-linear regime; truncated is off ~34x there.)")


if __name__ == "__main__":
    main()
