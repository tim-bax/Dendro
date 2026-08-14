# Backward-pass comparison: BPTT_fix vs BPTT_fix_auto

- arch: 3 inputs -> 6 -> 6 two-comp hidden -> 2 readout, target class 0, burst amp 4.0 on t=5..45
- L0: plateau neurons [0, 2, 4] (steps [52, 0, 54, 0, 53, 0])
- L1: plateau neurons [0, 2, 3] (steps [41, 0, 45, 46, 0, 0])
- L0: spiking neurons [0, 1, 2, 4, 5] (spikes [38, 13, 39, 0, 6, 40])
- L1: spiking neurons [0, 2, 3, 4] (spikes [1, 0, 5, 7, 1, 0])

Three gradient sources:
  fix        — hand-derived, pruned, e-prop rule (BPTT_fix, as returned)
  auto_full  — exact reverse-mode surrogate BPTT (nothing pruned)
  auto_prune — exact autodiff, inter-layer dendrite edge stop_gradient'd
               (prunes exactly #1; keeps the exact within-layer time adjoint)
  fix_norm   — fix with its rescalings undone: readout,dend x temp; soma x temp/scale
               (temp=5, _SOMA_GRAD_SCALE=8)

Decomposition:  auto_full↔auto_prune = #1 inter-layer dendrite prune;
                auto_prune↔fix_norm  = #2 e-prop time truncation;  fix↔auto_full = total.

## Readout gradient (exact in all; sanity check)
  cos(fix_norm, auto_full) = +1.000000   (expect ~1.0)
  med ratio fix/auto_full  = +5.0000   (expect temp=5)

## Per-hidden-layer gradients at T=60

### hidden layer 0
  ||W_soma grad||:  fix_norm=2.5866e+00  prune=1.5120e+05  full=1.4331e+05
  ||W_dend grad||:  fix_norm=3.1903e-01  prune=2.9145e+04  full=2.9579e+04
  --- #1 inter-layer dendrite prune (auto_full vs auto_prune) ---
    soma  ||full-prune||/||full|| = 0.0679   cos = +0.9990
    dend  ||full-prune||/||full|| = 0.0550   cos = +0.9986
  --- #2 e-prop time truncation (auto_prune vs fix_norm) ---
    soma  cos = +0.5341   med ratio prune/fix_norm = +25811.879
    dend  cos = +0.8503   med ratio prune/fix_norm = +111986.773
  --- total (fix_norm vs auto_full) ---
    soma  cos = +0.5693    dend  cos = +0.8668

### hidden layer 1
  ||W_soma grad||:  fix_norm=4.3294e+00  prune=5.7249e+04  full=5.7249e+04
  ||W_dend grad||:  fix_norm=6.7021e-01  prune=4.8766e+03  full=4.8766e+03
  --- #1 inter-layer dendrite prune (auto_full vs auto_prune) ---
    soma  ||full-prune||/||full|| = 0.0000   cos = +1.0000
    dend  ||full-prune||/||full|| = 0.0000   cos = +1.0000
  --- #2 e-prop time truncation (auto_prune vs fix_norm) ---
    soma  cos = +0.0860   med ratio prune/fix_norm = +0.232
    dend  cos = +0.0255   med ratio prune/fix_norm = +0.355
  --- total (fix_norm vs auto_full) ---
    soma  cos = +0.0860    dend  cos = +0.0255
  --- exact inter-layer adjoint into L1: ||dendrite route|| / ||soma route|| = 0.0747  (soma=3.943e+03, dend=2.945e+02) ---

## Gradient magnitude vs sequence length (the key difference)
     T           fix    auto_prune     auto_full    full/fix
    20     2.253e-01     5.133e-01     5.185e-01    2.30e+00
    30     5.704e-01     5.497e+00     5.148e+00    9.03e+00
    40     8.898e-01     1.150e+02     1.084e+02    1.22e+02
    60     1.388e+00     9.802e+04     9.059e+04    6.52e+04
    80     1.421e+00     8.941e+07     8.009e+07    5.63e+07
   120     1.110e+00     1.867e+10     1.636e+10    1.47e+10
   175     7.786e-01     5.274e+10     4.604e+10    5.91e+10

per-step growth factor (max|dW_soma|):  auto_full ~ 1.177/step,  auto_prune ~ 1.178/step,  fix ~ 1.008/step (>1 ⇒ exponential blow-up; e-prop hand rule ≈ flat).
