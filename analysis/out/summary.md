# BPTT vs BPTT_adapt — dynamics comparison

- **BPTT**: 117 input ch, v_th=1.0, adaptation=fixed scalar, bin=4.0 ms, T=175
- **BPTT_adapt**: 140 input ch, v_th=0.8, adaptation=per-neuron trained, bin=4.0 ms, T=175

> Side-by-side characterization, not a controlled ablation: the two models differ in input width, v_th, and adaptation style.

```
METRIC                                                  BPTT                BPTT_adapt
--------------------------------------------------------------------------------------
test accuracy (this batch)                            90.82%                    91.80%
readout v_mean (pred class)                           0.3173                    0.5492
readout margin (top1-top2)                            0.4711                    0.5147

--- hidden layer 0 (N=512) ---
  firing rate (spk/neuron/step)                       0.0793                    0.1158
  firing rate (Hz)                                      19.8                      29.0
  plateau occupancy                                   0.3099                    0.5150
  mean plateau dur (steps)                             67.07                     66.75
  mean plateau dur (ms)                                268.3                     267.0
  P(spike | plateau)                                  0.1464                    0.1476
  P(spike | no plateau)                               0.0492                    0.0822
  plateau spike boost (x)                               2.98                      1.80
  dead-neuron fraction                                 0.000                     0.000
  spikes/neuron/seq (med)                              11.95                     14.25
  spikes/neuron/seq (max)                               93.4                     127.6
  T_p range (steps, ref)                              37-123                    37-123

--- hidden layer 1 (N=512) ---
  firing rate (spk/neuron/step)                       0.0311                    0.0367
  firing rate (Hz)                                       7.8                       9.2
  plateau occupancy                                   0.0952                    0.3982
  mean plateau dur (steps)                             57.25                     64.00
  mean plateau dur (ms)                                229.0                     256.0
  P(spike | plateau)                                  0.1149                    0.0527
  P(spike | no plateau)                               0.0223                    0.0261
  plateau spike boost (x)                               5.15                      2.02
  dead-neuron fraction                                 0.002                     0.000
  spikes/neuron/seq (med)                               4.51                      5.54
  spikes/neuron/seq (max)                               32.6                      36.3
  T_p range (steps, ref)                              37-123                    37-123

```
