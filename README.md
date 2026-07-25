# Dendroprop

A two-compartment (dendritic plateau + LIF soma) spiking neural network trained with
**hand-derived analytic gradients — never autodiff**. The four backward-pass variants
are three configurable components on one gradient, so switching learning rule is a
config, not a forked codebase.

Methods write-up: the paper's Methods §2 (eqs 14–16) —
<https://ifftex.fz-juelich.de/project/6889fe76de1c1a67c085152f>.

## `src/dendroprop/` — the core

Three levels of abstraction; the single neuron is absent (everything is vectorized).

| Module | What it does |
|---|---|
| `config.py` | `NeuronConfig` (fixed neuron parameters + `soma_grad_scale`) and `RunConfig` (optimizer / lr / clip / decay / dropout). |
| `state.py` | SoA `flax.struct` state: `DynamicsState`, `EligState`, and the credit accumulators (`LayerCredit`/`TrainAccum`/`TrainableParams`). No wasted `(J,N,K)` tensors. |
| `dynamics.py` | The forward state-update equations (eqs 17–23), shared by inference and training. |
| `surrogate.py` | Surrogate derivative: `fast_sigmoid` \| `ste` (boxcar). |
| `feedback.py` | Top-down error `e_n = δ @ M` and fixed random feedback `B` init. |
| `credit.py` | Credit routing: `bptt` \| `dfa` \| `rfa`. |
| `layers.py` | `TwoCompLayer` (hidden), `ReadoutLayer` (LI); per-layer eligibility + grad assembly; the `dendritic_credit` (prune) option. |
| `network.py` | `Network` = hidden stack + readout + routing. One all-layers scan (`forward_accumulate`), inference twin (`forward`/`activity`), post-scan grad assembly, `build_network`. |
| `optim.py` | Hand-rolled AdamW / SGD (`adam_step`/`sgd_step`) — applies grads outside the credit computation; state threaded by the caller. |
| `train.py` | `train() -> (net, MetricsHistory)` + `evaluate()` — returns data instead of printing. |
| `io.py` | Run-dir artifacts (`write_run`/`load_run`) + `save_model`/`load_model`. |
| `data/` | Cached, TensorFlow-free dataset layer (below). |

**The gradient and its three components.** Top-down error `e_n = Σ_j δ_j·M_jn`
(`δ = p − y`); the gradients are J-collapsed to `(N,K)`, accumulated in the scan carry
and finalized post-scan. The three configurable components:

- **credit routing** (network level): `M = w_readout` → BPTT · `M = fixed random B` →
  DFA/RFA.
- **surrogate** (per layer): the derivative of the spike threshold.
- **dendritic-credit** (per layer): on → full · off → dendrite pruned (`grad_d ≡ 0`).

Inter-layer credit is soma→soma only; the dendrite's gradient is local. DFA is
implemented for N layers; the coupled bptt/rfa multi-layer recursion is a
`NotImplementedError` placeholder.

## `src/dendroprop/data/` — datasets

One generalized count-bin preprocessor + per-dataset source adapters, behind a
`SpikeDataset` Protocol. No TensorFlow.

- `preprocess.py` — `bin_events` (spatial pool = channel-map gather); `consecutive_pool_map`
  (SHD/SSC), `nmnist_pool_map` (2×2, 2312→578). uint8-safe counts.
- `sources.py` — `SHDSource`/`SSCSource` (HDF5 via h5py, urllib download) + `NMNISTSource`
  (AER binary decode).
- `cache.py` — preprocess once → uncompressed `.npy` + `np.load(mmap_mode='r')`, keyed by a
  JSON param hash. `build_cache` / `load_cached` / `default_params`.
- `dataset.py` — `SpikeDataset` Protocol, `CachedSpikeDataset` (augment-on-read: clean uint8
  on disk, channel-shift / spike-dropout applied per read), `iterate_batches`.

## `experiments/` and `figures/`

`experiments/run_shd.py` is a thin CLI (`--rule bptt|dfa|rfa|bptt_pruned|ste`,
`--synthetic N` for a no-download smoke) that emits one self-describing `runs/<name>/`
directory (`config.json` + `metrics.csv` + `model.npz`). `figures/fig_accuracy.py` ingests
those dirs with the stdlib only — it imports neither `dendroprop`, JAX, nor matplotlib
at import time.

## Tests & the golden

`tests/golden/` holds frozen reference outputs on fixed-seed synthetic count tensors.
The tests replay them as a CPU-float64 reference at `rtol=1e-9, atol=1e-11`; the coupled
methods (which approximate backprop, so there is no autodiff result to check against)
are tested against independent materialized references. Run everything:

```bash
python -m pytest tests/ -q          # golden + data + train checks
python experiments/run_shd.py --synthetic 64 --rule dfa --run-dir runs/smoke
```

## Install

`jax[cpu]` (or `jax[cuda12]` on Linux+NVIDIA) + `flax` + `numpy` + `h5py`; see
`requirements.txt` and `requirements-dev.txt` (pytest). GPU note: train float32 on the
GPU while the golden stays a CPU float64 reference (GPU FP64 is slow; the golden is not a
cross-backend bit-match).

## Symbol reference

| Symbol | Meaning | Code |
|---|---|---|
| $\alpha_s = e^{-\Delta t/\tau_s}$ | somatic membrane decay | `alpha_s` |
| $\alpha_d = e^{-\Delta t/\tau_d}$ | dendritic membrane decay | `alpha_d` |
| $\gamma$ | plateau-induced threshold reduction | `config.gamma` |
| $\theta_\nu$ | somatic spike threshold | `v_th` |
| $\theta_\mu$ | dendritic plateau threshold | `mu_th` |
| $T_{plat} = \lfloor \tau_p/\Delta t \rfloor$ | plateau duration (steps) | `T_p` |
| $\Delta t$ | simulation time step | `dt` = `bin_size_ms` |
| $w_i,\ \omega_i$ | somatic / dendritic input weights | `w_soma`, `w_dend` |
| $N_i[t]$ | input (spike count) of channel $i$ at step $t$ | `x_input` |
| $\alpha_m = e^{-\Delta t/\tau_m}$ | readout membrane decay | `alpha_m` |
| $W^{r}_{jn}$ | readout (LI) weight: class $j$ from neuron $n$ | `w_readout` |
| $\beta_s,\ \beta_d$ | surrogate sharpness (soma / dend) | `beta_s`, `beta_d` |
