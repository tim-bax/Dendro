# Dendroprop — architecture

`Dendroprop` trains a two-compartment (LIP dendrite + LIF soma) spiking network with
**hand-derived analytic gradients — never autodiff** (paper methods §2, eqs 14–16). The
whole model is the modular `src/dendroprop` core.

Design goals the core meets:

1. Four backward-pass variants are three configurable components on one gradient —
   switching rule is a config, not a forked codebase.
2. SHD / SSC / NMNIST behind one dataset interface.
3. Efficient (SoA state, no wasted `(J,N,K)` tensors) and structured for paper figures and
   HPC — a clean `src/` library, `experiments/` that emit run dirs, `figures/` that ingest
   them without importing JAX.

## The gradient and its three components

Top-down error `e_n = Σ_j δ_j·M_jn` with `δ = p − y`. The gradients are already in
J-collapsed `(N,K)` form — no `(J,N,K)` tensor exists in the math. The scan accumulates
three sums in the carry, then applies `e_n = δ @ M` once, post-scan:

```
G_r (N,)  = Σ_t ε^r                        grad_r = outer(δ, G_r) / T
G_s (N,K) = Σ_t ψ^s ε^s                     grad_s = e[:,None]·G_s / (T · soma_grad_scale)
G_d (N,K) = Σ_t γ ψ^s ψ^d ε̄^d               grad_d = e[:,None]·G_d / T
```

| Component | Lives at | Options → methods |
|---|---|---|
| **credit routing** | `credit.py` (`e_n = δ @ M`) | `M = w_readout` → **BPTT** · `M = fixed random B` → **DFA/RFA** (identical for 1 hidden layer) |
| **surrogate** | per-layer (`surrogate.py`) | fast-sigmoid `1/(1+β|x|)²` (default) · **STE** boxcar |
| **dendritic-credit** | per-layer (`layers.py`, `grad_d` term) | on → full · off → **pruned** (`grad_d ≡ 0`, forward unchanged) |

Routing is a network-level static field (`Network.credit_method` ∈ `bptt|dfa|rfa`), resolved
once at trace time to a `credit.CreditStrategy`; the surrogate and dendritic-credit are
per-layer static fields. All three compose freely; the composite presets (`bptt_pruned`,
`ste`) are just `build_network(..., surrogate=, dendritic_credit=)` kwargs (see
`experiments/run_shd.py`).

## Three levels of abstraction

The single neuron is absent (everything vectorized over N):

```
src/dendroprop/
  config.py     NeuronConfig (fixed neuron parameters + soma_grad_scale), RunConfig (optimizer/lr/…)
  state.py      DynamicsState, EligState, LayerCredit/LayerIngredients/TrainAccum/TrainableParams (SoA)
  dynamics.py   forward state-update equations (eqs 17–23), shared by inference + training
  surrogate.py  fast_sigmoid | ste                                   (the surrogate)
  feedback.py   e_n = δ @ M; fixed random B init                     (credit-routing helpers)
  credit.py     credit routing: bptt | dfa | rfa
  layers.py     TwoCompLayer (hidden), ReadoutLayer (LI); eligibility + grad assembly (dendritic-credit)
  network.py    Network (compose + one all-layers scan + post-scan grads), build_network
  optim.py      hand-rolled AdamW/SGD — applies grads, outside credit; state threaded by caller
  train.py      train() -> (net, MetricsHistory); evaluate()
  io.py         write_run/load_run + save_model/load_model (run-dir artifacts)
  data/         SpikeDataset Protocol + count-bin preprocessor + shd/ssc/nmnist adapters
experiments/    run_shd.py → one run dir per run
figures/        fig_*.py → glob & ingest run dirs → plots (never import training/JAX)
tests/golden/   frozen reference artifacts (immutable)
(future) experiments/slurm/  array-job sbatch template + config generator
```

The training pass is one all-layers time-scan (`Network.forward_accumulate`): a bottom-up
dynamics sweep with the input projection **inside** the scan + on-the-fly eligibility/credit
accumulation, **no trajectory storage**. Inference (`forward`/`activity`) runs the same sweep
with the credit machinery dropped and no dropout. Batching collapses the positional vmap
spec to one `0` per struct arg, `None` for the batch-shared `Network`.

## Backward topology & invariants (do not change)

- **Inter-layer credit is soma→soma only**; the dendrite's gradient is local (no
  `dendrite_backward`, else a `2^L` trace blow-up). Grad_d is credited from a layer's own
  soma error, never propagated earlier.
- **DFA is the only method that decouples across time** — it collapses to a post-scan
  `e^L = δ @ B_L` per layer and is fully implemented for N layers. Coupled `bptt`/`rfa` for
  **>1 hidden layer** are a `NotImplementedError` placeholder (`credit.soma_backward`); the
  running accumulator must be **O(Σ params), not O(T)/depth**. The coupled methods
  approximate backprop, so there is no autodiff result to check against — test
  running==materialized reference, never `jax.grad`.
- **The `/8` divisor.** `grad_s` is divided by `T · soma_grad_scale` with
  `soma_grad_scale = 8.0` (a `NeuronConfig` field). Without it somatic grads come out ~8×
  too large and SHD regresses.
- **Scan-carry dtype under x64.** `lax.scan` requires the carry pytree's dtypes to
  round-trip exactly; keep `h`/`t_prime` explicitly `int32`. A stray Python-int branch
  upcasts to int64 and the scan fails.
- **Dropout wiring:** mask hidden spikes **only** on the readout-input path; `E_soma`/`ε^d`/`ψ`
  never see the mask. Post-scan `e_n` is then exactly equivalent — dropout is baked into
  `mean_voltage` (→δ) and `G_r` before δ is formed.

## Datasets — unified count-bin, cached, augment-on-read

The model consumes dense **count** tensors `N_i[t]`, so every dataset routes through one
generalized count-bin preprocessor (`data/preprocess.py:bin_events`). Per-dataset **source
adapters** (`data/sources.py`) yield raw events; the preprocessor bins + spatially pools
(a channel-map gather) + zero-pads to `(T, C)`. Spatial pool: SHD/SSC sum consecutive
channels (700→140); **NMNIST needs a 2×2 pool** over its 34×34×2 grid (2312→578) — dense is
~55 GB f32, its own adapter concern.

- **Cache** (`data/cache.py`): preprocess once → uncompressed `.npy` + `np.load(mmap_mode='r')`
  for the big `X`; small `.npy` for `y`/`lengths`; a **JSON sidecar whose param hash is the
  cache key**. Counts stored `uint8` (cast to float on read; ~4× smaller, exact ≤255). A warm
  cache needs no source, no download, no TensorFlow.
- **Augmentation** (channel-shift, spike-dropout) is applied per read in `__getitem__`, so the
  cache holds only clean tensors.

```python
@runtime_checkable
class SpikeDataset(Protocol):
    n_inputs: int; n_classes: int; T: int
    def __len__(self) -> int: ...
    def __getitem__(self, i) -> tuple[np.ndarray, int]: ...   # (T,C) float, label
    def cache_key(self) -> str: ...
def build_cache(name, split, params, *, data_path=None, root=None, force=False) -> Path
def load_cached(name, split, params, *, augment=None, rng=None, ...) -> SpikeDataset
```

## Run artifacts, experiments, figures

`train() -> (net, MetricsHistory)` returns data instead of printing. `io.write_run` emits
one self-describing directory per run:

```
runs/<name>/
  config.json    RunConfig + NeuronConfig + dims + credit_method
  metrics.csv    per-epoch: epoch, loss, train_acc, test_acc, lr, gnorm_{r,s,d}, firing
  model.npz      weights + T_p + alphas + feedback B + a JSON meta blob
  log.txt        optional
```

`experiments/run_shd.py` is a thin CLI (`--rule`, `--synthetic` for no-download smokes) →
one run dir. `figures/fig_accuracy.py` ingests run dirs with the **stdlib only** — it imports
neither `dendroprop`, JAX, nor matplotlib (matplotlib lazy inside the plot fns).

## Testing & the golden

- **Golden replay** (`tests/golden/golden.npz`, frozen/immutable): the `(N,K)` path
  reproduces the reference `loss/preds/grad_{r,s,d}/mean_voltage/activity`, single+batched,
  dropout on/off, within `rtol=1e-9` (a CPU float64 reference — not a cross-backend
  bit-match). `init_recipe.npz` pins `build_network`'s init. Component properties: routing
  (`M=w_readout`) ≡ golden; dendrite pruned ⇒ `grad_d==0` & forward identical; STE surrogate
  ⇒ boxcar, finite.
- **End-to-end:** `python experiments/run_shd.py --rule bptt` on real SHD reaches ~68% test
  accuracy and writes a run dir; `figures/fig_accuracy.py` renders it with no JAX.
- **Run:** `.venv/bin/python -m pytest tests/ -q` (uv venv, `jax[cpu]`; `conftest.py` sets
  `JAX_PLATFORMS=cpu` + `jax_enable_x64` before jax import and paths `src/`).

## Roadmap — next steps

1. **Back up / publish.** The codebase lives only on local `pcl-dev` (unpushed). Merge to
   `main` and/or push to origin — pure risk mitigation.
2. **GPU migration.** Swap `jax[cpu]` → `jax[cuda12]==0.10.0` (already the `requirements.txt`
   pin) on Linux+NVIDIA; train float32 on GPU, keep the golden as the CPU float64 reference.
3. **Coupled bptt/rfa multi-layer `soma_backward`.** The one feature gap — deep BPTT/RFA
   (>1 hidden layer) still raise. Needs the hand-derived equations from
   `dendroprop_methods.pdf` (accumulator O(Σ params)).
4. **Reproduce paper accuracy.** SHD reaches 68.5% test / 93% train (overfitting) — enable
   `--spike-dropout`/`--channel-shift`/`--dropout`, tune, and run the rule comparison.
5. **SSC + NMNIST validation** on real data (NMNIST via the 2×2 AER pool path).
6. **Sweep/figure tooling.** Add `matplotlib` to `requirements-dev.txt`; the
   `experiments/slurm/` array-job template; paper figures.
