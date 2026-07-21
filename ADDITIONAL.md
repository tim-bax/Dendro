# Dendro

Training spiking networks of **two-compartment dendritic neurons** on the Spiking
Heidelberg Digits (SHD) dataset, in JAX.

The distinguishing feature: **there is no autodiff anywhere.** Weight gradients are
derived by hand and computed online from eligibility traces carried alongside the forward
pass, contracted at the end with a single global error signal — a three-factor learning
rule. `jax.grad` is never called. This is what makes the code look unusual, and it is the
part worth reading first (`Original/network.py`).

## Status

Single hidden layer, trained with a hand-derived three-factor rule. **Feedback alignment
is not implemented yet** — see [Roadmap](#roadmap). `DFA/` and `RFA/` are placeholders and
are intentionally not committed until they contain real code.

## The model

```
input spikes (T, K)
      │
      ├─ w_dend ─→ dendritic compartment  μ  ─→ plateau potential h ∈ {0,1}
      │                                          (latches for T_p steps once μ crosses μ_th)
      └─ w_soma ─→ somatic compartment    v  ─→ spike o ∈ {0,1}
                                                 when v ≥ v_th − γ·h
                            │
                    (dropout, training only)
                            │
                     w_readout ─→ leaky integrator, no threshold, no reset
                            │
                     mean membrane voltage over T  ─→ softmax ─→ class
```

Each hidden neuron has its own plateau duration `T_p`, sampled once at init uniformly from
`[tau_plat_min, tau_plat_max]`. A plateau lowers the somatic threshold by `γ`, coupling the
two compartments. The readout is a **leaky integrator** (`LINeuron`) — it accumulates
voltage and never spikes; the class is the argmax of its time-averaged voltage.

Learning: `global_error = target − softmax(...)` is contracted with per-synapse eligibility
accumulators (`A_readout`, `A_soma`, `A_dend`) built during the forward `lax.scan`.
Non-differentiable spike thresholds are handled with the surrogate
`σ(x) = 1/(1+β|x|)²` (`Original/config.py`).

## Layout

| Path | What it is |
|---|---|
| `Original/config.py` | `NeuronConfig` (all hyperparameters) + surrogate gradient. Imports nothing local. |
| `Original/two_comp_neuron.py` | `TwoCompNeuron` — the dendritic hidden layer. Pure functions. |
| `Original/lif_neuron.py` | `LINeuron` — the leaky-integrator readout. **Note: despite the filename, there is no LIF neuron here** — no threshold, no reset. |
| `Original/network.py` | Composes the above. Forward+eligibility scan, hand-derived gradients, SGD/AdamW, save/load. |
| `Original/run_shd.py` | The only entry point. CLI, data loading, training loop. |
| `data/shd_binned.py` | SHD loading + count-binning. The only loader `Original/` uses. Standalone by design: NumPy + h5py, no TensorFlow. |
| `data/shd.py`, `ssc.py`, `nmnist.py` | Unused by `Original/`. `shd.py` imports TensorFlow (see Known issues). |

Dependency direction is strictly one-way: `config` ← {`two_comp_neuron`, `lif_neuron`} ←
`network` ← `run_shd`. The two neuron modules do not import each other.

## Install

```bash
conda create -n dendroprop python=3.12
conda activate dendroprop
pip install "jax[cuda12]==0.10.0"     # GPU. Use jax[cpu] for CPU-only.
pip install -r requirements.txt
```

`requirements.txt` pins the versions from the reference `dendroprop` env. **Do not install
plain `jaxlib` for GPU** — that is the CPU wheel and JAX will silently fall back to the CPU
backend. Check you got the GPU:

```bash
python -c "import jax; print(jax.default_backend(), jax.devices())"
# expected: gpu [CudaDevice(id=0)]
```

## Data

**SHD downloads itself on first run** from <https://zenkelab.org/datasets>; there is nothing
to fetch manually and no dataset in this repo. The cache directory is chosen by
`_default_cache_dir()` in `data/shd_binned.py`, first match wins:

1. `$SHD_CACHE_DIR`
2. `/share/neurocomputation/Tim/SHD_data` — an institution-specific path that predates this
   repo. If you are on that cluster you will silently read from it; set `$SHD_CACHE_DIR`
   to override.
3. `$SCRATCH/data`, then `$TMPDIR/data`, then `/scratch/data`, then `~/data`

```bash
export SHD_CACHE_DIR=/path/you/control     # recommended
```

Preprocessing follows Bittar & Garner (sparch) and Fabre et al. 2025
([arXiv:2506.06374](https://arxiv.org/abs/2506.06374)): spikes are counted into `bin_size_ms`
time bins (counts, not binary occupancy), then every `collapse_factor` input channels are
sum-pooled (700 → 140 at the default of 5), then zero-padded to a fixed `max_duration_ms`
window.

## Run

```bash
python Original/run_shd.py                      # defaults: 64 hidden, 10 epochs, SGD, lr 1e-3
```

A more typical configuration:

```bash
python Original/run_shd.py \
    --n_hidden 128 --epochs 50 --batch_size 32 \
    --optimizer adam --lr 1e-3 --weight_decay 1e-4 \
    --dropout 0.1 --early_stop_patience 10
```

`--help` lists all 34 flags. The trained model is written to
`Original/models/shd_seed{seed}_{timestamp}.npz` unless `--no_save_model` is passed; the
`.npz` bundles the weights, the per-neuron `T_p` array, the full `NeuronConfig`, and the CLI
args used, and `Network.load(path)` restores a fully usable model.

**Hyperparameters live on the CLI, not in `config.py`.** `main()` overrides every
`NeuronConfig` field from the parsed args, so the defaults in `config.py` are never the
values used at runtime. `run_shd.py --help` is the source of truth.

## Known issues

- **TensorFlow is installed but barely used.** `shd_binned.py` is standalone by design, but
  `from data.shd_binned import ...` runs `data/__init__.py` first, which eagerly re-exports
  from `data/shd.py`, which does `from tensorflow.keras.utils import get_file` at module
  level. So every run imports TF (and pays its startup cost) for a download helper it never
  calls. It is pinned in `requirements.txt` because it is genuinely needed as things stand.
  Making the `data/__init__.py` re-exports lazy would drop the dependency entirely.
- **`--precision 32` half-works.** `Original/network.py` casts to `float64`
  unconditionally in six places. With `--precision 32` (i.e. `jax_enable_x64=False`) JAX
  silently downcasts these to float32 and prints a warning per site rather than erroring.
  Results are float32 but the run is noisy. `--precision 64` (the default) is the tested path.
- **The plateau/spike dynamics are written twice** — in `TwoCompNeuron.forward_step` and
  hand-copied into `network._predict_only`. They are currently equivalent (verified), but
  nothing enforces it: **if you change one, change the other.**
- **`w` is overloaded**: the adaptation *state variable* inside `TwoCompNeuron`, but *weights*
  (`w_dend`, `w_soma`, `w_readout`) everywhere else. Both meanings appear in one scope in
  `_predict_only`.
- **No tests.** Since the gradients are hand-derived, the most valuable thing to add is a
  finite-difference check of `_loss_and_grads` against a numerical gradient.

## Roadmap

`DFA/` and `RFA/` will hold Direct and Random Feedback Alignment variants, replacing the
exact-weight feedback path (`w_readout` in the `eta` term of `_forward_and_accum`) with a
fixed random matrix.

**When adding them, import the shared core — do not copy `Original/`.** They began as
byte-identical copies of it; keeping three copies in sync by hand is what this structure has
to avoid.
