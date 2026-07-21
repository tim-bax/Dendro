# Dendro — quick reference

So far /Original is the 1-Layer model folder which all results have been reproduced with. THe DFA and RFA folders are placeholders for future implementations. 

More information about this project can be found in the Methods section of the ifftex document:" https://ifftex.fz-juelich.de/project/6889fe76de1c1a67c085152f"

## `Original/` — the model and training code

| File | What it does |
|---|---|
| `config.py` | `NeuronConfig` dataclass — every hyperparameter (defaults overridden by CLI) — plus `surrogate_sigma`, the surrogate derivative `σ(x)=1/(1+β|x|)²`. Imports nothing local. |
| `two_comp_neuron.py` | `TwoCompNeuron`, the dendritic hidden layer. `dynamics_step` = the 6-slot physics (single source of truth); `forward_step` = same equations wearing the 9-slot training carry; `update_{somatic,dendritic}_eligibility` = the eligibility traces. |
| `lif_neuron.py` | `LINeuron`, the leaky-integrator readout — **no threshold, no reset**. `integrate` (inference) and `forward_step` (adds the readout eligibility trace). |
| `network.py` | `Network` — ties it together. Forward+eligibility `lax.scan` (`_forward_and_accum`), forward-only twin (`_predict_only`), hand-derived three-factor gradients (`_loss_and_grads`), SGD/AdamW optimizers, activity diagnostics, `save`/`load`. jit/vmap wrappers give single- and batch-sample variants. |
| `run_shd.py` | The only entry point. CLI (34 flags, the real source of truth for hyperparameters), data loading, training loop, LR scheduling / early stopping, model saving. |

Dependency direction is one-way: `config` ← {`two_comp_neuron`, `lif_neuron`} ← `network` ← `run_shd`.

## `data/` — dataset loaders

| File | What it does |
|---|---|
| `shd_binned.py` | **The only loader `Original/` uses.** Count-bin SHD preprocessing (Bittar & Garner / Fabre et al. 2025): time-bin into counts, sum-pool every `collapse_factor` channels, zero-pad to a fixed window. Exposes `load_shd_binned`, `apply_channel_shift`. |
| `shd.py` | Older SHD loader with alpha-kernel input currents. **Imports TensorFlow** (`get_file` download helper). Unused by `Original/`. |
| `ssc.py` | SSC (Spiking Speech Commands) loader, same alpha-kernel style. Unused by `Original/`. |
| `nmnist.py` | NMNIST loader. Unused by `Original/`. |
| `__init__.py` |  |

## Symbol reference

At a glance:

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

LaTeX source for the paper (requires `amsmath` and `booktabs`):

