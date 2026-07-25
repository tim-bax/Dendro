# tests/golden/ — frozen reference outputs

Frozen reference outputs on fixed-seed synthetic count tensors (no dataset download).
The tests replay these artifacts; they are immutable and are never regenerated.

## Files
- `golden.npz` — frozen inputs (`input__*`) and reference outputs, one array each.
- `manifest.json` — jax/flax/numpy versions, scenario seeds/sizes, the full
  `NeuronConfig`, the tolerance contract, and scalar outputs for eyeballing.
- `init_recipe.npz` — the single-layer weights (`w_dend/w_soma/T_p/w_readout/alpha_*`)
  that `network.build_network`'s init must reproduce; pinned by
  `tests/test_multilayer.py::test_build_network_matches_frozen_init_recipe`.

## Coverage
`mean_voltage · prediction · loss · grad_{readout,soma,dend} · activity`, over
{single, batch} × {dropout off, dropout on (fixed key)}, plus the predict-only path.
The `*__aux_A_{r,s,d}` arrays are the `(J,N,K)` accumulators kept as intermediate
debugging aids; the primary contract is the final `grad_*`.

## Contract
- **CPU float64 reference** (`JAX_PLATFORMS=cpu`, `jax_enable_x64`). CPU/GPU reduction
  orders differ — this is not a cross-backend bit-match. Tolerance: `rtol=1e-9,
  atol=1e-11`; predictions match exactly.
- Inputs (weights, `T_p`, decays, dropout keys, tensors) are frozen as arrays, so
  `src/dendroprop` is fed byte-identical inputs regardless of how it initialises.

## Use
```bash
pytest tests/test_dynamics.py tests/test_grads.py -v      # replay vs golden.npz
pytest tests/test_multilayer.py -k init_recipe -v         # build_network vs init_recipe.npz
```
