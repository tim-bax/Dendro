# figures/

Paper-figure scripts (`fig_*.py`) that **glob and ingest run dirs** from
`experiments/` (`config.json` + `metrics.csv`, plus `model.npz` when a figure
needs weights) and render plots.

Hard rule: **never import the training stack or JAX/CUDA here.** Figures read CSV
+ JSON only, so they run anywhere with just numpy + a plotting lib.

See `docs/architecture.md`.
