# experiments/

Thin entry points (`run_shd.py`, `run_ssc.py`, `run_nmnist.py`) that call
`dendroprop.train()` and emit **one self-describing run dir** per run:

```
runs/<name>/
  config.json    full RunConfig (dataset, rule preset, seed, hyperparams)
  metrics.csv    per-epoch: epoch, loss, train_acc, test_acc, lr, gnorm_{r,s,d}, firing
  model.npz      weights + T_p + alphas + feedback B (if FA) + meta
  log.txt        optional stdout capture
```

Keep these thin — the model lives in `src/dendroprop/`. See `docs/architecture.md`.
