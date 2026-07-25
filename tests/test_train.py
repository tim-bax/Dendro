"""Train loop + run artifacts.

The analytic loss/grads are already checked against the golden at the ``loss_and_grads``
level, so this file covers the orchestration ``train`` adds: that it performs exactly
the per-step sequence (pinned bit-for-bit against a manual replication), that presets
run and move weights while the fixed constants stay frozen, that ``evaluate`` is
consistent, that ``io`` round-trips a model + run dir, and that ``figures/`` import
neither JAX nor ``dendroprop``. No real dataset needed — synthetic count tensors.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from jax import random

from dendroprop import network, io
from dendroprop.config import NeuronConfig, RunConfig
from dendroprop.network import build_network, smooth_targets
from dendroprop.optim import adam_init, adam_step, hypers_from_run
from dendroprop.train import train, evaluate, MetricsHistory, METRIC_FIELDS
from dendroprop.data.dataset import CachedSpikeDataset

_ROOT = Path(__file__).resolve().parents[1]


def _dataset(n, T, C, n_classes, seed):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 3, size=(n, T, C), dtype=np.uint8)
    y = rng.integers(0, n_classes, size=n).astype(np.int64)
    lengths = np.full(n, T, dtype=np.int64)
    meta = {"n_inputs": C, "n_classes": n_classes, "T": T, "cache_key": f"syn-{seed}"}
    return CachedSpikeDataset(X, y, lengths, meta)


def _stack_all(ds):
    xs = [ds[i][0] for i in range(len(ds))]
    ys = [ds[i][1] for i in range(len(ds))]
    return jnp.asarray(np.stack(xs)), jnp.asarray(np.asarray(ys, dtype=np.int64))


# --------------------------------------------------------------------------- #
# 1. Integration gate: train() == the validated manual step, bit-for-bit      #
# --------------------------------------------------------------------------- #
def test_train_step_matches_manual_replication():
    C, T, N, J = 10, 12, 4, 6
    cfg = NeuronConfig(weight_scale=1.0)
    run = RunConfig(optimizer="adam", lr=1e-2, dropout_rate=0.0)
    ds = _dataset(N, T, C, J, seed=3)
    net0 = build_network(random.PRNGKey(0), (C, [8], J), cfg, "bptt")

    # single full-batch epoch, no shuffle -> exactly one deterministic step
    trained, hist = train(net0, ds, ds, run, epochs=1, batch_size=N,
                          seed=0, shuffle=False)

    # manual replication of train's documented protocol
    xb, y = _stack_all(ds)
    sm = smooth_targets(y, J, cfg.loss_label_smoothing)
    epoch_key = random.fold_in(random.PRNGKey(0), 0)
    keys = random.split(random.fold_in(epoch_key, 0), N)
    mv, accum = network.forward_accumulate_batch(net0, xb, keys, run.dropout_rate)
    loss_b, _, grads_b = network.loss_and_grads_batch(net0, mv, accum, sm, T)
    grads = jax.tree.map(lambda g: jnp.mean(g, axis=0), grads_b)
    params, _ = adam_step(net0.trainable(), grads, adam_init(net0.trainable()),
                          hypers_from_run(run))

    for a, b in zip(jax.tree_util.tree_leaves(trained.trainable()),
                    jax.tree_util.tree_leaves(params)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert len(hist.epochs) == 1
    assert np.isfinite(hist.last.loss)


def test_train_is_reproducible_and_seed_sensitive():
    C, T, N, J = 8, 10, 6, 5
    cfg = NeuronConfig(weight_scale=1.0)
    run = RunConfig(optimizer="adam", lr=1e-2)
    ds = _dataset(N, T, C, J, seed=1)
    net = build_network(random.PRNGKey(0), (C, [8], J), cfg, "dfa")

    _, h1 = train(net, ds, ds, run, epochs=2, batch_size=3, seed=7)
    _, h2 = train(net, ds, ds, run, epochs=2, batch_size=3, seed=7)
    _, h3 = train(net, ds, ds, run, epochs=2, batch_size=3, seed=8)
    assert h1.column("loss") == h2.column("loss")        # same seed -> identical
    assert h1.column("loss") != h3.column("loss")        # different seed -> differs


# --------------------------------------------------------------------------- #
# 2. Preset smoke: runs, moves weights, freezes intrinsics                     #
# --------------------------------------------------------------------------- #
def _build(method, seed=0, **kw):
    cfg = NeuronConfig(weight_scale=1.0)
    return build_network(random.PRNGKey(seed), (10, [8], 6), cfg, method, **kw), cfg


@pytest.mark.parametrize("method,kw", [
    ("bptt", {}),
    ("dfa", {}),
    ("bptt", {"dendritic_credit": False}),   # pruned dendrite
    ("bptt", {"surrogate": "ste"}),          # straight-through surrogate
])
def test_train_preset_smoke(method, kw):
    net, cfg = _build(method, **kw)
    run = RunConfig(optimizer="adam", lr=1e-2, dropout_rate=0.1)
    train_ds = _dataset(12, 12, 10, 6, seed=2)
    test_ds = _dataset(6, 12, 10, 6, seed=9)

    before = net.trainable()
    Tp0 = [np.asarray(l.T_p) for l in net.hidden]
    B0 = [np.asarray(b) for b in net.B]

    trained, hist = train(net, train_ds, test_ds, run, epochs=3, batch_size=4, seed=5)

    assert len(hist.epochs) == 3
    assert all(np.isfinite(m.loss) for m in hist.epochs)
    for m in hist.epochs:
        assert 0.0 <= m.train_acc <= 1.0 and 0.0 <= m.test_acc <= 1.0
        assert m.firing >= 0.0
    # readout weights moved; frozen intrinsics byte-identical
    assert not np.allclose(np.asarray(trained.trainable().w_readout),
                           np.asarray(before.w_readout))
    for l, tp in zip(trained.hidden, Tp0):
        np.testing.assert_array_equal(np.asarray(l.T_p), tp)
    for b_new, b_old in zip(trained.B, B0):
        np.testing.assert_array_equal(np.asarray(b_new), b_old)
    assert trained.config == cfg

    if not kw.get("dendritic_credit", True):     # pruned -> dendrite weights never move
        np.testing.assert_array_equal(np.asarray(trained.trainable().w_dend[0]),
                                      np.asarray(before.w_dend[0]))





def test_train_sgd_and_lr_schedule():
    net, _ = _build("dfa")
    run = RunConfig(optimizer="sgd", lr=5e-3)
    ds = _dataset(8, 10, 10, 6, seed=4)
    sched = lambda e: 5e-3 * (0.5 ** e)
    trained, hist = train(net, ds, ds, run, epochs=3, batch_size=4, seed=0,
                          lr_schedule=sched)
    assert [round(m.lr, 8) for m in hist.epochs] == [5e-3, 2.5e-3, 1.25e-3]
    assert all(np.isfinite(m.loss) for m in hist.epochs)


# --------------------------------------------------------------------------- #
# 3. evaluate consistency                                                      #
# --------------------------------------------------------------------------- #
def test_evaluate_matches_manual_argmax():
    net, _ = _build("bptt")
    ds = _dataset(9, 12, 10, 6, seed=6)
    acc, firing = evaluate(net, ds, batch_size=4)
    assert 0.0 <= acc <= 1.0 and 0.0 <= firing <= 1.0

    xb, y = _stack_all(ds)
    mv = network.predict_batch(net, xb)
    pred = np.asarray(jnp.argmax(mv, axis=1))
    manual_acc = float(np.mean(pred == np.asarray(y)))
    assert abs(acc - manual_acc) < 1e-12


# --------------------------------------------------------------------------- #
# 4. io: model + run-dir round-trip                                            #
# --------------------------------------------------------------------------- #
def test_io_model_and_run_roundtrip(tmp_path):
    net, _ = _build("dfa")
    run = RunConfig(optimizer="adam", lr=1e-2, dropout_rate=0.1)
    ds = _dataset(8, 12, 10, 6, seed=1)
    trained, hist = train(net, ds, ds, run, epochs=2, batch_size=4, seed=0)

    run_dir = io.write_run(tmp_path / "run", run, hist, trained)
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "metrics.csv").is_file()
    assert (run_dir / "model.npz").is_file()

    # config.json round-trips + metrics.csv shape/headers
    config, rows = io.load_run(run_dir)
    assert config["credit_method"] == "dfa"
    assert config["dims"] == [10, [8], 6]
    assert config["run"]["lr"] == 1e-2
    assert [k for k in rows[0]] == list(METRIC_FIELDS)
    assert len(rows) == 2 and rows[0]["epoch"] == 0

    # model round-trips bit-for-bit (weights + intrinsics + B) and preserves behavior
    reloaded = io.load_model(run_dir / "model.npz")
    for a, b in zip(jax.tree_util.tree_leaves(trained.trainable()),
                    jax.tree_util.tree_leaves(reloaded.trainable())):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    for la, lb in zip(trained.hidden, reloaded.hidden):
        np.testing.assert_array_equal(np.asarray(la.T_p), np.asarray(lb.T_p))
    for ba, bb in zip(trained.B, reloaded.B):
        np.testing.assert_array_equal(np.asarray(ba), np.asarray(bb))
    assert reloaded.credit_method == "dfa" and reloaded.config == trained.config

    x0, _ = ds[0]
    np.testing.assert_array_equal(
        np.asarray(network.predict(trained, jnp.asarray(x0))),
        np.asarray(network.predict(reloaded, jnp.asarray(x0))),
    )


# --------------------------------------------------------------------------- #
# 5. figures import neither JAX nor dendroprop                                 #
# --------------------------------------------------------------------------- #
def test_figures_import_is_jax_and_dendroprop_free():
    code = (
        "import fig_accuracy, sys; "
        "assert 'jax' not in sys.modules, 'jax leaked'; "
        "assert not any(m == 'dendroprop' or m.startswith('dendroprop.') "
        "for m in sys.modules), 'dendroprop leaked'; "
        "print('ok')"
    )
    r = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PYTHONPATH": str(_ROOT / "figures")},
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"stdout={r.stdout!r} stderr={r.stderr!r}"


def test_figures_load_run_reads_io_output(tmp_path):
    # cross-check: the figures' stdlib reader parses what io.write_run produced
    import importlib.util

    net, _ = _build("bptt")
    run = RunConfig(optimizer="adam", lr=1e-2)
    ds = _dataset(6, 10, 10, 6, seed=0)
    trained, hist = train(net, ds, ds, run, epochs=2, batch_size=3, seed=0)
    run_dir = io.write_run(tmp_path / "run", run, hist, trained)

    spec = importlib.util.spec_from_file_location(
        "fig_accuracy", _ROOT / "figures" / "fig_accuracy.py"
    )
    fig = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fig)
    config, rows = fig.load_run(run_dir)
    assert len(rows) == 2
    assert config["credit_method"] == "bptt"
    assert isinstance(rows[0]["test_acc"], float) and rows[0]["epoch"] == 0


# =========================================================================== #
# 6. Hardening from adversarial review                                         #
# =========================================================================== #
def test_train_multistep_pins_key_protocol_and_adam_threading():
    """Bit-for-bit over 2 epochs x 2 batches with dropout>0 — pins the per-batch key
    protocol (fold_in(epoch_key, b), inert at dropout=0) AND cross-epoch Adam-state
    threading (both survive the single-epoch match test)."""
    C, T, N, J = 10, 12, 4, 6
    cfg = NeuronConfig(weight_scale=1.0)
    run = RunConfig(optimizer="adam", lr=1e-2, dropout_rate=0.3)   # dropout ON
    ds = _dataset(N, T, C, J, seed=11)
    net0 = build_network(random.PRNGKey(0), (C, [8], J), cfg, "bptt")

    trained, _ = train(net0, ds, ds, run, epochs=2, batch_size=2, seed=0, shuffle=False)

    # manual replication threading Adam state across both epochs
    xs = [ds[i][0] for i in range(N)]
    ys = [ds[i][1] for i in range(N)]
    net, opt, h = net0, adam_init(net0.trainable()), hypers_from_run(run)
    base = random.PRNGKey(0)
    for e in range(2):
        epoch_key = random.fold_in(base, e)
        for b, idx in enumerate([[0, 1], [2, 3]]):           # shuffle=False order
            xb = jnp.asarray(np.stack([xs[i] for i in idx]))
            sm = smooth_targets(jnp.asarray([ys[i] for i in idx]), J,
                                cfg.loss_label_smoothing)
            keys = random.split(random.fold_in(epoch_key, b), 2)
            mv, accum = network.forward_accumulate_batch(net, xb, keys, run.dropout_rate)
            _, _, grads_b = network.loss_and_grads_batch(net, mv, accum, sm, T)
            grads = jax.tree.map(lambda g: jnp.mean(g, axis=0), grads_b)
            params, opt = adam_step(net.trainable(), grads, opt, h)
            net = net.replace_trainable(params)
    assert int(opt.step) == 4                                 # 2 epochs * 2 batches

    for a, b in zip(jax.tree_util.tree_leaves(trained.trainable()),
                    jax.tree_util.tree_leaves(net.trainable())):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_metric_column_identity():
    """The right metric lands in the right column (swaps must be caught)."""
    net, _ = _build("bptt", dendritic_credit=False)   # pruned -> grad_d == 0
    run = RunConfig(optimizer="adam", lr=1e-2)
    train_ds = _dataset(12, 12, 10, 6, seed=2)
    test_ds = _dataset(6, 12, 10, 6, seed=99)          # distinct distribution
    trained, hist = train(net, train_ds, test_ds, run, epochs=2, batch_size=4, seed=0)

    m = hist.last
    tr_acc, tr_fire = evaluate(trained, train_ds, 4)
    te_acc, _ = evaluate(trained, test_ds, 4, want_firing=False)
    assert m.train_acc == tr_acc and m.test_acc == te_acc      # correct columns
    assert m.firing == tr_fire
    assert m.gnorm_d == 0.0                                     # pruned -> dend col is 0
    assert m.gnorm_r > 0.0 and m.gnorm_s > 0.0                  # other cols non-zero


def test_train_rejects_unknown_optimizer_before_any_batch():
    net, _ = _build("bptt")
    ds = _dataset(4, 10, 10, 6, seed=0)
    empty = _dataset(0, 10, 10, 6, seed=0)
    run = RunConfig(optimizer="rmsprop")
    with pytest.raises(ValueError):
        train(net, empty, empty, run, epochs=1, batch_size=3)   # 0 batches
    with pytest.raises(ValueError):
        train(net, ds, ds, run, epochs=0, batch_size=3)         # 0 epochs


def test_io_multilayer_and_dtype_and_no_net(tmp_path):
    # multilayer save/load (per-layer loop + 2 feedback matrices) + int T_p dtype
    net = build_network(random.PRNGKey(1), (10, [8, 6], 5),
                        NeuronConfig(weight_scale=1.0), "dfa")
    path = io.save_model(tmp_path / "m.npz", net)
    reloaded = io.load_model(path)
    assert len(reloaded.hidden) == 2 and len(reloaded.B) == 2
    for a, b in zip(jax.tree_util.tree_leaves(net.trainable()),
                    jax.tree_util.tree_leaves(reloaded.trainable())):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    for ba, bb in zip(net.B, reloaded.B):
        np.testing.assert_array_equal(np.asarray(ba), np.asarray(bb))
    for l in reloaded.hidden:                                   # int T_p contract
        assert np.issubdtype(np.asarray(l.T_p).dtype, np.integer)

    # write_run(net=None) -> config.json (run only) + metrics.csv, NO model.npz
    run = RunConfig(optimizer="sgd", lr=1e-3)
    hist = MetricsHistory()
    run_dir = io.write_run(tmp_path / "norun", run, hist, net=None)
    assert (run_dir / "config.json").is_file() and (run_dir / "metrics.csv").is_file()
    assert not (run_dir / "model.npz").exists()
    config, rows = io.load_run(run_dir)
    assert list(config.keys()) == ["run"] and rows == []


def test_run_shd_synthetic_subprocess(tmp_path):
    """End-to-end runner smoke (x64-off production regime) incl. a mapped preset."""
    out = tmp_path / "run"
    r = subprocess.run(
        [sys.executable, str(_ROOT / "experiments" / "run_shd.py"),
         "--synthetic", "16", "--hidden", "8", "--epochs", "1",
         "--batch-size", "8", "--rule", "ste", "--run-dir", str(out)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"stdout={r.stdout!r} stderr={r.stderr!r}"
    assert (out / "config.json").is_file() and (out / "model.npz").is_file()
    config, rows = io.load_run(out)
    assert len(rows) == 1 and np.isfinite(rows[0]["loss"])
