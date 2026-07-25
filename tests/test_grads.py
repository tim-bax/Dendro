"""The J-collapsed grad assembly reproduces the golden.

Checks ``network.loss_and_grads`` (+ ``forward_accumulate`` returning a J-less
``TrainAccum``) against the frozen reference: ``loss``, ``pred`` and the grad tree
``TrainableParams`` (``w_readout`` / ``w_soma[0]`` / ``w_dend[0]`` for the single hidden
layer), single & batch, dropout off & on, plus the averaged batch grads. The J-collapse
(e_n = delta @ w_readout applied once post-scan) reassociates the sums, and the input
projection happens inside the scan, so this is a *within-tol* check (rtol=1e-9,
atol=1e-11), not bit-for-bit — the frozen tolerance has ~5 orders of margin.

``mean_voltage`` is not re-asserted here (test_backward covers it); the batched
accumulators — never frozen standalone — are checked end-to-end here through the
batched grads.
"""
import json
from pathlib import Path

import numpy as np
import jax.numpy as jnp

from dendroprop.config import NeuronConfig
from dendroprop.state import LayerCredit, TrainAccum
from dendroprop import network
from dendroprop.layers import TwoCompLayer, ReadoutLayer
from dendroprop.network import Network


def _net(config):
    """Single-hidden-layer bptt Network from the golden params."""
    w_dend, w_soma, w_readout = _weights()
    alpha_s, alpha_d, alpha_m, T_p, _, alpha_w = _params(config)
    layer = TwoCompLayer(
        w_dend=w_dend, w_soma=w_soma, T_p=T_p,
        alpha_s=alpha_s, alpha_d=alpha_d, alpha_w=alpha_w,
    )
    return Network(hidden=(layer,), readout=ReadoutLayer(w=w_readout, alpha_m=alpha_m), config=config)


def _dummy_net(w_readout, N, K, config):
    """Net whose loss_and_grads uses only readout.w/config (for the argmax test)."""
    z = jnp.zeros((N, K))
    layer = TwoCompLayer(
        w_dend=z, w_soma=z, T_p=jnp.zeros(N, jnp.int32),
        alpha_s=jnp.asarray(0.0), alpha_d=jnp.asarray(0.0), alpha_w=jnp.asarray(0.0),
    )
    return Network(hidden=(layer,), readout=ReadoutLayer(w=w_readout, alpha_m=jnp.asarray(0.0)), config=config)


def _accum_zeros(N, K, batch=None):
    """A zero single-layer TrainAccum (shared G_r + one LayerCredit)."""
    G_r = jnp.zeros((batch, N)) if batch else jnp.zeros(N)
    return TrainAccum(G_r=G_r, credit=(LayerCredit.zeros(N, K, batch=batch),))

_GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
_G = np.load(_GOLDEN_DIR / "golden.npz", allow_pickle=False)
_M = json.loads((_GOLDEN_DIR / "manifest.json").read_text())
_RTOL = _M["tolerance"]["rtol"]
_ATOL = _M["tolerance"]["atol"]


def _inp(name):
    return _G[f"input__{name}"]


def _config():
    # manifest config has no soma_grad_scale (captured pre-refactor); it defaults to 8.0.
    return NeuronConfig(**_M["config"])


def _weights():
    return (
        jnp.asarray(_inp("w_dend")),
        jnp.asarray(_inp("w_soma")),
        jnp.asarray(_inp("w_readout")),
    )


def _params(config):
    return (
        jnp.asarray(_inp("alpha_s")),
        jnp.asarray(_inp("alpha_d")),
        jnp.asarray(_inp("alpha_m")),
        jnp.asarray(_inp("T_p")).astype(jnp.int32),
        config,
        jnp.asarray(_inp("alpha_w")),
    )


def _drop_rate():
    return float(_inp("dropout_rate"))


def _key_single():
    return jnp.asarray(_inp("dropout_key_single")).astype(jnp.uint32)


def _keys_batch():
    return jnp.asarray(_inp("dropout_keys_batch")).astype(jnp.uint32)


def _n_classes():
    return _weights()[2].shape[0]  # J = w_readout.shape[0]


def _sm_single():
    c = _config()
    return network.smooth_targets(int(_inp("y_single")), _n_classes(), c.loss_label_smoothing)


def _sm_batch():
    c = _config()
    labels = jnp.asarray(_inp("y_batch")).astype(jnp.int32)
    return network.smooth_targets(labels, _n_classes(), c.loss_label_smoothing)


def _run_single(rate, key):
    c = _config()
    net = _net(c)
    T = _inp("x_single").shape[0]
    mv, accum = network.forward_accumulate(net, jnp.asarray(_inp("x_single")), key, rate)
    return network.loss_and_grads(net, mv, accum, _sm_single(), T)


def _run_batch(rate, keys):
    c = _config()
    net = _net(c)
    T = _inp("x_batch").shape[1]
    mv, accum = network.forward_accumulate_batch(net, jnp.asarray(_inp("x_batch")), keys, rate)
    return network.loss_and_grads_batch(net, mv, accum, _sm_batch(), T)


def _grad_leaves(grads):
    """Single-hidden-layer view of the TrainableParams grad tree -> (g_r, g_s, g_d)."""
    return grads.w_readout, grads.w_soma[0], grads.w_dend[0]


def _assert_grads(tag, g_r, g_s, g_d):
    np.testing.assert_allclose(np.asarray(g_r), _G[f"{tag}__grad_r"], rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(np.asarray(g_s), _G[f"{tag}__grad_s"], rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(np.asarray(g_d), _G[f"{tag}__grad_d"], rtol=_RTOL, atol=_ATOL)


def test_layercredit_shapes_dtypes():
    """LayerCredit carries the J-less per-layer accumulators at float64, batch-aware;
    the shared G_r rides TrainAccum with the top-layer width."""
    n, k = 6, 8
    lc = LayerCredit.zeros(n, k)
    assert lc.G_s.shape == (n, k)
    assert lc.G_d.shape == (n, k)
    for field in (lc.G_s, lc.G_d):
        assert field.dtype == jnp.float64  # x64 regime (conftest)
    lcb = LayerCredit.zeros(n, k, batch=3)
    assert lcb.G_s.shape == (3, n, k)
    ta = TrainAccum(G_r=jnp.zeros(n), credit=(lc,))
    assert ta.G_r.shape == (n,)
    assert ta.G_r.dtype == jnp.float64
    assert len(ta.credit) == 1


def test_prediction_tracks_argmax():
    """pred = argmax(mean_voltage), pinned independently of the golden.

    Every frozen golden scenario happens to predict class 2, so the golden pred asserts
    can't tell a correct argmax from a constant/argmin/mis-axed-vmap stub. These crafted
    voltages have non-2, per-sample-distinct winners, constraining the argmax logic (and
    the batched vmap axis) directly.
    """
    c = _config()
    J, N, Kk = 4, 6, 8
    w_r = jnp.zeros((J, N))
    sm = network.smooth_targets(0, J, c.loss_label_smoothing)
    for mv, want in (([0.9, 0.1, 0.2, 0.3], 0), ([0.1, 0.2, 0.3, 0.9], 3)):
        dnet = _dummy_net(w_r, N, Kk, c)
        _, pred, *_ = network.loss_and_grads(dnet, jnp.asarray(mv), _accum_zeros(N, Kk), sm, 10)
        assert int(pred) == want
    mv_b = jnp.asarray([[0.9, 0.1, 0.2, 0.3], [0.1, 0.2, 0.3, 0.9]])
    sm_b = jnp.stack([sm, sm])
    _, preds, *_ = network.loss_and_grads_batch(
        _dummy_net(w_r, N, Kk, c), mv_b, _accum_zeros(N, Kk, batch=2), sm_b, 10)
    np.testing.assert_array_equal(np.asarray(preds), [0, 3])


def test_loss_single_off():
    loss, pred, *_ = _run_single(0.0, _key_single())
    np.testing.assert_allclose(float(loss), float(_G["train_single_off__loss"]), rtol=_RTOL, atol=_ATOL)
    assert int(pred) == int(_G["train_single_off__pred"])


def test_grads_single_off():
    g_r, g_s, g_d = _grad_leaves(_run_single(0.0, _key_single())[2])
    _assert_grads("train_single_off", g_r, g_s, g_d)


def test_loss_single_on():
    loss, pred, *_ = _run_single(_drop_rate(), _key_single())
    np.testing.assert_allclose(float(loss), float(_G["train_single_on__loss"]), rtol=_RTOL, atol=_ATOL)
    assert int(pred) == int(_G["train_single_on__pred"])


def test_grads_single_on():
    g_r, g_s, g_d = _grad_leaves(_run_single(_drop_rate(), _key_single())[2])
    _assert_grads("train_single_on", g_r, g_s, g_d)


def test_loss_batch_off():
    losses, preds, *_ = _run_batch(0.0, _keys_batch())
    np.testing.assert_allclose(np.asarray(losses), _G["train_batch_off__loss"], rtol=_RTOL, atol=_ATOL)
    np.testing.assert_array_equal(np.asarray(preds), _G["train_batch_off__pred"])


def test_grads_batch_off():
    g_r, g_s, g_d = _grad_leaves(_run_batch(0.0, _keys_batch())[2])
    _assert_grads("train_batch_off", g_r, g_s, g_d)


def test_grads_avg_batch_off():
    g_r, g_s, g_d = _grad_leaves(_run_batch(0.0, _keys_batch())[2])
    np.testing.assert_allclose(np.asarray(jnp.mean(g_r, axis=0)), _G["train_batch_off__grad_r_avg"], rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(np.asarray(jnp.mean(g_s, axis=0)), _G["train_batch_off__grad_s_avg"], rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(np.asarray(jnp.mean(g_d, axis=0)), _G["train_batch_off__grad_d_avg"], rtol=_RTOL, atol=_ATOL)


def test_loss_batch_on():
    losses, preds, *_ = _run_batch(_drop_rate(), _keys_batch())
    np.testing.assert_allclose(np.asarray(losses), _G["train_batch_on__loss"], rtol=_RTOL, atol=_ATOL)
    np.testing.assert_array_equal(np.asarray(preds), _G["train_batch_on__pred"])


def test_grads_batch_on():
    g_r, g_s, g_d = _grad_leaves(_run_batch(_drop_rate(), _keys_batch())[2])
    _assert_grads("train_batch_on", g_r, g_s, g_d)


def test_grads_avg_batch_on():
    g_r, g_s, g_d = _grad_leaves(_run_batch(_drop_rate(), _keys_batch())[2])
    np.testing.assert_allclose(np.asarray(jnp.mean(g_r, axis=0)), _G["train_batch_on__grad_r_avg"], rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(np.asarray(jnp.mean(g_s, axis=0)), _G["train_batch_on__grad_s_avg"], rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(np.asarray(jnp.mean(g_d, axis=0)), _G["train_batch_on__grad_d_avg"], rtol=_RTOL, atol=_ATOL)
