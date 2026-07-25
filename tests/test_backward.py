"""The eligibility scan reproduces the golden.

Checks the training forward pass in ``network.py`` and the ``EligState`` struct against
the frozen reference: the training ``mean_voltage`` and the (J,N,.) accumulators
``A_r/A_s/A_d`` (frozen as ``aux_A_*``), single & batched, dropout off & on.

``forward_accumulate`` returns a ``TrainAccum`` — the shared
readout accumulator ``G_r`` plus a per-hidden-layer ``credit`` tuple of ``LayerCredit``
(``G_s``/``G_d``). For the single hidden layer this gate reconstructs the golden's
``aux_A_*`` from that accumulator — ``A_r = broadcast(G_r)``, ``A_s = w_readout ⊙
credit[0].G_s``, ``A_d = w_readout ⊙ credit[0].G_d`` — pinning the exact factorization so
the collapse is shown to drop nothing. That reconstruction is an independent path from
``loss_and_grads`` (which contracts via ``e_n = δ @ M``), so it is not circular. The
w_readout-factoring reassociates the sums, and the input projection now happens inside the
scan, so this is a within-tol check (rtol=1e-9), not bit-for-bit.
"""
import json
from pathlib import Path

import numpy as np
import jax.numpy as jnp

from dendroprop.config import NeuronConfig
from dendroprop.state import EligState
from dendroprop import network
from dendroprop.layers import TwoCompLayer, ReadoutLayer
from dendroprop.network import Network


def _net(w_dend, w_soma, w_readout, alpha_s, alpha_d, alpha_m, T_p, config, alpha_w):
    """Assemble a single-hidden-layer bptt Network from the golden params."""
    layer = TwoCompLayer(
        w_dend=w_dend, w_soma=w_soma, T_p=T_p,
        alpha_s=alpha_s, alpha_d=alpha_d, alpha_w=alpha_w,
    )
    return Network(hidden=(layer,), readout=ReadoutLayer(w=w_readout, alpha_m=alpha_m), config=config)


def _fa(x, w_dend, w_soma, w_readout, alpha_s, alpha_d, alpha_m, T_p, config, alpha_w, key, rate):
    net = _net(w_dend, w_soma, w_readout, alpha_s, alpha_d, alpha_m, T_p, config, alpha_w)
    return network.forward_accumulate(net, x, key, rate)


def _fa_batch(x, w_dend, w_soma, w_readout, alpha_s, alpha_d, alpha_m, T_p, config, alpha_w, keys, rate):
    net = _net(w_dend, w_soma, w_readout, alpha_s, alpha_d, alpha_m, T_p, config, alpha_w)
    return network.forward_accumulate_batch(net, x, keys, rate)

_GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
_G = np.load(_GOLDEN_DIR / "golden.npz", allow_pickle=False)
_M = json.loads((_GOLDEN_DIR / "manifest.json").read_text())
_RTOL = _M["tolerance"]["rtol"]
_ATOL = _M["tolerance"]["atol"]


def _inp(name):
    return _G[f"input__{name}"]


def _config():
    # manifest config has no soma_grad_scale (captured pre-refactor); it defaults.
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


def _assert_aux(tag, accum, w_readout):
    """Reconstruct the (J,N,.) accumulators from the J-less TrainAccum (single hidden
    layer: the sole ``credit[0]`` LayerCredit + the shared ``G_r``)."""
    J, N = w_readout.shape
    A_r = jnp.broadcast_to(accum.G_r, (J, N))
    A_s = w_readout[:, :, None] * accum.credit[0].G_s[None, :, :]
    A_d = w_readout[:, :, None] * accum.credit[0].G_d[None, :, :]
    np.testing.assert_allclose(np.asarray(A_r), _G[f"{tag}__aux_A_r"], rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(np.asarray(A_s), _G[f"{tag}__aux_A_s"], rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(np.asarray(A_d), _G[f"{tag}__aux_A_d"], rtol=_RTOL, atol=_ATOL)


def test_eligstate_shapes_dtypes():
    """The four eligibility traces are carried at float64, batch-aware."""
    n, k = 6, 8
    es = EligState.zeros(n, k)
    assert es.eps_s.shape == (k,)
    assert es.eps_d.shape == (n, k)
    assert es.eps_d_bar.shape == (n, k)
    assert es.eps_r.shape == (n,)
    for field in (es.eps_s, es.eps_d, es.eps_d_bar, es.eps_r):
        assert field.dtype == jnp.float64  # x64 regime (conftest)
    esb = EligState.zeros(n, k, batch=3)
    assert esb.eps_s.shape == (3, k)
    assert esb.eps_d.shape == (3, n, k)
    assert esb.eps_r.shape == (3, n)


def test_forward_accum_single_off__mean_voltage():
    mv, *_ = _fa(
        jnp.asarray(_inp("x_single")), *_weights(), *_params(_config()), _key_single(), 0.0,
    )
    np.testing.assert_allclose(
        np.asarray(mv), _G["train_single_off__mean_voltage"], rtol=_RTOL, atol=_ATOL,
    )


def test_forward_accum_single_off__aux():
    _, accum = _fa(
        jnp.asarray(_inp("x_single")), *_weights(), *_params(_config()), _key_single(), 0.0,
    )
    _assert_aux("train_single_off", accum, _weights()[2])


def test_forward_accum_single_on__mean_voltage():
    mv, *_ = _fa(
        jnp.asarray(_inp("x_single")), *_weights(), *_params(_config()), _key_single(), _drop_rate(),
    )
    np.testing.assert_allclose(
        np.asarray(mv), _G["train_single_on__mean_voltage"], rtol=_RTOL, atol=_ATOL,
    )


def test_forward_accum_single_on__aux():
    _, accum = _fa(
        jnp.asarray(_inp("x_single")), *_weights(), *_params(_config()), _key_single(), _drop_rate(),
    )
    _assert_aux("train_single_on", accum, _weights()[2])


def test_forward_accum_batch_off__mean_voltage():
    mv, *_ = _fa_batch(
        jnp.asarray(_inp("x_batch")), *_weights(), *_params(_config()), _keys_batch(), 0.0,
    )
    np.testing.assert_allclose(
        np.asarray(mv), _G["train_batch_off__mean_voltage"], rtol=_RTOL, atol=_ATOL,
    )


def test_forward_accum_batch_on__mean_voltage():
    mv, *_ = _fa_batch(
        jnp.asarray(_inp("x_batch")), *_weights(), *_params(_config()), _keys_batch(), _drop_rate(),
    )
    np.testing.assert_allclose(
        np.asarray(mv), _G["train_batch_on__mean_voltage"], rtol=_RTOL, atol=_ATOL,
    )
