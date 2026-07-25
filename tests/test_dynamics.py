"""``dynamics.py`` reproduces the golden.

Checks ``dynamics.py`` + ``state.py`` against the frozen reference: the predict-only
``mean_voltage`` (single & batch), the argmax predictions, and the ``activity`` firing
rate. Training-path outputs (loss, grads) are checked in test_backward/test_grads.
"""
import json
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import pytest

from dendroprop.config import NeuronConfig
from dendroprop.state import DynamicsState
from dendroprop import dynamics

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


def test_dynamics_state_int32_dtypes():
    """R3 groundwork: plateau flags stay int32 in the carry."""
    st = DynamicsState.zeros(5)
    assert st.h.dtype == jnp.int32
    assert st.t_prime.dtype == jnp.int32
    assert st.mu.dtype == jnp.float64  # x64 regime (conftest)
    stb = DynamicsState.zeros(5, batch=3)
    assert stb.h.shape == (3, 5) and stb.h.dtype == jnp.int32


def test_soma_grad_scale_default():
    """The field exists and defaults to the empirical 8.0."""
    assert _config().soma_grad_scale == 8.0


def test_predict_single_mean_voltage():
    mv = dynamics.predict(jnp.asarray(_inp("x_single")), *_weights(), *_params(_config()))
    np.testing.assert_allclose(
        np.asarray(mv), _G["predict_single__mean_voltage"], rtol=_RTOL, atol=_ATOL,
    )
    assert int(jnp.argmax(mv)) == int(_G["predict_single__pred"])


def test_predict_batch_mean_voltage():
    mv = dynamics.predict_batch(jnp.asarray(_inp("x_batch")), *_weights(), *_params(_config()))
    np.testing.assert_allclose(
        np.asarray(mv), _G["predict_batch__mean_voltage"], rtol=_RTOL, atol=_ATOL,
    )
    np.testing.assert_array_equal(
        np.asarray(jnp.argmax(mv, axis=1)), _G["predict_batch__pred"],
    )


def test_activity_batch_rate():
    rates = dynamics.activity_batch(jnp.asarray(_inp("x_batch")), *_weights(), *_params(_config()))
    np.testing.assert_allclose(
        float(jnp.mean(rates)), float(_G["activity_batch__rate"]), rtol=_RTOL, atol=_ATOL,
    )


def test_predict_matches_dropout_off_training_voltage():
    """Cross-check: inference voltage equals the golden's dropout-off train voltage."""
    mv = dynamics.predict(jnp.asarray(_inp("x_single")), *_weights(), *_params(_config()))
    np.testing.assert_allclose(
        np.asarray(mv), _G["train_single_off__mean_voltage"], rtol=_RTOL, atol=_ATOL,
    )
