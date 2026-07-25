"""optim.py (hand-rolled AdamW/SGD) + the trainable() partition.

The optimizer applies the grads that credit.py/loss_and_grads compute; the golden is
untouched. This checks:
  - adam_step / sgd_step leaf-by-leaf against an independent numpy recompute of the
    AdamW/SGD update (clip-first, ascent, decoupled AdamW decay; step=1 first);
  - the moment tree mirrors Network.trainable() and the fixed constants
    (T_p/alpha_*/B/config) are structurally absent from the moment leaves;
  - batch-mean == jnp.mean(axis=0); replace_trainable(trainable(net)) == net; the trainable
    leaves are exactly {w_dend_L, w_soma_L, w_readout};
  - an end-to-end DFA-multilayer smoke: build_network -> forward_accumulate_batch ->
    loss_and_grads_batch -> tree.map(mean) -> adam_step moves the trainable weights while
    the frozen intrinsics (T_p/alpha_*/B/config) stay byte-identical, loss finite.
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import pytest

from dendroprop import network
from dendroprop.config import NeuronConfig, RunConfig
from dendroprop.state import TrainableParams
from dendroprop.network import build_network
from dendroprop.optim import (
    AdamState, AdamHypers, adam_init, adam_step, sgd_step, hypers_from_run,
)


def _eq(a, b):
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def _differs(a, b):
    assert not np.allclose(np.asarray(a), np.asarray(b))


def _mk_params(seed):
    """A 2-hidden-layer-shaped TrainableParams with distinct per-leaf shapes."""
    k = random.split(random.PRNGKey(seed), 5)
    return TrainableParams(
        w_dend=(random.normal(k[0], (3, 4)), random.normal(k[1], (2, 3))),
        w_soma=(random.normal(k[2], (3, 4)), random.normal(k[3], (2, 3))),
        w_readout=random.normal(k[4], (5, 2)),
    )


# ── adam_step / sgd_step leaf-by-leaf vs independent numpy recompute ───────────

def _check_adam_leaves(p_in, grads, s_in, p_out, s_out, step, h):
    leaves = zip(
        jax.tree_util.tree_leaves(p_in), jax.tree_util.tree_leaves(grads),
        jax.tree_util.tree_leaves(s_in.m), jax.tree_util.tree_leaves(s_in.v),
        jax.tree_util.tree_leaves(p_out),
        jax.tree_util.tree_leaves(s_out.m), jax.tree_util.tree_leaves(s_out.v),
    )
    for w, g, m0, v0, w1, m1, v1 in leaves:
        w, g, m0, v0 = (np.asarray(a) for a in (w, g, m0, v0))
        gc = np.clip(g, -h.clip_value, h.clip_value)          # clip FIRST
        m = h.beta1 * m0 + (1 - h.beta1) * gc
        v = h.beta2 * v0 + (1 - h.beta2) * gc ** 2
        m_hat = m / (1 - h.beta1 ** step)
        v_hat = v / (1 - h.beta2 ** step)
        w_exp = w + h.lr * m_hat / (np.sqrt(v_hat) + h.eps) - h.lr * h.weight_decay * w  # ascent + decoupled decay
        np.testing.assert_allclose(np.asarray(w1), w_exp, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(np.asarray(m1), m, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(np.asarray(v1), v, rtol=1e-12, atol=1e-14)


def test_adam_step_leafwise_vs_numpy_two_steps():
    params = _mk_params(1)
    grads = jax.tree.map(lambda a: a * 3.0, _mk_params(2))   # scaled up so the clip bites
    h = AdamHypers(lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.1, clip_value=0.5)
    assert any(np.any(np.abs(np.asarray(l)) > h.clip_value)
               for l in jax.tree_util.tree_leaves(grads))  # precondition: the clip actually bites
    state = adam_init(params)
    assert int(state.step) == 0 and state.step.dtype == jnp.int32
    p1, s1 = adam_step(params, grads, state, h)              # step -> 1 (bias correction uses 1)
    _check_adam_leaves(params, grads, state, p1, s1, 1, h)
    p2, s2 = adam_step(p1, grads, s1, h)                     # step -> 2, moments carry
    _check_adam_leaves(p1, grads, s1, p2, s2, 2, h)
    assert int(s1.step) == 1 and int(s2.step) == 2


def test_sgd_step_leafwise_vs_numpy():
    params = _mk_params(3)
    grads = jax.tree.map(lambda a: a * 4.0, _mk_params(4))   # clip active
    h = AdamHypers(lr=0.05, weight_decay=0.2, clip_value=0.3)
    assert any(np.any(np.abs(np.asarray(l)) > h.clip_value)
               for l in jax.tree_util.tree_leaves(grads))  # precondition: the clip actually bites
    out = sgd_step(params, grads, h)
    for w, g, w1 in zip(jax.tree_util.tree_leaves(params),
                        jax.tree_util.tree_leaves(grads),
                        jax.tree_util.tree_leaves(out)):
        w, g = np.asarray(w), np.asarray(g)
        w_exp = w + h.lr * np.clip(g, -h.clip_value, h.clip_value) - h.lr * h.weight_decay * w
        np.testing.assert_allclose(np.asarray(w1), w_exp, rtol=1e-12, atol=1e-14)


def test_hypers_from_run_maps_run_config():
    run = RunConfig(optimizer="adam", lr=2e-3, clip_value=0.7, weight_decay=0.01,
                    beta1=0.8, beta2=0.99, adam_eps=1e-7)
    h = hypers_from_run(run)
    assert (h.lr, h.beta1, h.beta2, h.eps, h.weight_decay, h.clip_value) == \
        (2e-3, 0.8, 0.99, 1e-7, 0.01, 0.7)
    assert float(hypers_from_run(run, lr=9e-9).lr) == 9e-9   # schedulable lr override


# ── moment tree mirrors trainable(); intrinsics absent from moment leaves ──────

def _net(method="dfa", hidden=(16, 12)):
    return build_network(random.PRNGKey(11), (20, list(hidden), 6),
                         NeuronConfig(weight_scale=1.0), method)


def test_moment_tree_mirrors_trainable_and_excludes_intrinsics():
    net = _net()
    tp = net.trainable()
    opt = adam_init(tp)
    assert jax.tree_util.tree_structure(opt.m) == jax.tree_util.tree_structure(tp)
    assert jax.tree_util.tree_structure(opt.v) == jax.tree_util.tree_structure(tp)
    # trainable leaves are EXACTLY the 2*L+1 weight matrices — no int32 T_p, no scalar
    # alpha_*, no B, no config.
    leaves = jax.tree_util.tree_leaves(tp)
    assert len(leaves) == 2 * len(net.hidden) + 1
    for lf in leaves:
        assert lf.ndim == 2 and jnp.issubdtype(lf.dtype, jnp.floating)
    for lf in jax.tree_util.tree_leaves(opt.m):
        assert jnp.issubdtype(lf.dtype, jnp.floating)   # moments float, no int32 T_p leaked


def test_trainable_leaves_are_exactly_the_weights():
    net = _net(hidden=(16, 12))
    tp = net.trainable()
    got = [tuple(lf.shape) for lf in jax.tree_util.tree_leaves(tp)]
    want = [(16, 20), (12, 16), (16, 20), (12, 16), (6, 12)]  # wd0, wd1, ws0, ws1, w_readout
    assert got == want


# ── replace_trainable(trainable(net)) == net; partition round-trips ────────────

def test_replace_trainable_is_identity_and_static_preserved():
    net = _net()
    net2 = net.replace_trainable(net.trainable())
    assert jax.tree_util.tree_structure(net2) == jax.tree_util.tree_structure(net)
    for a, b in zip(jax.tree_util.tree_leaves(net), jax.tree_util.tree_leaves(net2)):
        _eq(a, b)
    assert net2.credit_method == net.credit_method
    # a MODIFIED trainable round-trips into the weights only, intrinsics untouched
    doubled = jax.tree.map(lambda w: w * 2.0, net.trainable())
    net3 = net.replace_trainable(doubled)
    _eq(net3.hidden[0].w_soma, net.hidden[0].w_soma * 2.0)
    _eq(net3.hidden[0].T_p, net.hidden[0].T_p)          # intrinsic untouched
    _eq(net3.B[0], net.B[0])                            # feedback untouched


# ── batch-mean == jnp.mean(axis=0) over the grad tree ──────────────────────────

def test_batch_mean_is_axis0_mean():
    B = 5
    k = random.split(random.PRNGKey(7), 5)
    grads_b = TrainableParams(
        w_dend=(random.normal(k[0], (B, 3, 4)),),
        w_soma=(random.normal(k[1], (B, 3, 4)),),
        w_readout=random.normal(k[2], (B, 5, 3)),
    )
    mean = jax.tree.map(lambda g: jnp.mean(g, axis=0), grads_b)
    # Independent reference: reduce over the leading BATCH axis by hand (sum / B), NOT
    # jnp.mean again — so a wrong-axis convention (e.g. axis=1) is actually caught rather
    # than re-derived from the same expression.
    ref_r = np.add.reduce(np.asarray(grads_b.w_readout), axis=0) / B
    ref_s = np.add.reduce(np.asarray(grads_b.w_soma[0]), axis=0) / B
    np.testing.assert_allclose(np.asarray(mean.w_readout), ref_r, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(np.asarray(mean.w_soma[0]), ref_s, rtol=1e-12, atol=1e-14)
    assert mean.w_readout.shape == (5, 3) and mean.w_soma[0].shape == (3, 4)


# ── end-to-end DFA-multilayer smoke: adam moves weights, intrinsics frozen ─────

def test_end_to_end_dfa_multilayer_adam_smoke():
    c = NeuronConfig(weight_scale=1.0)   # scale that reliably spikes under x64
    run = RunConfig(optimizer="adam", lr=1e-2, dropout_rate=0.0)
    K, J, T, Bn = 20, 6, 25, 4
    net = build_network(random.PRNGKey(0), (K, [16, 12], J), c, "dfa")

    before = net.trainable()
    T_p_before = [np.asarray(l.T_p) for l in net.hidden]
    B_before = [np.asarray(b) for b in net.B]

    xb = (random.uniform(random.PRNGKey(1), (Bn, T, K)) < 0.5).astype(jnp.float64)
    labels = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
    sm = network.smooth_targets(labels, J, c.loss_label_smoothing)

    opt = adam_init(net.trainable())
    h = hypers_from_run(run)
    loss = None
    for i in range(3):
        keys = random.split(random.PRNGKey(100 + i), Bn)
        mv, accum = network.forward_accumulate_batch(net, xb, keys, run.dropout_rate)
        loss, _, grads_b = network.loss_and_grads_batch(net, mv, accum, sm, T)
        grads = jax.tree.map(lambda g: jnp.mean(g, axis=0), grads_b)
        new_params, opt = adam_step(net.trainable(), grads, opt, h)
        net = net.replace_trainable(new_params)

    assert bool(jnp.all(jnp.isfinite(jnp.asarray(loss))))
    after = net.trainable()
    _differs(after.w_soma[0], before.w_soma[0])   # bottom-layer weights moved
    _differs(after.w_readout, before.w_readout)
    assert int(opt.step) == 3
    # Frozen intrinsics byte-identical after training.
    for l, tpb in zip(net.hidden, T_p_before):
        _eq(l.T_p, tpb)
    for b_new, b_old in zip(net.B, B_before):
        _eq(b_new, b_old)
    assert net.config == c
    assert net.credit_method == "dfa"
