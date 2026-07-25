"""The layer/network object surface (``TwoCompLayer`` / ``ReadoutLayer`` / ``Network``).

The Network's golden checks (mean_voltage, accumulators, loss, grads, single+batch,
dropout) live in ``test_backward`` / ``test_grads``. This file covers the object-specific
concerns:

  - inference forward / activity ≈ the ``dynamics.*`` free functions, within tolerance:
    the Network runs its own N-layer sweep with the input projection inside the scan, so
    it reassociates vs the ``dynamics`` reference (which projects once outside). The
    ``dynamics`` free functions are the bit-for-bit single-layer reference
    (``test_dynamics``); here the two agree at rtol=1e-9;
  - an independent plain-loop recompute reproduces the training forward, and pins the
    per-layer surrogate wiring on both traces;
  - distinct-alpha wiring (the golden has alpha_s == alpha_d, hiding a swap);
  - the credit components by property/reduction identity: prune, routing invariance +
    reduction dfa[B=w_readout]==bptt, composition;
  - pytree staticness (str+bool leak guard, per-component treedef);
  - the surrogate / feedback / routing unit checks;
  - primitive wiring behind degenerate golden values.

Multi-hidden-layer forward/credit (N-layer DFA, coupled guards, build_network) is covered
by ``test_multilayer.py``; ``loss_and_grads`` returns a ``TrainableParams`` grad tree
(``w_readout`` / ``w_soma[i]`` / ``w_dend[i]``) and ``forward_accumulate`` a ``TrainAccum``
(``G_r`` + per-layer ``credit[i]``).
"""
import json
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import pytest

from dendroprop import network, dynamics, surrogate, credit, feedback
from dendroprop.config import NeuronConfig
from dendroprop.dynamics import dynamics_step, readout_integrate
from dendroprop.state import DynamicsState, EligState
from dendroprop.layers import (
    TwoCompLayer, ReadoutLayer,
    update_somatic_eligibility, update_dendritic_eligibility,
)
from dendroprop.network import Network

_GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
_G = np.load(_GOLDEN_DIR / "golden.npz", allow_pickle=False)
_M = json.loads((_GOLDEN_DIR / "manifest.json").read_text())
_RTOL = _M["tolerance"]["rtol"]
_ATOL = _M["tolerance"]["atol"]


def _inp(name):
    return _G[f"input__{name}"]


def _config():
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


def _key_single():
    return jnp.asarray(_inp("dropout_key_single")).astype(jnp.uint32)


def _keys_batch():
    return jnp.asarray(_inp("dropout_keys_batch")).astype(jnp.uint32)


def _n_classes():
    return _weights()[2].shape[0]  # J


def _sm_single():
    c = _config()
    return network.smooth_targets(int(_inp("y_single")), _n_classes(), c.loss_label_smoothing)


def _sm_batch():
    c = _config()
    labels = jnp.asarray(_inp("y_batch")).astype(jnp.int32)
    return network.smooth_targets(labels, _n_classes(), c.loss_label_smoothing)


def _run_net_batch(net, rate=0.0):
    """Run the batched Network closures -> single-hidden-layer view
    (loss, pred, grad_r, grad_s, grad_d)."""
    xb, kb = jnp.asarray(_inp("x_batch")), _keys_batch()
    T = xb.shape[1]
    mv, accum = network.forward_accumulate_batch(net, xb, kb, rate)
    loss, pred, grads = network.loss_and_grads_batch(net, mv, accum, _sm_batch(), T)
    return loss, pred, grads.w_readout, grads.w_soma[0], grads.w_dend[0]


def _golden_network():
    """A single-hidden-layer BPTT ``Network`` built from the frozen golden arrays."""
    c = _config()
    w_dend, w_soma, w_readout = _weights()
    alpha_s, alpha_d, alpha_m, T_p, _, alpha_w = _params(c)
    layer = TwoCompLayer(
        w_dend=w_dend, w_soma=w_soma, T_p=T_p,
        alpha_s=alpha_s, alpha_d=alpha_d, alpha_w=alpha_w,
    )
    return Network(hidden=(layer,), readout=ReadoutLayer(w=w_readout, alpha_m=alpha_m), config=c)


def _feedback_B():
    """Test OWNS B (the Network must not generate it). (J,N) != w_readout."""
    J, N = _weights()[2].shape
    return random.normal(random.PRNGKey(20260723), (J, N))


def _eq(a, b):
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def _close(a, b):
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=_RTOL, atol=_ATOL)


def _differs(a, b):
    assert not np.allclose(np.asarray(a), np.asarray(b), rtol=_RTOL, atol=_ATOL)


def _reference_forward(w_dend, w_soma, w_readout, alpha_s, alpha_d, alpha_m,
                       T_p, config, alpha_w, x, key, rate, surrogate_fn):
    """Independent plain-Python-loop recompute of ``(mean_voltage, G_r, G_s, G_d)``.

    Reuses the golden-validated primitives (``dynamics_step``, ``readout_integrate``,
    the eligibility updates) but re-does the accumulation OUTSIDE the Network scan (and
    projects the input once, up front — the Network now projects inside the scan, so this
    matches within tol), with the surrogate passed in explicitly. Applies
    ``surrogate_fn`` to BOTH ``sp`` and ``hp`` (pins the S wiring), and takes the decays
    positionally (pins the alpha wiring).
    """
    x = x.astype(jnp.float64)
    T, N, K, J = x.shape[0], w_dend.shape[0], w_dend.shape[1], w_readout.shape[0]
    dend, soma = x @ w_dend.T, x @ w_soma.T
    tix = jnp.arange(T, dtype=jnp.int32)
    keys = random.split(key, T)
    scale = 1.0 / (1.0 - rate)

    dyn, elig = DynamicsState.zeros(N), EligState.zeros(N, K)
    r_v, r_sum = jnp.zeros(J), jnp.zeros(J)
    G_r, G_s, G_d = jnp.zeros(N), jnp.zeros((N, K)), jnp.zeros((N, K))
    for t in range(T):
        dyn, o, v_pre, h_h, h_prev, mu_at_tp = dynamics_step(
            dyn, dend[t], soma[t], tix[t], alpha_s, alpha_d, T_p, config, alpha_w)
        ho = o.astype(jnp.float64)
        mask = random.bernoulli(keys[t], 1.0 - rate, (N,)).astype(jnp.float64)
        ho = ho * mask * scale
        r_v, r_sum = readout_integrate(r_v, r_sum, ho, w_readout, alpha_m)
        eps_r = alpha_m * elig.eps_r + ho
        eps_s = update_somatic_eligibility(elig.eps_s, x[t], alpha_s)
        eps_d, eps_d_bar = update_dendritic_eligibility(elig.eps_d, elig.eps_d_bar, x[t], h_prev, alpha_d)
        elig = EligState(eps_s=eps_s, eps_d=eps_d, eps_d_bar=eps_d_bar, eps_r=eps_r)
        sp = surrogate_fn(v_pre + config.gamma * h_h - config.v_th, config.beta_s)
        hp = surrogate_fn(mu_at_tp - config.mu_th, config.beta_d)
        G_r = G_r + eps_r
        G_s = G_s + sp[:, None] * eps_s[None, :]
        G_d = G_d + (sp * hp * config.gamma)[:, None] * eps_d_bar
    return r_sum / T, G_r, G_s, G_d


# ── inference forward / activity ≈ dynamics free functions (within tol) ─────────

def test_forward_matches_predict():
    c, w = _config(), _weights()
    x = jnp.asarray(_inp("x_single"))
    _close(network.predict(_golden_network(), x), dynamics.predict(x, *w, *_params(c)))


def test_activity_matches():
    c, w = _config(), _weights()
    x = jnp.asarray(_inp("x_single"))
    _close(network.activity(_golden_network(), x), dynamics.activity(x, *w, *_params(c)))


def test_predict_batch_matches():
    c, w = _config(), _weights()
    xb = jnp.asarray(_inp("x_batch"))
    _close(network.predict_batch(_golden_network(), xb), dynamics.predict_batch(xb, *w, *_params(c)))


def test_activity_batch_matches():
    c, w = _config(), _weights()
    xb = jnp.asarray(_inp("x_batch"))
    _close(network.activity_batch(_golden_network(), xb), dynamics.activity_batch(xb, *w, *_params(c)))


# ── independent reference recompute reproduces the training forward ────────────

def test_reference_reproduces_network_bptt():
    """Check the plain-loop reference itself against the Network (fast_sigmoid)."""
    c, w = _config(), _weights()
    x, k = jnp.asarray(_inp("x_single")), _key_single()
    mv, accum = network.forward_accumulate(_golden_network(), x, k, 0.0)
    mv_ref, G_r, G_s, G_d = _reference_forward(*w, *_params(c), x, k, 0.0, surrogate.fast_sigmoid)
    _close(mv, mv_ref)
    _close(accum.G_r, G_r)
    _close(accum.credit[0].G_s, G_s)
    _close(accum.credit[0].G_d, G_d)


def test_grads_match_golden():
    x, k = jnp.asarray(_inp("x_single")), _key_single()
    T = x.shape[0]
    net = _golden_network()
    mv, accum = network.forward_accumulate(net, x, k, 0.0)
    _, _, grads = network.loss_and_grads(net, mv, accum, _sm_single(), T)
    _close(grads.w_readout, _G["train_single_off__grad_r"])
    _close(grads.w_soma[0], _G["train_single_off__grad_s"])
    _close(grads.w_dend[0], _G["train_single_off__grad_d"])


# ── distinct-alpha wiring (golden has alpha_s == alpha_d, hiding a swap) ────────

def _distinct_network():
    """Golden weights, but alpha_s != alpha_d != alpha_m so any swap of two decays in
    the delegation DIVERGES. Returns (net, weights, free_params) in reference order."""
    c = _config()
    w_dend, w_soma, w_readout = _weights()
    _, _, _, T_p, _, alpha_w = _params(c)
    a_s, a_d, a_m = jnp.asarray(0.5), jnp.asarray(0.9), jnp.asarray(0.7)
    layer = TwoCompLayer(
        w_dend=w_dend, w_soma=w_soma, T_p=T_p, alpha_s=a_s, alpha_d=a_d, alpha_w=alpha_w
    )
    net = Network(hidden=(layer,), readout=ReadoutLayer(w=w_readout, alpha_m=a_m), config=c)
    return net, (w_dend, w_soma, w_readout), (a_s, a_d, a_m, T_p, c, alpha_w)


def test_distinct_alpha_forward_accumulate_wiring():
    net, w, params = _distinct_network()
    x, k = jnp.asarray(_inp("x_single")), _key_single()
    mv_o, accum = network.forward_accumulate(net, x, k, 0.0)
    mv_ref, G_r, G_s, G_d = _reference_forward(*w, *params, x, k, 0.0, surrogate.fast_sigmoid)
    _close(mv_o, mv_ref)
    _close(accum.G_r, G_r)
    _close(accum.credit[0].G_s, G_s)
    _close(accum.credit[0].G_d, G_d)


def test_distinct_alpha_predict_wiring():
    net, w, params = _distinct_network()
    x = jnp.asarray(_inp("x_single"))
    _close(network.predict(net, x), dynamics.predict(x, *w, *params))


def test_distinct_alpha_activity_wiring():
    net, w, params = _distinct_network()
    x = jnp.asarray(_inp("x_single"))
    _close(network.activity(net, x), dynamics.activity(x, *w, *params))


# ── pytree staticness ──────────────────────────────────────────────────────────

def test_static_components_not_leaves():
    """The components (credit_method/surrogate: str, dendritic_credit: bool) must be
    static aux data, never traced leaves. Config floats are legitimately leaves; a
    leaked component would be a str or bool — rejecting only str misses a bool leak."""
    net = _golden_network()
    leaves = jax.tree_util.tree_leaves(net)
    assert all(not isinstance(leaf, (str, bool)) for leaf in leaves)
    assert net.credit_method == "bptt"
    assert net.hidden[0].surrogate == "fast_sigmoid"
    assert net.hidden[0].dendritic_credit is True
    assert net.B == ()


def test_treedef_changes_on_static_components_not_weights():
    net = _golden_network()
    t0 = jax.tree_util.tree_structure(net)
    lay = net.hidden[0]
    assert jax.tree_util.tree_structure(net.replace(credit_method="dfa")) != t0
    assert jax.tree_util.tree_structure(net.replace(hidden=(lay.replace(surrogate="ste"),))) != t0
    assert jax.tree_util.tree_structure(net.replace(hidden=(lay.replace(dendritic_credit=False),))) != t0
    assert jax.tree_util.tree_structure(net.replace(hidden=(lay.replace(w_dend=lay.w_dend * 2.0),))) == t0


# ── Credit Assignment Method components (property / reference / reduction) ─────

def test_ste_surrogate_reference_parity():
    """Decisive: the ste Network's G_s AND G_d match a plain-loop recompute applying
    ste_boxcar to both traces. A bug leaving hp on fast_sigmoid moves G_d off this
    reference and fails here — the hole a mere 'grad_d differs' check cannot catch."""
    net = _golden_network()
    net = net.replace(hidden=(net.hidden[0].replace(surrogate="ste"),))
    x, k = jnp.asarray(_inp("x_single")), _key_single()
    _, accum = network.forward_accumulate(net, x, k, 0.0)
    c, w = _config(), _weights()
    _, _, G_s, G_d = _reference_forward(*w, *_params(c), x, k, 0.0, surrogate.ste_boxcar)
    _close(accum.credit[0].G_s, G_s)
    _close(accum.credit[0].G_d, G_d)
    _differs(accum.credit[0].G_s, _reference_forward(*w, *_params(c), x, k, 0.0, surrogate.fast_sigmoid)[2])


def test_prune_grad_d_zero_forward_unchanged():
    base = _golden_network()
    pruned = base.replace(hidden=(base.hidden[0].replace(dendritic_credit=False),))
    x, k = jnp.asarray(_inp("x_single")), _key_single()
    T = x.shape[0]
    mv_b, accum_b = network.forward_accumulate(base, x, k, 0.0)
    mv_p, accum_p = network.forward_accumulate(pruned, x, k, 0.0)
    _eq(accum_p.credit[0].G_d, accum_b.credit[0].G_d)  # forward (incl G_d) unchanged
    _, _, gb = network.loss_and_grads(base, mv_b, accum_b, _sm_single(), T)
    _, _, gp = network.loss_and_grads(pruned, mv_p, accum_p, _sm_single(), T)
    _eq(gp.w_dend[0], jnp.zeros_like(gp.w_dend[0]))     # D fires: grad_d == 0
    _eq(gp.w_readout, gb.w_readout)                     # grad_r, grad_s invariant to prune
    _eq(gp.w_soma[0], gb.w_soma[0])


def test_dfa_routing_grad_r_invariant_and_fires():
    base = _golden_network()
    dfa = base.replace(credit_method="dfa", B=(_feedback_B(),))
    x, k = jnp.asarray(_inp("x_single")), _key_single()
    T = x.shape[0]
    mv, accum = network.forward_accumulate(base, x, k, 0.0)  # forward is routing-independent
    _, _, gb = network.loss_and_grads(base, mv, accum, _sm_single(), T)
    _, _, gd = network.loss_and_grads(dfa, mv, accum, _sm_single(), T)
    _eq(gd.w_readout, gb.w_readout)                    # grad_r invariant to routing
    _differs(gd.w_soma[0], gb.w_soma[0])               # F fires on grad_s/grad_d
    _differs(gd.w_dend[0], gb.w_dend[0])


def test_dfa_with_w_readout_reduces_to_bptt():
    """Reduction identity: dfa fed B = w_readout reproduces bptt (both runtime M)."""
    base = _golden_network()
    dfa = base.replace(credit_method="dfa", B=(_weights()[2],))
    x, k = jnp.asarray(_inp("x_single")), _key_single()
    T = x.shape[0]
    mv, accum = network.forward_accumulate(base, x, k, 0.0)
    loss_d, pred_d, gd = network.loss_and_grads(dfa, mv, accum, _sm_single(), T)
    loss_b, pred_b, gb = network.loss_and_grads(base, mv, accum, _sm_single(), T)
    _eq(loss_d, loss_b)
    _eq(pred_d, pred_b)
    _eq(gd.w_readout, gb.w_readout)
    _eq(gd.w_soma[0], gb.w_soma[0])
    _eq(gd.w_dend[0], gb.w_dend[0])


def test_composed_components():
    """S + D + F compose without interference."""
    base = _golden_network()
    lay = base.hidden[0].replace(surrogate="ste", dendritic_credit=False)
    net = base.replace(credit_method="dfa", B=(_feedback_B(),), hidden=(lay,))
    x, k = jnp.asarray(_inp("x_single")), _key_single()
    T = x.shape[0]
    mv, accum = network.forward_accumulate(net, x, k, 0.0)
    _, _, g = network.loss_and_grads(net, mv, accum, _sm_single(), T)
    _eq(g.w_dend[0], jnp.zeros_like(g.w_dend[0]))       # D fired: grad_d == 0
    _eq(g.w_readout, _G["train_single_off__grad_r"])    # grad_r invariant to all three
    # F + S both fire on grad_s -> differs from the bptt-net grad_s
    mv_b, accum_b = network.forward_accumulate(base, x, k, 0.0)
    _, _, g_b = network.loss_and_grads(base, mv_b, accum_b, _sm_single(), T)
    _differs(g.w_soma[0], g_b.w_soma[0])


def test_forward_accumulate_routing_invariant():
    """The forward accumulation (TrainAccum) is routing-independent."""
    x, k = jnp.asarray(_inp("x_single")), _key_single()
    _, accum_bptt = network.forward_accumulate(_golden_network(), x, k, 0.0)
    _, accum_dfa = network.forward_accumulate(
        _golden_network().replace(credit_method="dfa", B=(_feedback_B(),)), x, k, 0.0
    )
    _eq(accum_bptt.credit[0].G_s, accum_dfa.credit[0].G_s)
    _eq(accum_bptt.credit[0].G_d, accum_dfa.credit[0].G_d)
    _eq(accum_bptt.G_r, accum_dfa.G_r)


def test_dfa_requires_B():
    """Alignment routing without B must error, never silently use w_readout (R2)."""
    net = _golden_network().replace(credit_method="dfa")  # B == ()
    x, k = jnp.asarray(_inp("x_single")), _key_single()
    mv, accum = network.forward_accumulate(net, x, k, 0.0)  # forward is routing-independent
    with pytest.raises(ValueError):
        network.loss_and_grads(net, mv, accum, _sm_single(), x.shape[0])


# ── surrogate / feedback / routing unit checks ─────────────────────────────────

def test_surrogate_unit():
    assert float(surrogate.fast_sigmoid(0.0, 1.0)) == 1.0  # exact peak at threshold
    xs = jnp.linspace(-5.0, 5.0, 101)
    fs = np.asarray(surrogate.fast_sigmoid(xs, 1.0))
    assert fs.max() <= 1.0 and np.isclose(fs.max(), 1.0)
    assert (fs > 0.0).all() and (fs <= 1.0).all()
    bc = np.asarray(surrogate.ste_boxcar(xs, 1.0))
    assert set(np.unique(bc)) <= {0.0, 1.0}
    assert float(surrogate.ste_boxcar(0.0, 1.0)) == 1.0
    assert float(surrogate.ste_boxcar(5.0, 1.0)) == 0.0


def test_unknown_surrogate_and_routing_raise():
    with pytest.raises(ValueError):
        surrogate.get_surrogate("nope")
    with pytest.raises(ValueError):
        credit.credit_method("nope")


def test_feedback_weights_shape_scale():
    c = _config()
    J, N = _weights()[2].shape
    B = feedback.init_feedback_weights(random.PRNGKey(1), J, N, c)
    assert B.shape == (J, N)
    # Large draw so a sqrt(2) convention error fails (the 24-sample band can't).
    Jb, Nb = 200, 200
    Bbig = feedback.init_feedback_weights(random.PRNGKey(2), Jb, Nb, c)
    expected_std = float(jnp.sqrt(2.0 / Nb) * c.weight_scale)
    assert abs(float(jnp.std(Bbig)) - expected_std) < 0.1 * expected_std


# ── rfa routing (single-layer path collapses to dfa) ───────────────────────────

def test_rfa_routing_equals_dfa_single_layer():
    """rfa is a live routing accepted by the guard; for ONE hidden layer it collapses
    to the same e = delta @ B as dfa. Catches a registry/branch drop of 'rfa'."""
    base = _golden_network()
    B = _feedback_B()
    x, k = jnp.asarray(_inp("x_single")), _key_single()
    T = x.shape[0]
    mv, accum = network.forward_accumulate(base, x, k, 0.0)
    l_d, p_d, g_d = network.loss_and_grads(base.replace(credit_method="dfa", B=(B,)), mv, accum, _sm_single(), T)
    l_r, p_r, g_r = network.loss_and_grads(base.replace(credit_method="rfa", B=(B,)), mv, accum, _sm_single(), T)
    _eq(l_r, l_d)
    _eq(p_r, p_d)
    _eq(g_r.w_readout, g_d.w_readout)
    _eq(g_r.w_soma[0], g_d.w_soma[0])
    _eq(g_r.w_dend[0], g_d.w_dend[0])


# ── batched components (the _batch vmap boundary honours surrogate/dendrite/routing) ─

def test_batch_dfa_grad_r_invariant_and_fires():
    base = _golden_network()
    dfa = base.replace(credit_method="dfa", B=(_feedback_B(),))
    _, _, gr_b, gs_b, _ = _run_net_batch(base)
    _, _, gr_d, gs_d, _ = _run_net_batch(dfa)
    _eq(gr_d, gr_b)        # grad_r invariant to routing, batched
    _differs(gs_d, gs_b)   # routing fires on grad_s, batched


def test_batch_prune_grad_d_zero():
    net = _golden_network()
    net = net.replace(hidden=(net.hidden[0].replace(dendritic_credit=False),))
    _, _, _, _, gd = _run_net_batch(net)
    _eq(gd, jnp.zeros_like(gd))


def test_batch_ste_grad_r_invariant():
    base = _golden_network()
    ste = base.replace(hidden=(base.hidden[0].replace(surrogate="ste"),))
    _, _, gr_b, _, _ = _run_net_batch(base)
    _, _, gr_s, _, _ = _run_net_batch(ste)
    _eq(gr_s, gr_b)


# ── primitive wiring hidden behind degenerate golden values ────────────────────

def test_dynamics_step_alpha_wiring():
    """The dendrite (mu) decays by alpha_d and the soma (v) by alpha_s — pinned
    independently of the golden's alpha_s == alpha_d degeneracy. At t>0 with zero drive
    and no plateau/spike: mu = alpha_d * mu_prev, v_pre = alpha_s * v_prev."""
    c = _config()
    st = DynamicsState(
        mu=jnp.array([0.3]), v=jnp.array([0.4]), h=jnp.array([0], jnp.int32),
        t_prime=jnp.array([0], jnp.int32), mu_at_tprime=jnp.array([0.3]), w=jnp.array([0.0]),
    )
    new, o, v_pre, *_ = dynamics_step(
        st, jnp.array([0.0]), jnp.array([0.0]), jnp.int32(1),
        jnp.asarray(0.5), jnp.asarray(0.9), jnp.array([100], jnp.int32), c, jnp.asarray(0.0),
    )
    np.testing.assert_allclose(float(new.mu[0]), 0.9 * 0.3)   # alpha_d drives the dendrite
    np.testing.assert_allclose(float(v_pre[0]), 0.5 * 0.4)    # alpha_s drives the soma


def test_dynamics_step_adaptation_wiring():
    """v_pre_reset SUBTRACTS the adaptation current w_prev (sign), and w evolves per
    alpha_w / a_adapt / b_adapt. Untested by the golden (a_adapt == b_adapt == 0)."""
    c = _config().replace(a_adapt=0.3, b_adapt=0.7)
    st = DynamicsState(
        mu=jnp.array([0.0]), v=jnp.array([0.5]), h=jnp.array([0], jnp.int32),
        t_prime=jnp.array([0], jnp.int32), mu_at_tprime=jnp.array([0.0]), w=jnp.array([0.2]),
    )
    alpha_s, alpha_w = 0.5, 0.8
    new, o, v_pre, *_ = dynamics_step(
        st, jnp.array([0.0]), jnp.array([1.0]), jnp.int32(1),
        jnp.asarray(alpha_s), jnp.asarray(0.9), jnp.array([100], jnp.int32), c, jnp.asarray(alpha_w),
    )
    vp = alpha_s * 0.5 + 1.0 - 0.2                       # alpha_s*v_prev + soma_in - w_prev
    np.testing.assert_allclose(float(v_pre[0]), vp)     # the `- w_prev` sign
    expected_w = alpha_w * 0.2 + (1 - alpha_w) * 0.3 * vp + 0.7 * float(o[0])
    np.testing.assert_allclose(float(new.w[0]), expected_w)


def test_credit_step_threshold_wiring():
    """credit_step's somatic surrogate uses v_th and the dendritic uses mu_th — pinned
    with distinct thresholds (golden has mu_th == v_th == 1.0), via fast_sigmoid's form.
    Exercises the back-compat ``credit_step`` wrapper over ``credit_ingredients``."""
    c = _config().replace(v_th=1.2, mu_th=0.8, gamma=0.5, beta_s=1.0, beta_d=1.5)
    lay = TwoCompLayer(
        w_dend=jnp.zeros((1, 1)), w_soma=jnp.zeros((1, 1)), T_p=jnp.array([100], jnp.int32),
        alpha_s=jnp.asarray(0.5), alpha_d=jnp.asarray(0.9), alpha_w=jnp.asarray(0.0),
    )
    _, g_s, g_d = lay.credit_step(
        EligState.zeros(1, 1),
        jnp.array([0.7]), jnp.array([0], jnp.int32), jnp.array([0], jnp.int32),
        jnp.array([0.6]), jnp.array([1.0]), c,
    )
    sp = float(surrogate.fast_sigmoid(0.7 + 0.5 * 0 - 1.2, 1.0))   # somatic uses v_th
    hp = float(surrogate.fast_sigmoid(0.6 - 0.8, 1.5))             # dendritic uses mu_th
    np.testing.assert_allclose(float(g_s[0, 0]), sp)              # eps_s == 1 -> g_s == sp
    np.testing.assert_allclose(float(g_d[0, 0]), sp * hp * 0.5)   # eps_d_bar == 1 -> g_d == sp*hp*gamma


def test_credit_taxonomy():
    """Pin the routing taxonomy the multi-layer dispatch keys on."""
    assert credit.is_decoupled("dfa")
    assert not credit.is_decoupled("bptt") and not credit.is_decoupled("rfa")
    assert credit.needs_feedback("dfa") and credit.needs_feedback("rfa")
    assert not credit.needs_feedback("bptt")
