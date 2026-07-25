"""N-hidden-layer forward/credit + DFA multilayer + build_network.

The single-layer golden checks (test_dynamics/test_backward/test_grads, test_objects)
cover the one-layer path. This file covers what only exists at N > 1 hidden layers, with
no autodiff and no golden (the golden is single-layer). The DFA multilayer credit is
checked against an independent plain-Python N-layer reference recompute (the
"materialized reference"): a bottom-up dynamics sweep threading dropout-scaled spikes
L->L+1, per-layer G_s/G_d accumulation, a shared readout G_r, and the decoupled finalize
e^L = delta @ B_L. Plus: per-layer decoupling (the dendrite's gradient is local), the
multilayer method guards (coupled train raises; DFA train + all-method infer succeed),
int32 scan-carry dtype across the unrolled scan, and the build_network init recipe
(reproduces the single-layer reference weights).
"""
import json
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import pytest

from dendroprop import network, surrogate
from dendroprop.config import NeuronConfig
from dendroprop.dynamics import dynamics_step, readout_integrate
from dendroprop.state import DynamicsState, EligState
from dendroprop.layers import (
    TwoCompLayer, ReadoutLayer,
    update_somatic_eligibility, update_dendritic_eligibility,
)
from dendroprop.network import Network, build_network, _split_layer_keys, _dynamics_sweep

_GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
_M = json.loads((_GOLDEN_DIR / "manifest.json").read_text())
_RTOL = _M["tolerance"]["rtol"]
_ATOL = _M["tolerance"]["atol"]


def _cfg():
    return NeuronConfig(**_M["config"])


def _eq(a, b):
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def _close(a, b):
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=_RTOL, atol=_ATOL)


def _differs(a, b):
    assert not np.allclose(np.asarray(a), np.asarray(b), rtol=_RTOL, atol=_ATOL)


# ── a controlled 2-hidden-layer network with DISTINCT decays (catches alpha swaps) ──

_K, _N1, _N2, _J, _T = 5, 4, 3, 6, 12


def _two_layer_net(method="dfa", *, dendritic_credit=(True, True)):
    """Fixed-seed 2-hidden-layer Network built DIRECTLY (independent of build_network),
    with distinct alphas per layer so any decay miswiring diverges. dfa/rfa carry
    per-layer feedback B; bptt carries none (B=())."""
    c = _cfg()
    k = random.split(random.PRNGKey(20260724), 8)
    scale = 0.8
    w_dend1 = random.normal(k[0], (_N1, _K)) * scale
    w_soma1 = random.normal(k[1], (_N1, _K)) * scale
    w_dend2 = random.normal(k[2], (_N2, _N1)) * scale
    w_soma2 = random.normal(k[3], (_N2, _N1)) * scale
    w_r = random.normal(k[4], (_J, _N2)) * scale
    T_p1 = jnp.full((_N1,), 3, dtype=jnp.int32)
    T_p2 = jnp.full((_N2,), 4, dtype=jnp.int32)
    layer1 = TwoCompLayer(
        w_dend=w_dend1, w_soma=w_soma1, T_p=T_p1,
        alpha_s=jnp.asarray(0.5), alpha_d=jnp.asarray(0.9), alpha_w=jnp.asarray(0.0),
        dendritic_credit=dendritic_credit[0],
    )
    layer2 = TwoCompLayer(
        w_dend=w_dend2, w_soma=w_soma2, T_p=T_p2,
        alpha_s=jnp.asarray(0.6), alpha_d=jnp.asarray(0.8), alpha_w=jnp.asarray(0.0),
        dendritic_credit=dendritic_credit[1],
    )
    readout = ReadoutLayer(w=w_r, alpha_m=jnp.asarray(0.7))
    B = ()
    if method in ("dfa", "rfa"):
        B = (random.normal(k[5], (_J, _N1)) * scale, random.normal(k[6], (_J, _N2)) * scale)
    return Network(hidden=(layer1, layer2), readout=readout, config=c,
                   credit_method=method, B=B)


def _input():
    x = (random.uniform(random.PRNGKey(7), (_T, _K)) < 0.3).astype(jnp.float64)
    key = random.PRNGKey(99)
    return x, key


def _sm(label=1):
    c = _cfg()
    return network.smooth_targets(label, _J, c.loss_label_smoothing)


def _reference_dfa(net, x, key, rate, target_smoothed):
    """Independent plain-Python N-layer DFA recompute -> (mv, grad_r, [grad_s], [grad_d]).

    Mirrors _dynamics_sweep's math (projection inside; dropout every hidden->next edge;
    layer i>0's input = layer i-1's dropped spikes; readout eligibility from the top
    layer) but re-does everything OUTSIDE the Network scan. Uses fast_sigmoid on BOTH
    surrogate factors (the layer default). DFA finalize: e^L = delta @ B_L per layer."""
    hidden, ro, cfg = net.hidden, net.readout, net.config
    n_layers = len(hidden)
    T = x.shape[0]
    x = x.astype(jnp.float64)
    tix = jnp.arange(T, dtype=jnp.int32)
    dkeys_all = random.split(key, T)
    scale = 1.0 / (1.0 - rate)

    dyns = [DynamicsState.zeros(l.n) for l in hidden]
    eligs = [EligState.zeros(l.n, l.n_in) for l in hidden]
    r_v = jnp.zeros(ro.n)
    r_sum = jnp.zeros(ro.n)
    eps_r = jnp.zeros(hidden[-1].n)
    G_r = jnp.zeros(hidden[-1].n)
    G_s = [jnp.zeros((l.n, l.n_in)) for l in hidden]
    G_d = [jnp.zeros((l.n, l.n_in)) for l in hidden]

    for t in range(T):
        dkeys = _split_layer_keys(dkeys_all[t], n_layers)
        layer_in = x[t]
        obs, spikes_list = [], []
        for i, l in enumerate(hidden):
            dend = layer_in @ l.w_dend.T
            soma = layer_in @ l.w_soma.T
            dyns[i], o, v_pre, h_new, h_prev, mu_at_tp = dynamics_step(
                dyns[i], dend, soma, tix[t], l.alpha_s, l.alpha_d, l.T_p, cfg, l.alpha_w)
            sp_out = o.astype(jnp.float64)
            mask = random.bernoulli(dkeys[i], 1.0 - rate, (l.n,)).astype(jnp.float64)
            sp_out = sp_out * mask * scale
            obs.append((v_pre, h_new, h_prev, mu_at_tp))
            spikes_list.append(sp_out)
            layer_in = sp_out
        top = spikes_list[-1]
        r_v, r_sum = readout_integrate(r_v, r_sum, top, ro.w, ro.alpha_m)
        eps_r = ro.alpha_m * eps_r + top
        G_r = G_r + eps_r
        inputs = [x[t]] + spikes_list[:-1]
        for i, l in enumerate(hidden):
            v_pre, h_new, h_prev, mu_at_tp = obs[i]
            eps_s = update_somatic_eligibility(eligs[i].eps_s, inputs[i], l.alpha_s)
            eps_d, eps_d_bar = update_dendritic_eligibility(
                eligs[i].eps_d, eligs[i].eps_d_bar, inputs[i], h_prev, l.alpha_d)
            eligs[i] = eligs[i].replace(eps_s=eps_s, eps_d=eps_d, eps_d_bar=eps_d_bar)
            sp = surrogate.fast_sigmoid(v_pre + cfg.gamma * h_new - cfg.v_th, cfg.beta_s)
            hp = surrogate.fast_sigmoid(mu_at_tp - cfg.mu_th, cfg.beta_d)
            G_s[i] = G_s[i] + sp[:, None] * eps_s[None, :]
            G_d[i] = G_d[i] + (sp * hp * cfg.gamma)[:, None] * eps_d_bar

    mv = r_sum / T
    scaled = mv / cfg.loss_temperature + cfg.loss_count_bias
    probs = jnp.exp(scaled - jnp.max(scaled))
    probs = probs / jnp.sum(probs)
    delta = target_smoothed - probs

    grad_r = (delta[:, None] * G_r[None, :]) / T
    grad_s, grad_d = [], []
    for i, l in enumerate(hidden):
        e = delta @ net.B[i]
        grad_s.append((e[:, None] * G_s[i]) / (T * cfg.soma_grad_scale))
        gd = (e[:, None] * G_d[i]) / T if l.dendritic_credit else jnp.zeros_like(G_d[i])
        grad_d.append(gd)
    return mv, grad_r, grad_s, grad_d


# ── DFA multilayer == independent materialized reference (dropout off AND on) ───

@pytest.mark.parametrize("rate", [0.0, 0.3])
def test_dfa_2layer_matches_reference(rate):
    net = _two_layer_net("dfa")
    x, key = _input()
    sm = _sm()
    mv, accum = network.forward_accumulate(net, x, key, rate)
    loss, pred, grads = network.loss_and_grads(net, mv, accum, sm, _T)
    mv_ref, gr_ref, gs_ref, gd_ref = _reference_dfa(net, x, key, rate, sm)
    _close(mv, mv_ref)
    _close(grads.w_readout, gr_ref)
    for i in range(2):
        _close(grads.w_soma[i], gs_ref[i])
        _close(grads.w_dend[i], gd_ref[i])


def test_trainaccum_structure_2layer():
    net = _two_layer_net("dfa")
    x, key = _input()
    _, accum = network.forward_accumulate(net, x, key, 0.0)
    assert accum.G_r.shape == (_N2,)              # readout accumulator over the TOP layer
    assert len(accum.credit) == 2                 # one LayerCredit per hidden layer
    assert accum.credit[0].G_s.shape == (_N1, _K)
    assert accum.credit[0].G_d.shape == (_N1, _K)
    assert accum.credit[1].G_s.shape == (_N2, _N1)


# ── DFA is decoupled per layer: B_L credits ONLY layer L; grad_r invariant to B ──

def test_dfa_decoupled_per_layer_feedback():
    net = _two_layer_net("dfa")
    x, key = _input()
    sm = _sm()
    mv, accum = network.forward_accumulate(net, x, key, 0.0)  # accumulation is B-free
    _, _, g0 = network.loss_and_grads(net, mv, accum, sm, _T)
    # Perturb ONLY the top layer's feedback B[1].
    net_top = net.replace(B=(net.B[0], net.B[1] * 2.0))
    _, _, g_top = network.loss_and_grads(net_top, mv, accum, sm, _T)
    _eq(g_top.w_soma[0], g0.w_soma[0])   # layer-0 grads untouched by B[1] (dendrite sink)
    _eq(g_top.w_dend[0], g0.w_dend[0])
    _differs(g_top.w_soma[1], g0.w_soma[1])
    _eq(g_top.w_readout, g0.w_readout)   # grad_r never uses B
    # Perturb ONLY the bottom layer's feedback B[0].
    net_bot = net.replace(B=(net.B[0] * 2.0, net.B[1]))
    _, _, g_bot = network.loss_and_grads(net_bot, mv, accum, sm, _T)
    _differs(g_bot.w_soma[0], g0.w_soma[0])
    _eq(g_bot.w_soma[1], g0.w_soma[1])   # layer-1 grads untouched by B[0]


def test_dfa_2layer_prune_zeros_only_that_layer():
    net = _two_layer_net("dfa", dendritic_credit=(True, False))  # prune layer 1's dendrite
    x, key = _input()
    sm = _sm()
    mv, accum = network.forward_accumulate(net, x, key, 0.0)
    _, _, g = network.loss_and_grads(net, mv, accum, sm, _T)
    _eq(g.w_dend[1], jnp.zeros_like(g.w_dend[1]))   # pruned layer -> grad_d == 0
    _differs(g.w_dend[0], jnp.zeros_like(g.w_dend[0]))  # unpruned layer still credits


# ── multilayer method guards: coupled train raises; DFA train + all infer succeed ──

def test_multilayer_coupled_train_raises_dfa_and_infer_succeed():
    x, key = _input()
    sm = _sm()
    dfa = _two_layer_net("dfa")
    bptt = _two_layer_net("bptt")
    rfa = _two_layer_net("rfa")
    # Inference (dynamics only) succeeds for every method at 2 layers.
    for net in (dfa, bptt, rfa):
        _ = network.predict(net, x)
        _ = network.activity(net, x)
    # DFA training succeeds end to end.
    mv, accum = network.forward_accumulate(dfa, x, key, 0.0)
    _ = network.loss_and_grads(dfa, mv, accum, sm, _T)
    # Coupled (bptt/rfa) training at >1 hidden layer raises (soma_backward placeholder):
    # at forward_accumulate (via strat.init, pre-scan) ...
    with pytest.raises(NotImplementedError):
        network.forward_accumulate(bptt, x, key, 0.0)
    with pytest.raises(NotImplementedError):
        network.forward_accumulate(rfa, x, key, 0.0)
    # ... and independently at loss_and_grads (via strat.finalize -> soma_backward), a
    # separately-callable public entry point. Forward accumulation is routing-independent,
    # so the dfa net's 2-layer (mv, accum) is a valid input the coupled finalize guard
    # rejects on len(hidden) > 1 before touching the accumulator.
    for net in (bptt, rfa):
        with pytest.raises(NotImplementedError):
            network.loss_and_grads(net, mv, accum, sm, _T)


def test_dfa_requires_one_B_per_layer():
    net = _two_layer_net("dfa").replace(B=(random.normal(random.PRNGKey(3), (_J, _N1)),))  # only 1 B
    x, key = _input()
    mv, accum = network.forward_accumulate(net, x, key, 0.0)  # accumulation is B-free
    with pytest.raises(ValueError):
        network.loss_and_grads(net, mv, accum, _sm(), _T)


# ── per-layer int32 R3 across the unrolled N-layer sweep ────────────────────────

def test_per_layer_int32_preserved():
    net = _two_layer_net("dfa")
    x, _ = _input()
    cfg = _cfg()
    dyns = tuple(DynamicsState.zeros(l.n) for l in net.hidden)
    dkeys = _split_layer_keys(random.PRNGKey(0), len(net.hidden))
    # Two sweeps: the second consumes the first's carry, so any int32->int64 drift shows.
    dyns, _, _ = _dynamics_sweep(net.hidden, dyns, x[0], jnp.int32(0), dkeys, 0.0, 1.0, cfg, True)
    dyns, _, _ = _dynamics_sweep(net.hidden, dyns, x[1], jnp.int32(1), dkeys, 0.0, 1.0, cfg, True)
    for d in dyns:
        assert d.h.dtype == jnp.int32
        assert d.t_prime.dtype == jnp.int32


# ── build_network: init recipe vs frozen single-layer weights; multilayer shapes ─

def test_build_network_matches_frozen_init_recipe():
    """The fresh-init recipe reproduces the frozen single-layer weights bit-for-bit
    under the same seed + config (key-split order k1->w_dend, k2->w_soma, k3->T_p;
    readout on key_r).

    ``tests/golden/init_recipe.npz`` holds those reference weights — the frozen
    reference for the init recipe, as ``golden.npz`` is for the forward/gradient path."""
    R = np.load(_GOLDEN_DIR / "init_recipe.npz", allow_pickle=False)
    c = _cfg()
    key = random.PRNGKey(12345)
    K, N, J = 7, 5, 4
    net = build_network(key, (K, [N], J), c, "bptt")
    lay = net.hidden[0]
    _eq(lay.w_dend, R["w_dend"])
    _eq(lay.w_soma, R["w_soma"])
    _eq(lay.T_p, R["T_p"])
    assert np.issubdtype(np.asarray(lay.T_p).dtype, np.integer)  # int contract (R3)
    _eq(net.readout.w, R["w_readout"])
    _eq(lay.alpha_s, R["alpha_s"])
    _eq(lay.alpha_d, R["alpha_d"])
    _eq(lay.alpha_w, R["alpha_w"])
    _eq(net.readout.alpha_m, R["alpha_m"])


def test_build_network_multilayer_dfa_shapes():
    c = _cfg()
    net = build_network(random.PRNGKey(1), (5, [4, 3], 6), c, "dfa")
    assert len(net.hidden) == 2
    assert net.hidden[0].w_dend.shape == (4, 5)
    assert net.hidden[1].w_dend.shape == (3, 4)
    assert net.readout.w.shape == (6, 3)
    assert len(net.B) == 2                  # one feedback matrix per hidden layer
    assert net.B[0].shape == (6, 4)
    assert net.B[1].shape == (6, 3)
    # DFA multilayer trains through the built net.
    x = (random.uniform(random.PRNGKey(2), (10, 5)) < 0.3).astype(jnp.float64)
    mv, accum = network.forward_accumulate(net, x, random.PRNGKey(3), 0.0)
    _, _, grads = network.loss_and_grads(net, mv, accum, network.smooth_targets(0, 6, c.loss_label_smoothing), 10)
    assert grads.w_soma[0].shape == (4, 5) and grads.w_dend[1].shape == (3, 4)


def test_build_network_bptt_no_feedback_and_rfa_multilayer_raises():
    c = _cfg()
    bptt = build_network(random.PRNGKey(1), (5, [4, 3], 6), c, "bptt")
    assert bptt.B == ()
    rfa1 = build_network(random.PRNGKey(1), (5, [4], 6), c, "rfa")  # single-layer rfa ok
    assert len(rfa1.B) == 1 and rfa1.B[0].shape == (6, 4)
    with pytest.raises(NotImplementedError):
        build_network(random.PRNGKey(1), (5, [4, 3], 6), c, "rfa")  # rfa multilayer placeholder
