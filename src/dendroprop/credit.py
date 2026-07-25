"""Credit routing — the network-level component that names how the top-down error
reaches each hidden layer's soma.

This module is the same kind of thing as ``surrogate.py``: a few named pure functions
the ``Network`` selects, not a closure factory that replaces the Network.

Routings:
  - ``bptt`` — symmetric: the feedback matrix is the live ``w_readout`` (and, for
    multi-layer, each forward soma weight transposed).
  - ``dfa``  — direct feedback alignment: the output error is projected directly onto
    every hidden soma via a fixed random matrix ``B_L`` (one per hidden layer). The
    only method that decouples traces across layers/time, so each layer collapses to a
    post-scan ``e^L = delta @ B_L``.
  - ``rfa``  — (regular) feedback alignment: like bptt but with fixed random feedback
    weights instead of ``w^T``; it still propagates soma->soma through the layer-above
    surrogate, so it is coupled (identical to dfa for a single hidden layer).

Strategy dispatch: a routing is a ``CreditStrategy`` record, selected once at trace
time on the static ``credit_method`` field (never a runtime ``if is_decoupled``). The
strategy owns the routing:
  - ``init(net) -> credit_carry``          — the per-hidden-layer scan carrier
    (``tuple[LayerCredit]`` for the decoupled/independent accumulation).
  - ``step(credit_carry, ingredients, net) -> credit_carry`` — in-scan, per timestep.
  - ``finalize(net, delta, credit_carry, T) -> tuple[(grad_s, grad_d), ...]`` —
    post-scan, per hidden layer: turn ``delta`` + the accumulator into ``e^L`` and
    assemble ``(grad_s, grad_d)``.
The record holds Python callables closed over at trace time; it must never enter the
scan carry (only ``LayerCredit`` pytree carriers do). ``grad_r`` / ``G_r`` are
routing-invariant and shared — computed once in ``Network.loss_and_grads``, outside
the strategy.

Backward topology (do not change): inter-layer credit is soma(L)->soma(L-1) only; the
dendrite's gradient is local — ``assemble_grads`` credits grad_d from this layer's own
soma error and never propagates it to an earlier layer (a dendrite edge would branch
the plateau trace 2**L). DFA is the only method that decouples across time and is fully
implemented; the coupled soma->soma recursion for >1 hidden layer (``soma_backward``)
is the hand-derived ``dendroprop_methods.pdf`` math and stays a ``NotImplementedError``
placeholder — its accumulator must be O(Σ params), never O(T)/depth. See
docs/architecture.md.
"""
from typing import Callable, NamedTuple

from . import feedback
from .state import LayerCredit

_ROUTINGS = ("bptt", "dfa", "rfa")


def credit_method(name):
    """Validate/resolve a routing name."""
    if name not in _ROUTINGS:
        raise ValueError(f"unknown credit_method {name!r}; options: {list(_ROUTINGS)}")
    return name


def needs_feedback(routing):
    """dfa/rfa need fixed random feedback matrices B; bptt uses the live w_readout."""
    return routing in ("dfa", "rfa")


def is_decoupled(routing):
    """DFA is the only method whose per-layer traces do not link across time/layers."""
    return routing == "dfa"


def top_feedback_matrix(routing, w_readout, B):
    """M for the readout -> top-hidden-soma edge (base case of the credit recursion).

    Symmetric (``bptt``) uses the live ``w_readout`` (optimizer-updated), never a
    frozen copy. Alignment (``dfa``/``rfa``) uses the frozen fixed random ``B`` (the
    last backward edge). Raises if an alignment routing lacks ``B`` — it must never
    fall back to ``w_readout``, which would degrade feedback alignment to BPTT.
    """
    if routing == "bptt":
        return w_readout
    if routing in ("dfa", "rfa"):
        if not B:
            raise ValueError(f"{routing} routing requires fixed feedback matrices B (got none)")
        return B[-1]
    raise ValueError(f"unknown credit_method {routing!r}")


def soma_backward(routing, *args, **kwargs):
    """Coupled inter-layer step: layer L's soma error -> layer L-1's soma error, the
    hand-derived multi-layer credit math. Placeholder — lands with the coupled
    multi-layer edge. The dendrite has no analogue: its gradient is local (a dendrite
    edge would branch the plateau trace 2**L)."""
    raise NotImplementedError(
        "coupled soma->soma inter-layer credit (bptt/rfa, >1 hidden layer) is the "
        "hand-derived rule from dendroprop_methods.pdf; the accumulator must be O(Σ "
        "params), never O(T)/depth"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  CreditStrategy — a fully-swapped routing strategy, dispatched once on the static
#  credit_method. Plain NamedTuple of Python callables; closed over at TRACE TIME
#  and never in the scan carry (only LayerCredit pytree carriers are).
# ══════════════════════════════════════════════════════════════════════════════

class CreditStrategy(NamedTuple):
    """The routing strategy for one Credit Assignment Method.

    ``couples_layers`` — False only for dfa (decoupled per-layer traces); True for the
    coupled bptt/rfa soma->soma recursion. ``needs_feedback`` — dfa/rfa carry fixed
    random ``B``; bptt uses the live ``w_readout``. ``init`` / ``step`` / ``finalize``
    are the routing callables (see the module docstring).
    """

    name: str
    couples_layers: bool
    needs_feedback: bool
    init: Callable
    step: Callable
    finalize: Callable


# ── independent (decoupled) per-layer accumulation ────────────────────────────
#  Each hidden layer accumulates its own G_s/G_d with no cross-layer term. Used by
#  DFA (all N layers) and by the coupled bptt/rfa SINGLE-layer base case (one layer
#  has no cross-layer product, so the coupled carry degenerates to this).

def _independent_init(net):
    return tuple(LayerCredit.zeros(layer.n, layer.n_in) for layer in net.hidden)


def _independent_step(credit_carry, ingredients, net):
    return tuple(
        LayerCredit(G_s=lc.G_s + ing.g_s, G_d=lc.G_d + ing.g_d)
        for lc, ing in zip(credit_carry, ingredients)
    )


def _dfa_finalize(net, delta, credit_carry, T):
    """DFA is decoupled: each hidden layer L is credited independently by
    ``e^L = delta @ B_L`` (a direct random projection of the output error onto layer
    L's somas via ``feedback.error_signal``), then assembles its own
    ``(grad_s, grad_d)`` from that layer's accumulator. One feedback matrix per hidden
    layer (``len(B) == len(hidden)``); the dendrite's gradient stays local."""
    if len(net.B) != len(net.hidden):
        raise ValueError(
            f"dfa needs one fixed feedback matrix per hidden layer "
            f"(got {len(net.B)} for {len(net.hidden)} hidden layer(s))"
        )
    grads = []
    for layer, B_L, lc in zip(net.hidden, net.B, credit_carry):
        e = feedback.error_signal(delta, B_L)
        grads.append(layer.assemble_grads(e, lc, T, net.config))
    return tuple(grads)


def _coupled(name, uses_feedback):
    """Build the coupled bptt/rfa strategy. For a SINGLE hidden layer both collapse to
    one post-scan projection ``e = delta @ M`` (``M`` = live ``w_readout`` for bptt,
    fixed random ``B[-1]`` for rfa) over the independent accumulator. The coupled
    soma->soma recursion for >1 hidden layer is the hand-derived PDF math — every
    callable routes to the ``soma_backward`` placeholder there (raises at trace time)."""

    def init(net):
        if len(net.hidden) > 1:
            soma_backward(name)
        return _independent_init(net)

    def step(credit_carry, ingredients, net):
        if len(net.hidden) > 1:
            soma_backward(name)
        return _independent_step(credit_carry, ingredients, net)

    def finalize(net, delta, credit_carry, T):
        if len(net.hidden) > 1:
            soma_backward(name)
        layer = net.hidden[0]
        M = top_feedback_matrix(name, net.readout.w, net.B)
        e = feedback.error_signal(delta, M)
        return (layer.assemble_grads(e, credit_carry[0], T, net.config),)

    return CreditStrategy(
        name=name, couples_layers=True, needs_feedback=uses_feedback,
        init=init, step=step, finalize=finalize,
    )


DFA = CreditStrategy(
    name="dfa", couples_layers=False, needs_feedback=True,
    init=_independent_init, step=_independent_step, finalize=_dfa_finalize,
)


STRATEGIES = {"dfa": DFA, "bptt": _coupled("bptt", False), "rfa": _coupled("rfa", True)}


def get_strategy(name):
    """Resolve a validated ``credit_method`` name to its ``CreditStrategy`` record.

    Called at TRACE TIME on the static ``credit_method`` field; returns a Python record
    of callables that produce the traced computation. Never a traced value itself."""
    return STRATEGIES[credit_method(name)]
