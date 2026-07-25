"""Surrogate-gradient functions (the per-layer surrogate).

The forward pass emits a hard spike (dynamics.py); the surrogate is used only as the
backward derivative of that non-differentiable threshold (the ``sp`` / ``hp`` factors
in layers.py). Swapping the surrogate changes the gradient and nothing in the forward
pass — a straight-through estimator.

Both surrogates share the ``(x, beta)`` signature: ``x`` is the distance to threshold,
``beta`` the sharpness (``config.beta_s`` / ``config.beta_d``). Both peak at 1.0 at
``x = 0`` and have support scale ``1/beta`` — so switching surrogate does not rescale
gradient magnitude, and ``soma_grad_scale`` / LR carry over. See docs/architecture.md.
"""
import jax.numpy as jnp


def fast_sigmoid(x, beta):
    """1 / (1 + beta*|x|)^2 — smooth bump, peak 1.0 at threshold (default).

    A neuron sitting on its threshold gets full gradient; one far away gets almost
    none. Larger beta = narrower bump = sparser credit assignment.
    """
    return 1.0 / (1.0 + beta * jnp.abs(x)) ** 2


def ste_boxcar(x, beta):
    """Straight-through boxcar: 1.0 where |beta*x| <= 1, else 0.0.

    Support radius 1/beta in threshold-distance — the edge sits at fast_sigmoid's
    quarter-height point, so peak (1.0) and scale match. beta-aware so it keeps
    honouring the config sharpness. Cast to float64 for the x64 regime.
    """
    return (jnp.abs(beta * x) <= 1.0).astype(jnp.float64)


_SURROGATES = {"fast_sigmoid": fast_sigmoid, "ste": ste_boxcar}


def get_surrogate(name):
    """Resolve a surrogate name to its ``(x, beta) -> derivative`` function."""
    try:
        return _SURROGATES[name]
    except KeyError:
        raise ValueError(f"unknown surrogate {name!r}; options: {sorted(_SURROGATES)}")
