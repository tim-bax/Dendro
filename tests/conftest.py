"""Shared test bootstrap.

Locks JAX into the golden's regime (CPU, float64) before jax.numpy is first imported,
and puts ``src/`` (the ``dendroprop`` package) on the path. The frozen artifacts under
``tests/golden/*.npz`` are the reference the tests replay against.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

_ROOT = Path(__file__).resolve().parent.parent
_SRC = str(_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
