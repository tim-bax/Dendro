"""Persistence + run-dir artifacts.

Two audiences, cleanly separated:
  - the training side calls :func:`write_run` / :func:`save_model` / :func:`load_model`;
  - ``figures/`` call :func:`load_run` (config.json + metrics.csv) and never touch JAX.

To keep the reader half JAX-free, JAX/model imports are LAZY (inside the model
functions), so importing this module — or ``load_run`` — pulls only stdlib + numpy.
A run directory is self-describing::

    runs/<name>/
      config.json   {"run": RunConfig, "neuron": NeuronConfig, "dims", "credit_method"}
      metrics.csv   one row per epoch (train.METRIC_FIELDS)
      model.npz     weights + T_p + alphas + feedback B (FA) + a JSON meta blob
      log.txt       optional stdout capture
"""
from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .train import METRIC_FIELDS, MetricsHistory


# --------------------------------------------------------------------------- #
# Run directory                                                               #
# --------------------------------------------------------------------------- #
def _net_dims(net) -> list:
    """(n_inputs, [hidden dims], n_outputs) from a Network."""
    return [
        int(net.hidden[0].n_in),
        [int(layer.n) for layer in net.hidden],
        int(net.readout.w.shape[0]),
    ]


def write_run(
    run_dir,
    run_config,
    history: MetricsHistory,
    net=None,
    *,
    log: Optional[str] = None,
) -> Path:
    """Write a self-describing run directory; return its path.

    ``config.json`` + ``metrics.csv`` always; ``model.npz`` when ``net`` is given;
    ``log.txt`` when ``log`` is given.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {"run": dataclasses.asdict(run_config)}
    if net is not None:
        config["neuron"] = dataclasses.asdict(net.config)
        config["dims"] = _net_dims(net)
        config["credit_method"] = net.credit_method
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))

    with open(run_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(METRIC_FIELDS))
        writer.writeheader()
        writer.writerows(history.rows())

    if net is not None:
        save_model(run_dir / "model.npz", net)
    if log is not None:
        (run_dir / "log.txt").write_text(log)
    return run_dir


def load_run(run_dir) -> Tuple[dict, List[dict]]:
    """(config dict, metrics rows) for a run — JAX-free, for ``figures/``.

    Numeric metric columns are parsed to floats (``epoch`` to int); ``config`` is the
    parsed config.json.
    """
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    rows: List[dict] = []
    with open(run_dir / "metrics.csv", newline="") as f:
        for raw in csv.DictReader(f):
            row = {}
            for k, v in raw.items():
                if k == "epoch":
                    row[k] = int(v)
                else:
                    row[k] = float(v)
            rows.append(row)
    return config, rows


# --------------------------------------------------------------------------- #
# Model save / load (JAX imports are lazy so figures stay pure)               #
# --------------------------------------------------------------------------- #
def save_model(path, net) -> Path:
    """Serialize a ``Network`` to a single ``.npz`` (weights + intrinsics + B + meta)."""
    path = Path(path)
    arrays = {}
    per_layer = []
    for i, layer in enumerate(net.hidden):
        arrays[f"wd_{i}"] = np.asarray(layer.w_dend)
        arrays[f"ws_{i}"] = np.asarray(layer.w_soma)
        arrays[f"Tp_{i}"] = np.asarray(layer.T_p)
        arrays[f"alpha_s_{i}"] = np.asarray(layer.alpha_s)
        arrays[f"alpha_d_{i}"] = np.asarray(layer.alpha_d)
        arrays[f"alpha_w_{i}"] = np.asarray(layer.alpha_w)
        per_layer.append(
            {"surrogate": layer.surrogate, "dendritic_credit": bool(layer.dendritic_credit)}
        )
    arrays["w_readout"] = np.asarray(net.readout.w)
    arrays["alpha_m"] = np.asarray(net.readout.alpha_m)
    for i, b in enumerate(net.B):
        arrays[f"B_{i}"] = np.asarray(b)

    meta = {
        "config": dataclasses.asdict(net.config),
        "credit_method": net.credit_method,
        "n_layers": len(net.hidden),
        "per_layer": per_layer,
        "has_B": len(net.B) > 0,
        "dims": _net_dims(net),
    }
    arrays["meta_json"] = np.array(json.dumps(meta, sort_keys=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return path


def load_model(path):
    """Reconstruct a ``Network`` saved by :func:`save_model`. (Imports JAX lazily.)"""
    import jax.numpy as jnp

    from .config import NeuronConfig
    from .layers import ReadoutLayer, TwoCompLayer
    from .network import Network

    with np.load(path, allow_pickle=False) as npz:
        meta = json.loads(str(npz["meta_json"]))
        n_layers = int(meta["n_layers"])
        hidden = tuple(
            TwoCompLayer(
                w_dend=jnp.asarray(npz[f"wd_{i}"]),
                w_soma=jnp.asarray(npz[f"ws_{i}"]),
                T_p=jnp.asarray(npz[f"Tp_{i}"]).astype(jnp.int32),
                alpha_s=jnp.asarray(npz[f"alpha_s_{i}"]),
                alpha_d=jnp.asarray(npz[f"alpha_d_{i}"]),
                alpha_w=jnp.asarray(npz[f"alpha_w_{i}"]),
                surrogate=meta["per_layer"][i]["surrogate"],
                dendritic_credit=meta["per_layer"][i]["dendritic_credit"],
            )
            for i in range(n_layers)
        )
        readout = ReadoutLayer(
            w=jnp.asarray(npz["w_readout"]), alpha_m=jnp.asarray(npz["alpha_m"])
        )
        B = (
            tuple(jnp.asarray(npz[f"B_{i}"]) for i in range(n_layers))
            if meta["has_B"]
            else ()
        )
        config = NeuronConfig(**meta["config"])

    return Network(
        hidden=hidden, readout=readout, config=config,
        credit_method=meta["credit_method"], B=B,
    )
