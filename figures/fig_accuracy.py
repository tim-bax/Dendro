"""Accuracy / firing curves from run dirs — self-contained, NO JAX, NO training import.

Reads ``config.json`` + ``metrics.csv`` with the stdlib only, so this module (and the
whole ``figures/`` tree) never imports the ``dendroprop`` package or JAX. matplotlib is
imported lazily inside the plotting functions, so importing this module needs only the
standard library.

    python figures/fig_accuracy.py runs/shd_bptt runs/shd_dfa -o acc.png
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Tuple


def load_run(run_dir) -> Tuple[dict, List[dict]]:
    """(config dict, metrics rows) from a run dir. Pure stdlib — no JAX, no dendroprop."""
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    rows: List[dict] = []
    with open(run_dir / "metrics.csv", newline="") as f:
        for raw in csv.DictReader(f):
            rows.append(
                {k: (int(v) if k == "epoch" else float(v)) for k, v in raw.items()}
            )
    return config, rows


def _run_label(config: dict, run_dir) -> str:
    method = config.get("credit_method", "?")
    return f"{Path(run_dir).name} ({method})"


def plot_accuracy(run_dirs, out_path):
    """Train/test accuracy vs epoch, one pair of lines per run. Returns out_path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for rd in run_dirs:
        config, rows = load_run(rd)
        epochs = [r["epoch"] for r in rows]
        label = _run_label(config, rd)
        ax.plot(epochs, [r["train_acc"] for r in rows], label=f"{label} train")
        ax.plot(epochs, [r["test_acc"] for r in rows], linestyle="--",
                label=f"{label} test")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    ax.set_title("Accuracy")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_firing(run_dirs, out_path):
    """Mean hidden firing rate vs epoch, one line per run. Returns out_path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for rd in run_dirs:
        config, rows = load_run(rd)
        ax.plot([r["epoch"] for r in rows], [r["firing"] for r in rows],
                label=_run_label(config, rd))
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean hidden firing rate")
    ax.legend(fontsize=7)
    ax.set_title("Firing rate")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(argv=None):
    p = argparse.ArgumentParser(description="Plot accuracy/firing curves from run dirs.")
    p.add_argument("run_dirs", nargs="+", help="one or more runs/<name>/ directories")
    p.add_argument("-o", "--out", default="accuracy.png", help="output image path")
    p.add_argument("--firing", action="store_true", help="also write a firing-rate plot")
    args = p.parse_args(argv)

    out = plot_accuracy(args.run_dirs, args.out)
    print(f"[fig_accuracy] wrote {out}")
    if args.firing:
        fout = Path(args.out).with_name("firing.png")
        plot_firing(args.run_dirs, fout)
        print(f"[fig_accuracy] wrote {fout}")


if __name__ == "__main__":
    main()
