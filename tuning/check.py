#!/usr/bin/env python3
"""Progress + health check for the running Optuna studies.

Reads each tuning/*.db (safe to run while the studies are training) and prints,
per study: trial counts and the current epoch of the running trial + best value.
Also prints host RAM (free) and GPU memory. Run any time:

    python tuning/check.py
"""
import glob
import os
import subprocess

_DIR = os.path.dirname(os.path.abspath(__file__))

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


def host_ram_gb():
    """(total, available) GB from /proc/meminfo, no external deps."""
    vals = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":")
            vals[k] = int(v.strip().split()[0])  # kB
    return vals["MemTotal"] / 1048576, vals["MemAvailable"] / 1048576


def gpu_mem():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"], text=True).strip()
        return out
    except Exception as e:
        return f"(nvidia-smi failed: {e})"


print("=== studies ===")
for db in sorted(glob.glob(os.path.join(_DIR, "*.db"))):
    name = os.path.basename(db)[:-3]
    try:
        s = optuna.load_study(study_name=name, storage=f"sqlite:///{db}")
    except Exception as e:
        print(f"{name:22s} : could not load ({e})")
        continue
    trials = s.trials
    done = [t for t in trials if t.state.name == "COMPLETE"]
    pruned = [t for t in trials if t.state.name == "PRUNED"]
    running = [t for t in trials if t.state.name == "RUNNING"]
    last_epoch = -1
    for t in running:
        if t.intermediate_values:
            last_epoch = max(last_epoch, max(t.intermediate_values.keys()))
    ep = f"epoch {last_epoch}" if last_epoch >= 0 else "warming up"
    best = f"{s.best_value:.2f}%" if done else "--"
    print(f"{name:22s} : {len(trials):4d} trials "
          f"(done={len(done)} pruned={len(pruned)} running={len(running)}) | "
          f"running@{ep} | best={best}")

tot, avail = host_ram_gb()
print(f"\n=== host RAM ===  {tot:.0f} GB total, {avail:.0f} GB available "
      f"({'OK' if avail > 6 else 'LOW - consider killing one study'})")
print(f"=== GPU (util%, used MiB, total MiB) ===  {gpu_mem()}")
