#!/usr/bin/env python3
"""Terminal viewer for a Soft_Reset Optuna study (safe to run while it trains).

Prints trial counts, the best trial + its params, and the top-N trials. Optionally
param importances (needs scikit-learn). Defaults target the augmentation study; pass
--storage / --study_name for any other study.

    python Soft_Reset/show_study.py
    python Soft_Reset/show_study.py --top 20
    python Soft_Reset/show_study.py --storage sqlite:///Soft_Reset/soft_reset_shd_512.db \
        --study_name soft_reset_shd_512
"""
import argparse

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--storage", type=str,
                   default="sqlite:///Soft_Reset/soft_reset_shd_augment.db")
    p.add_argument("--study_name", type=str, default="soft_reset_shd_512_augment")
    p.add_argument("--top", type=int, default=10, help="How many best trials to list.")
    args = p.parse_args()

    s = optuna.load_study(study_name=args.study_name, storage=args.storage)
    done = [t for t in s.trials if t.state.name == "COMPLETE"]
    pruned = [t for t in s.trials if t.state.name == "PRUNED"]
    running = [t for t in s.trials if t.state.name == "RUNNING"]
    last_epoch = max((max(t.intermediate_values) for t in running
                      if t.intermediate_values), default=-1)

    print(f"study: {args.study_name}   storage: {args.storage}")
    print(f"trials={len(s.trials)}  done={len(done)}  pruned={len(pruned)}  "
          f"running={len(running)}"
          + (f" (running@epoch {last_epoch})" if last_epoch >= 0 else ""))
    if not done:
        print("no completed trials yet."); return

    bt = s.best_trial
    print(f"\nbest: {s.best_value:.3f}%  (trial #{bt.number})")
    for k, v in bt.params.items():
        print(f"   {k} = {v}")

    valued = [t for t in s.trials if t.value is not None]
    top = sorted(valued, key=lambda t: t.value, reverse=True)[:args.top]
    keys = list(top[0].params.keys())
    print(f"\nTop {len(top)} trials:")
    print("   #    val%  " + "  ".join(f"{k[:10]:>10}" for k in keys))
    for t in top:
        row = "  ".join(
            (f"{t.params[k]:>10.4g}" if isinstance(t.params[k], float)
             else f"{t.params[k]:>10}") for k in keys)
        print(f"{t.number:>4} {t.value:7.2f}  {row}")

    try:
        imp = optuna.importance.get_param_importances(s)
        print("\nParam importances:")
        for k, v in imp.items():
            print(f"   {v:6.3f}  {k}")
    except Exception as e:
        print(f"\n(param importances unavailable: {str(e).splitlines()[0]})")
        print(" -> pip install scikit-learn  in the dendroprop env to enable them.")


if __name__ == "__main__":
    main()
