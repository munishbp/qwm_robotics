"""Combine the fair H1 arms across mjlab training seeds.

Usage: python scripts/seeds_summary.py > docs/results_tables_seeds.md
Reads runs/mjlab*/results/day1_controls.json (group F1) and the training logs, prints per seed
numbers and the mean with the standard deviation across seeds.
"""

from __future__ import annotations

import glob
import json
import os
import statistics as st


def main() -> None:
    runs = sorted(glob.glob("runs/mjlab*/results/day1_controls.json"))
    seeds = {}
    for path in runs:
        d = json.load(open(path))
        name = path.split("/")[1]
        rows = [r for r in d["rows"] if r.get("group") == "F1"]
        if rows:
            seeds[name] = rows
    print("### Fair H1 on mjlab across training seeds (256 envs, 5 batches per cell)\n")
    print("Per seed, the best evaluation success during training (2 batches of 128 envs):\n")
    for name in seeds:
        log = os.path.join("runs", name, "results", "train_belief.jsonl")
        if os.path.exists(log):
            best = max(json.loads(l)["eval_success"] for l in open(log) if l.strip())
            print(f"- {name}: {100 * best:.1f}")
    print()
    arms = ["sampled", "random_candidate", "depth0", "depth6"]
    labels = {"sampled": "Sampled policy", "random_candidate": "Random candidate", "depth0": "Depth 0", "depth6": "Depth 6, beta 0.9"}
    for lag in [0, 1, 4]:
        print(f"\n#### Lag {lag}\n")
        print("| Arm | " + " | ".join(seeds) + " | Mean ± SD across seeds |")
        print("|---|" + "---|" * (len(seeds) + 1))
        vals = {}
        for arm in arms:
            cells = []
            for name, rows in seeds.items():
                r = next((x for x in rows if x.get("arm") == arm and x.get("lag") == lag), None)
                cells.append(100 * r["success"] if r else float("nan"))
            vals[arm] = cells
            ok = [c for c in cells if c == c]
            sd = st.stdev(ok) if len(ok) > 1 else 0.0
            print(f"| {labels[arm]} | " + " | ".join(f"{c:.1f}" for c in cells) + f" | {st.mean(ok):.1f} ± {sd:.1f} |")
        gaps = [d6 - sp for d6, sp in zip(vals["depth6"], vals["sampled"]) if d6 == d6 and sp == sp]
        if gaps:
            sd = st.stdev(gaps) if len(gaps) > 1 else 0.0
            print(f"\nDepth 6 minus sampled policy per seed: " + ", ".join(f"{g:+.1f}" for g in gaps) + f"; mean {st.mean(gaps):+.1f} ± {sd:.1f}. Sign agrees in {sum(g > 0 for g in gaps)} of {len(gaps)} seeds.")


if __name__ == "__main__":
    main()
