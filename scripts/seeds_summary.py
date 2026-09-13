"""Combine the fair H1 arms across mjlab training seeds, with paired bootstrap intervals.

Usage: python scripts/seeds_summary.py > docs/results_tables_seeds.md
Reads runs/mjlab*/results/day1_controls.json (group F1) and runs/mjlab*/results/lag_curve.json.
Two arms evaluated on the same env seeds share their first episodes, so the difference between
them is paired at the episode level. Where per episode outcomes exist the interval is a bootstrap
over 1,000 resamples of the paired differences; otherwise only the batch means are shown.
"""

from __future__ import annotations

import glob
import json
import os
import random
import statistics as st

LAGS = [0, 1, 2, 4, 8]
ARMS = ["sampled", "random_candidate", "depth0", "depth6"]
LABELS = {"sampled": "Sampled policy", "random_candidate": "Random candidate", "depth0": "Depth 0", "depth6": "Depth 6, beta 0.9"}


def load_rows(run: str) -> list[dict]:
    rows = []
    for name in ("day1_controls.json", "lag_curve.json"):
        path = os.path.join(run, "results", name)
        if os.path.exists(path):
            rows += [r for r in json.load(open(path))["rows"] if r.get("group") == "F1"]
    return rows


def find(rows, arm, lag):
    return next((r for r in rows if r.get("arm") == arm and r.get("lag") == lag), None)


def paired_bootstrap(a: dict, b: dict, n: int = 1000, seed: int = 0):
    """95 percent interval of mean(b) - mean(a) over paired episodes, in points."""
    if "per_batch_success" not in a or "per_batch_success" not in b:
        return None
    pa = [x for batch in a["per_batch_success"] for x in batch]
    pb = [x for batch in b["per_batch_success"] for x in batch]
    if len(pa) != len(pb):
        return None
    diffs = [y - x for x, y in zip(pa, pb)]
    rng = random.Random(seed)
    means = []
    m = len(diffs)
    for _ in range(n):
        sample = [diffs[rng.randrange(m)] for _ in range(m)]
        means.append(100 * sum(sample) / m)
    means.sort()
    return 100 * sum(diffs) / m, means[int(0.025 * n)], means[int(0.975 * n)]


def main() -> None:
    runs = sorted(d for d in glob.glob("runs/mjlab*") if load_rows(d))
    seeds = {os.path.basename(d): load_rows(d) for d in runs}
    print("### Fair H1 on mjlab across training seeds (256 envs, 5 batches per cell)\n")
    print("Per seed, the best evaluation success during training (2 batches of 128 envs):\n")
    for name in seeds:
        log = os.path.join("runs", name, "results", "train_belief.jsonl")
        if os.path.exists(log):
            best = max(json.loads(l)["eval_success"] for l in open(log) if l.strip())
            print(f"- {name}: {100 * best:.1f}")
    print()
    for lag in LAGS:
        if not any(find(rows, "depth6", lag) for rows in seeds.values()):
            continue
        print(f"\n#### Lag {lag}\n")
        print("| Arm | " + " | ".join(seeds) + " | Mean ± SD across seeds |")
        print("|---|" + "---|" * (len(seeds) + 1))
        vals = {}
        for arm in ARMS:
            cells = []
            for rows in seeds.values():
                r = find(rows, arm, lag)
                cells.append(100 * r["success"] if r else float("nan"))
            ok = [c for c in cells if c == c]
            if not ok:
                continue
            vals[arm] = cells
            sd = st.stdev(ok) if len(ok) > 1 else 0.0
            print(f"| {LABELS[arm]} | " + " | ".join("-" if c != c else f"{c:.1f}" for c in cells) + f" | {st.mean(ok):.1f} ± {sd:.1f} |")
        gaps, cis = [], []
        for name, rows in seeds.items():
            a, b = find(rows, "sampled", lag), find(rows, "depth6", lag)
            if a and b:
                gaps.append(100 * (b["success"] - a["success"]))
                bs = paired_bootstrap(a, b)
                cis.append(f"{name}: {bs[0]:+.1f} [{bs[1]:+.1f}, {bs[2]:+.1f}]" if bs else f"{name}: {gaps[-1]:+.1f} [no per episode data]")
        if gaps:
            sd = st.stdev(gaps) if len(gaps) > 1 else 0.0
            print(f"\nDepth 6 minus sampled policy: mean {st.mean(gaps):+.1f} ± {sd:.1f} across {len(gaps)} seeds, sign positive in {sum(g > 0 for g in gaps)} of {len(gaps)}.")
            print("Paired bootstrap 95 percent intervals per seed: " + "; ".join(cis) + ".")


if __name__ == "__main__":
    main()
