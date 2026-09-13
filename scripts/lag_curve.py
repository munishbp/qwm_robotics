"""The fair H1 arms at lags 2 and 8, to turn the three staleness points into a curve.

Usage: SWARM_SIM=mjlab python scripts/lag_curve.py --ckpt checkpoints/belief_best.pt --out results/lag_curve.json
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import SIM, make_env, setup, write_json  # noqa: E402

from swarm.belief import BeliefConfig  # noqa: E402
from swarm.evaluate import evaluate  # noqa: E402
from swarm.rlpd import Agent  # noqa: E402
from swarm.search import SearchConfig  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/belief_best.pt")
    p.add_argument("--out", default="results/lag_curve.json")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--lags", type=int, nargs="+", default=[2, 8])
    args = p.parse_args()
    dev = setup(args.seed)
    agent = Agent.load(args.ckpt, dev)
    rows = []
    for lag in args.lags:
        for arm, policy, scfg in (("sampled", "sample", None), ("depth6", "search", SearchConfig(depth=6, beta=0.9))):
            res = evaluate(agent, lambda s: make_env(256, s, False, "default", dev), BeliefConfig(lag=lag), policy, scfg, 5, args.seed)
            res.update({"group": "F1", "arm": arm, "lag": lag, "sim": SIM, "envs": 256, "batches": 5})
            rows.append(res)
            print(f"lag {lag} {arm}: success={res['success']:.3f}+-{res['success_se']:.3f}", flush=True)
    write_json(args.out, {"rows": rows, "sim": SIM})


if __name__ == "__main__":
    main()
