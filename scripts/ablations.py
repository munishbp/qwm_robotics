"""Ablations on the snapshot that explain the main results.

Usage: python scripts/ablations.py --ckpt checkpoints/belief_best.pt
1. No search across staleness 0 to 8, to show how the baseline itself depends on the lag.
2. Messages masked (self only beliefs) at several lags, to show whether messages help at all.
3. The sampled policy against the mean policy, since exploration noise helps this task.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import make_env, setup, write_json  # noqa: E402
from evaluate import add_eval_args  # noqa: E402

from swarm.belief import BeliefConfig  # noqa: E402
from swarm.evaluate import evaluate  # noqa: E402
from swarm.rlpd import Agent  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    add_eval_args(p)
    p.set_defaults(envs=128)
    p.add_argument("--out", default="results/ablations.json")
    args = p.parse_args()
    dev = setup(args.seed)
    agent = Agent.load(args.ckpt, dev)
    rows = []

    def cell(name: str, policy: str, cfg: BeliefConfig) -> None:
        res = evaluate(agent, lambda s: make_env(args.envs, s, False, "default", dev), cfg, policy, None,
                       args.batches, args.seed)
        res.update({"name": name, "policy": policy, "lag": cfg.lag, "mask_messages": cfg.mask_messages})
        rows.append(res)
        print(f"{name}: success={res['success']:.3f}+-{res['success_se']:.3f}")

    for lag in [0, 1, 2, 4, 6, 8]:
        cell(f"no search, lag {lag}", "mean", BeliefConfig(lag=lag))
    for lag in [1, 2, 4]:
        cell(f"messages masked, lag {lag}", "mean", BeliefConfig(lag=lag, mask_messages=True))
    for lag in [1, 2]:
        cell(f"sampled policy, lag {lag}", "sample", BeliefConfig(lag=lag))
    write_json(args.out, {"rows": rows, "args": vars(args)})


if __name__ == "__main__":
    main()
