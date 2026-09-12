"""Evaluate a checkpoint at test time settings.

Usage: python scripts/evaluate.py --ckpt checkpoints/belief.pt --depth 2 --lag 1 --beta 0.5
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import make_env, setup, write_json  # noqa: E402

from swarm.belief import BeliefConfig  # noqa: E402
from swarm.evaluate import evaluate  # noqa: E402
from swarm.rlpd import Agent  # noqa: E402
from swarm.search import SearchConfig  # noqa: E402


def add_eval_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--ckpt", default="checkpoints/belief.pt")
    p.add_argument("--envs", type=int, default=256)
    p.add_argument("--batches", type=int, default=3)
    p.add_argument("--team", default="default")
    p.add_argument("--seed", type=int, default=1000)


def main() -> None:
    p = argparse.ArgumentParser()
    add_eval_args(p)
    p.add_argument("--depth", type=int, default=-1, help="-1 means no search")
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--candidates", type=int, default=8)
    p.add_argument("--beam", type=int, default=4)
    p.add_argument("--mode", default="independent")
    p.add_argument("--lag", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--out", default="")
    args = p.parse_args()
    dev = setup(args.seed)
    agent = Agent.load(args.ckpt, dev)
    belief_cfg = BeliefConfig(lag=args.lag, dropout=args.dropout)
    if args.depth >= 0:
        policy, scfg = "search", SearchConfig(args.depth, args.candidates, args.beam, args.beta, args.mode)
    else:
        policy, scfg = "mean", None
    res = evaluate(agent, lambda s: make_env(args.envs, s, False, args.team, dev), belief_cfg, policy, scfg,
                   args.batches, args.seed)
    res["args"] = vars(args)
    print({k: round(v, 4) for k, v in res.items() if isinstance(v, float)})
    if args.out:
        write_json(args.out, res)


if __name__ == "__main__":
    main()
