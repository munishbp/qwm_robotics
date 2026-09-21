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
    p.add_argument("--policy", choices=["mean", "sample"], default="mean", help="no search policy")
    p.add_argument("--no-messages", action="store_true", help="ablation: self only beliefs")
    p.add_argument("--no-roll", action="store_true", help="ablation: stale messages used without forward correction")
    p.add_argument("--scorer", choices=["q", "decoded", "random"], default="q", help="search node score")
    p.add_argument("--gate", type=float, default=0.0, help="search only where r is at or above this value; 0 disables")
    p.add_argument("--gate-fallback", choices=["sample", "mean"], default="sample", help="action of a gated row")
    p.add_argument("--gate-shuffle", action="store_true", help="control: permute r across rows")
    p.add_argument("--lcb", type=float, default=2.0, help="score is the ensemble mean minus this many std; 0 reproduces results.md sections 3 to 20")
    p.add_argument("--wm", choices=["trained", "random"], default="trained", help="control: random world model")
    p.add_argument("--out", default="")
    args = p.parse_args()
    dev = setup(args.seed)
    agent = Agent.load(args.ckpt, dev)
    belief_cfg = BeliefConfig(lag=args.lag, dropout=args.dropout, mask_messages=args.no_messages,
                              roll_messages=not args.no_roll)
    if args.wm == "random":
        from swarm.nets import WorldModel

        agent.nets.wm = WorldModel().to(dev)
    if args.depth >= 0:
        policy, scfg = "search", SearchConfig(
            args.depth, args.candidates, args.beam, args.beta, args.mode, args.scorer,
            args.gate, args.gate_fallback, args.gate_shuffle, args.lcb)
    else:
        policy, scfg = args.policy, None
    res = evaluate(agent, lambda s: make_env(args.envs, s, False, args.team, dev), belief_cfg, policy, scfg,
                   args.batches, args.seed)
    res["args"] = vars(args)
    print({k: round(v, 4) for k, v in res.items() if isinstance(v, float)})
    if args.out:
        write_json(args.out, res)


if __name__ == "__main__":
    main()
