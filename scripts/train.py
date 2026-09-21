"""Train RLPD with beliefs or with the full state. docs/design.md section 6.

Usage: python scripts/train.py --obs belief --steps 12000 --out belief_seed0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import make_env, setup, write_json  # noqa: E402

from swarm.belief import BeliefConfig  # noqa: E402
from swarm.buffer import GUARD, Buffer  # noqa: E402
from swarm.env import full_dim  # noqa: E402
from swarm.evaluate import evaluate  # noqa: E402
from swarm.rlpd import Agent, RLPDConfig  # noqa: E402
from swarm.rollout import Runner  # noqa: E402
from swarm.search import SearchConfig  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--obs", choices=["belief", "full"], default="belief")
    p.add_argument("--envs", type=int, default=256)
    p.add_argument("--steps", type=int, default=12000, help="batched env steps")
    p.add_argument("--utd", type=int, default=4)
    p.add_argument("--lag", type=int, default=1)
    p.add_argument("--offline", default="data/offline.pt")
    p.add_argument("--no-offline", action="store_true")
    p.add_argument("--collect-depth", type=int, default=-1, help="search depth during collection, -1 for none")
    p.add_argument("--n-step", type=int, default=1, help="rows in the critic return window")
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-envs", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="belief")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.envs, args.steps, args.eval_every, args.eval_envs = 16, 60, 30, 16
        args.offline = "data/offline_smoke.pt"
        args.out = args.out + "_smoke"
    dev = setup(args.seed)
    env = make_env(args.envs, args.seed, training=True, device=dev)
    types = env.types
    # The full observation is stored only for the centralized run. The ring holds at most 4096
    # rows per env, about one million transitions at 256 envs, as the design states.
    fd = full_dim(env.num_robots) if args.obs == "full" else 0
    cfg = RLPDConfig(utd=args.utd, obs_mode=args.obs, full_dim=fd, n_step=args.n_step)
    agent = Agent(types, cfg, dev)
    online = Buffer(env.num_envs, min(args.steps + 1, 4096), types, dev, fd)
    offline = None if args.no_offline else Buffer.load(args.offline, dev)
    belief_cfg = BeliefConfig(lag=args.lag)
    runner = Runner(env, online, belief_cfg, agent.nets, args.obs)
    collect_search = SearchConfig(depth=args.collect_depth) if args.collect_depth >= 0 else None
    log_path = f"results/train_{args.out}.jsonl"
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    log = open(log_path, "w")
    ep_success, ep_count = 0, 0
    best = -1.0
    t0 = time.time()
    metrics: dict[str, float] = {}
    for step in range(args.steps):
        policy = "search" if collect_search is not None and step > GUARD else "sample"
        out = runner.step(policy, search_cfg=collect_search)
        ep_success += int(out["terminated"].sum())
        ep_count += int((out["terminated"] | out["truncated"]).sum())
        if step > GUARD:
            for _ in range(args.utd):
                metrics = agent.update(offline, online)
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            ev = evaluate(agent, lambda s: make_env(args.eval_envs, s, False, device=dev),
                          belief_cfg, "mean", None, batches=2)
            row = {"step": step + 1, "transitions": (step + 1) * env.num_envs,
                   "train_success": ep_success / max(ep_count, 1), "eval_success": ev["success"],
                   "eval_length": ev["length_on_success"], "wall_min": (time.time() - t0) / 60, **metrics}
            ep_success, ep_count = 0, 0
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}))
            agent.save(f"checkpoints/{args.out}.pt", {"step": step + 1, "args": vars(args)})
            # The snapshot every sweep uses is the best checkpoint by evaluation success. The
            # critic has shown a late decline in every run, so the last checkpoint is not the
            # best one. Ties go to the later checkpoint.
            if ev["success"] >= best:
                best = ev["success"]
                agent.save(f"checkpoints/{args.out}_best.pt", {"step": step + 1, "args": vars(args), "eval_success": best})
    log.close()
    write_json(f"results/train_{args.out}_final.json", {"args": vars(args), "last": row})


if __name__ == "__main__":
    main()
