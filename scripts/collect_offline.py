"""Collect the offline buffer with the scripted controller plus action noise.

Usage: python scripts/collect_offline.py [--envs 256] [--steps 800] [--noise 0.2] [--smoke]
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import make_env, setup, write_json  # noqa: E402

from swarm.belief import BeliefConfig  # noqa: E402
from swarm.buffer import Buffer  # noqa: E402
from swarm.env import full_dim  # noqa: E402
from swarm.rollout import Runner  # noqa: E402
from swarm.scripted import ScriptedController  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--envs", type=int, default=256)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--noise", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/offline.pt")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.envs, args.steps, args.out = 16, 200, "data/offline_smoke.pt"
    dev = setup(args.seed)
    env = make_env(args.envs, args.seed, training=True, device=dev)
    types = torch.tensor(env.types, device=dev)
    buf = Buffer(env.num_envs, args.steps, types, dev, full_dim(env.K))
    runner = Runner(env, buf, BeliefConfig(lag=1))
    ctrl = ScriptedController(env, noise=args.noise)
    successes = 0
    episodes = 0
    for t in range(args.steps):
        out = runner.step("scripted", scripted=ctrl)
        successes += int(out["terminated"].sum())
        episodes += int((out["terminated"] | out["truncated"]).sum())
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(buf.state_dict(), args.out)
    stats = {"transitions": buf.size, "episodes": episodes,
             "success_rate": successes / max(episodes, 1), "noise": args.noise, "envs": env.num_envs}
    print(stats)
    write_json(args.out.replace(".pt", ".json").replace("data/", "results/"), stats)


if __name__ == "__main__":
    main()
