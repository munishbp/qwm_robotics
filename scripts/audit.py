"""Reward hacking audit. Usage: SWARM_SIM=2d python scripts/audit.py --ckpt checkpoints/belief_best.pt

Checks that a null policy earns nothing, that no episode succeeds at reset or within five steps,
that the payload never moves faster than the physics allows within an episode, and that no NaN
appears under the trained policy. Static checks on the test time code path are listed in
docs/results.md section 15.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import SIM, make_env, setup, write_json  # noqa: E402

from swarm.belief import BeliefConfig  # noqa: E402
from swarm.buffer import Buffer  # noqa: E402
from swarm.rlpd import Agent  # noqa: E402
from swarm.rollout import Runner  # noqa: E402


@torch.no_grad()
def run(policy, label, ckpt, dev, E=128, seed=4242, depth=-1) -> dict:
    env = make_env(E, seed, False, device=dev)
    st = env.state()
    gap0 = (st["payload"][:, :2] - st["goal"][:, :2]).norm(dim=-1)
    at_reset = (gap0 < env.cfg.pos_tol).float().mean().item()
    done = torch.zeros(E, dtype=torch.bool, device=dev)
    succ, early = done.clone(), done.clone()
    maxjump = torch.zeros(E, device=dev)
    nan = False
    runner = None
    if ckpt:
        ag = Agent.load(ckpt, dev)
        runner = Runner(env, Buffer(E, 32, env.types, dev), BeliefConfig(1), ag.nets)
    # The runner resets the env once more, so the reference pose is read after it exists.
    st = env.state()
    gap0 = (st["payload"][:, :2] - st["goal"][:, :2]).norm(dim=-1)
    at_reset = (gap0 < env.cfg.pos_tol).float().mean().item()
    prev = st["payload"].clone()
    for t in range(env.cfg.horizon):
        if runner:
            from swarm.search import SearchConfig

            out = runner.step("search", search_cfg=SearchConfig(depth=depth, beta=0.9)) if depth >= 0 else runner.step("mean")
            term, trunc = out["terminated"], out["truncated"]
        else:
            _, _, term, trunc, _ = env.step(policy(env))
        cur = env.state()["payload"]
        nan |= bool(torch.isnan(cur).any())
        # The pose after a reset belongs to a new episode, so the jump at an ended step is skipped.
        jump = (cur[:, :2] - prev[:, :2]).norm(dim=-1)
        maxjump = torch.where(done | term | trunc, maxjump, torch.maximum(maxjump, jump))
        prev = cur.clone()
        first = (term | trunc) & ~done
        succ |= first & term
        early |= first & term & (t < 5)
        done |= term | trunc
    res = {"policy": label, "success": succ.float().mean().item(), "success_at_reset": at_reset,
           "success_in_first_5_steps": early.float().mean().item(),
           "max_payload_step_m": maxjump.max().item(), "nan": nan}
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()})
    return res


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/belief_best.pt")
    p.add_argument("--out", default="results/audit.json")
    p.add_argument("--depth", type=int, default=-1, help="also audit the search policy at this depth")
    args = p.parse_args()
    dev = setup(0)
    rows = [
        run(lambda env: torch.zeros(env.num_envs, env.num_robots, 3, device=dev), "zero actions", None, dev),
        run(lambda env: torch.rand(env.num_envs, env.num_robots, 3, device=dev) * 2 - 1, "random actions", None, dev),
        run(None, "trained mean policy", args.ckpt, dev),
    ]
    if args.depth >= 0:
        rows.append(run(None, f"trained policy with depth {args.depth} search", args.ckpt, dev, depth=args.depth))
    write_json(args.out, {"sim": SIM, "rows": rows})


if __name__ == "__main__":
    main()
