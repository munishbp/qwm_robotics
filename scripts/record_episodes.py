"""Record episodes for the interactive viewer.

Usage: python scripts/record_episodes.py --ckpt checkpoints/belief.pt --out results/episodes.json
Every setting records `--per-setting` episodes from the same env seed, so the viewer can compare
the same start under different search and staleness settings.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import make_env, setup  # noqa: E402

from swarm.belief import BeliefConfig  # noqa: E402
from swarm.buffer import Buffer  # noqa: E402
from swarm.rlpd import Agent  # noqa: E402
from swarm.rollout import Runner  # noqa: E402
from swarm.search import SearchConfig  # noqa: E402

SETTINGS = [
    {"label": "no search, lag 1", "lag": 1, "depth": -1},
    {"label": "search depth 2, lag 1", "lag": 1, "depth": 2},
    {"label": "search depth 2, lag 4", "lag": 4, "depth": 2},
    {"label": "search depth 6, lag 4", "lag": 4, "depth": 6},
    {"label": "no search, lag 4", "lag": 4, "depth": -1},
]


@torch.no_grad()
def record(agent: Agent, env, setting: dict, per_setting: int) -> list[dict]:
    dev = agent.device
    E = env.num_envs
    buf = Buffer(E, 32, agent.types, dev, agent.cfg.full_dim)
    runner = Runner(env, buf, BeliefConfig(lag=setting["lag"]), agent.nets, agent.cfg.obs_mode)
    scfg = SearchConfig(depth=setting["depth"]) if setting["depth"] >= 0 else None
    policy = "search" if scfg else "mean"
    T = env.cfg.episode_length
    st = env.state()
    eps = [{"label": setting["label"], "setting": setting, "types": [int(x) for x in env.types],
            "goal": st["goal"][e].tolist(), "payload": [st["payload"][e].tolist()],
            "robots": [st["robot_pos"][e].tolist()], "latched": [st["latched"][e].int().tolist()],
            "ages": [], "actions": [], "success": False, "length": T} for e in range(per_setting)]
    done = [False] * per_setting
    for t in range(T):
        out = runner.step(policy, search_cfg=scfg)
        st = env.state()
        # The message table at t is the one the robots used to act at t.
        age = (t - buf.stamp[:, t % buf.C]).clamp(min=0)
        a = buf.action[:, t % buf.C]
        for e in range(per_setting):
            if done[e]:
                continue
            ep = eps[e]
            ep["ages"].append(age[e].tolist())
            ep["actions"].append(a[e].tolist())
            fin = bool(out["terminated"][e] | out["truncated"][e])
            if fin:
                ep["success"] = bool(out["terminated"][e])
                ep["length"] = t + 1
                done[e] = True
                # The env has reset, so the final pose comes from the last observation errors.
                ep["final_pos_error"] = float(out["info"]["final_pos_error"][e])
                ep["final_angle_error"] = float(out["info"]["final_angle_error"][e])
            else:
                ep["payload"].append(st["payload"][e].tolist())
                ep["robots"].append(st["robot_pos"][e].tolist())
                ep["latched"].append(st["latched"][e].int().tolist())
    return eps


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/belief.pt")
    p.add_argument("--per-setting", type=int, default=4)
    p.add_argument("--seed", type=int, default=3000)
    p.add_argument("--out", default="results/episodes.json")
    args = p.parse_args()
    dev = setup(args.seed)
    agent = Agent.load(args.ckpt, dev)
    episodes = []
    for setting in SETTINGS:
        env = make_env(16, args.seed, False, device=dev)
        episodes += record(agent, env, setting, args.per_setting)
        print(setting["label"], [e["success"] for e in episodes[-args.per_setting:]])
    cfg = env.cfg
    data = {"arena_half": cfg.arena_half, "payload_half": [cfg.payload_hx, cfg.payload_hy],
            "robot_radius": cfg.robot_radius, "dt": cfg.dt, "goal_pos_tol": cfg.goal_pos_tol,
            "goal_angle_tol": cfg.goal_angle_tol, "episodes": episodes}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(data, f)
    print(f"wrote {args.out} with {len(episodes)} episodes")


if __name__ == "__main__":
    main()
