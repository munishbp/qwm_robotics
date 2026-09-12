"""Evaluation protocol. docs/design.md section 9.

One batch runs `num_envs` episodes from reset and records the outcome of the first episode in
every env. Later episodes in the same env are cut by the batch length, so they are not counted.
The success rate of a batch is the mean over envs. The reported number is the mean over batches
with its standard error.
"""

from __future__ import annotations

import math
import time

import torch

from swarm.belief import BeliefConfig
from swarm.buffer import Buffer
from swarm.rlpd import Agent
from swarm.rollout import Runner
from swarm.search import SearchConfig

HISTORY = 32


def run_batch(agent: Agent, env, belief_cfg: BeliefConfig, policy: str,
              search_cfg: SearchConfig | None, steps: int) -> dict[str, float]:
    E = env.num_envs
    dev = agent.device
    hist = Buffer(E, HISTORY, agent.types, dev, agent.cfg.full_dim)
    runner = Runner(env, hist, belief_cfg, agent.nets, agent.cfg.obs_mode)
    done_first = torch.zeros(E, dtype=torch.bool, device=dev)
    success = torch.zeros(E, dtype=torch.bool, device=dev)
    length = torch.zeros(E, device=dev)
    pos_err = torch.zeros(E, device=dev)
    ang_err = torch.zeros(E, device=dev)
    torch.cuda.synchronize()
    t0 = time.time()
    for t in range(steps):
        out = runner.step(policy, search_cfg=search_cfg)
        done = out["terminated"] | out["truncated"]
        first = done & ~done_first
        success |= first & out["terminated"]
        length = torch.where(first, torch.full_like(length, t + 1), length)
        pos_err = torch.where(first, out["info"]["final_pos_error"], pos_err)
        ang_err = torch.where(first, out["info"]["final_angle_error"], ang_err)
        done_first |= done
    torch.cuda.synchronize()
    wall = (time.time() - t0) / steps
    s = success.float()
    return {
        "success": s.mean().item(),
        "length_on_success": (length * s).sum().item() / max(s.sum().item(), 1),
        "pos_error": pos_err.mean().item(),
        "angle_error": ang_err.mean().item(),
        "ms_per_step": 1000 * wall,
        "finished": done_first.float().mean().item(),
    }


def evaluate(agent: Agent, make_env, belief_cfg: BeliefConfig, policy: str = "mean",
             search_cfg: SearchConfig | None = None, batches: int = 3, seed: int = 1000) -> dict:
    """Mean and standard error over batches. `make_env(seed)` builds a fresh env."""
    rows = []
    for i in range(batches):
        env = make_env(seed + i)
        rows.append(run_batch(agent, env, belief_cfg, policy, search_cfg, env.cfg.horizon))
    out = {}
    for k in rows[0]:
        vals = [r[k] for r in rows]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1)
        out[k] = mean
        out[k + "_se"] = math.sqrt(var / len(vals))
    out["batches"] = batches
    out["envs_per_batch"] = rows[0].get("envs", 0) or make_env(seed).num_envs
    return out
