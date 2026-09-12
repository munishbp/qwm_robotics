"""Open loop world model error against horizon on held out rollouts.

Usage: python scripts/world_model_error.py --ckpt checkpoints/belief.pt
For horizon k the script encodes the latent at t, rolls it k steps with the real recorded joint
actions, and compares with the encoding at t + k in latent units and, through the decoder, in
state units. The copy baseline predicts no change.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import make_env, setup, write_json  # noqa: E402

from swarm.belief import BeliefConfig  # noqa: E402
from swarm.buffer import Buffer  # noqa: E402
from swarm.nets import ACT_DIM  # noqa: E402
from swarm.rlpd import Agent  # noqa: E402
from swarm.rollout import Runner  # noqa: E402

HORIZONS = [1, 2, 3, 4, 6, 8]


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/belief.pt")
    p.add_argument("--envs", type=int, default=128)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=2000)
    p.add_argument("--out", default="results/wm_error.json")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.envs, args.steps, args.out = 16, 60, "results/wm_error_smoke.json"
    dev = setup(args.seed)
    agent = Agent.load(args.ckpt, dev)
    env = make_env(args.envs, args.seed, False, device=dev)
    buf = Buffer(env.num_envs, args.steps, agent.types, dev, agent.cfg.full_dim)
    runner = Runner(env, buf, BeliefConfig(lag=1), agent.nets, agent.cfg.obs_mode)
    for _ in range(args.steps):
        runner.step("mean")
    nets = agent.nets
    K = buf.K
    A = env.cfg.arena_half
    eye = torch.eye(K, dtype=torch.bool, device=dev)
    rows = {}
    for k in HORIZONS:
        env_idx, row = buf.sample_rows(4096)
        # Keep rows whose k successors lie in the same episode and inside the buffer.
        slot = buf._slot(row)
        end = buf.ep_start[env_idx, buf._slot((row + k).clamp(max=buf.t - 1))]
        ok = (row + k <= buf.t - 1) & (end == buf.ep_start[env_idx, slot])
        env_idx, row = env_idx[ok], row[ok]
        types = buf.types.view(1, K).expand(row.numel(), K)
        z = nets.enc(buf.stack(env_idx, row), types)
        for s in range(k):
            a = buf.action[env_idx, buf._slot(row + s)]
            others = a.unsqueeze(1).expand(-1, K, K, ACT_DIM)
            ctx = nets.wm.context(buf.types.view(1, 1, K).expand(row.numel(), K, K), others, (~eye).expand(row.numel(), K, K))
            z = nets.wm(z, a, ctx)
        z_true = nets.enc(buf.stack(env_idx, row + k), types)
        z0 = nets.enc(buf.stack(env_idx, row), types)
        target = buf.target[env_idx, buf._slot(row + k)]
        dec = nets.dec(z)
        pos_err = ((dec[..., :2] - target[..., :2]) * A).norm(dim=-1)
        ang_err = torch.atan2(dec[..., 3], dec[..., 2]) - torch.atan2(target[..., 3], target[..., 2])
        ang_err = torch.atan2(torch.sin(ang_err), torch.cos(ang_err)).abs()
        dec0 = nets.dec(z0)
        pos0 = ((dec0[..., :2] - target[..., :2]) * A).norm(dim=-1)
        rows[k] = {
            "latent_mse": (z - z_true).pow(2).mean().item(),
            "copy_latent_mse": (z0 - z_true).pow(2).mean().item(),
            "decoded_pos_error_m": pos_err.mean().item(),
            "copy_decoded_pos_error_m": pos0.mean().item(),
            "decoded_angle_error_rad": ang_err.mean().item(),
            "samples": int(row.numel()),
        }
        print(k, {a: round(b, 4) for a, b in rows[k].items()})
    write_json(args.out, {"horizons": rows, "args": vars(args)})


if __name__ == "__main__":
    main()
