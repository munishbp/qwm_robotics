"""Measure the critic's action span across the policy's own candidates on a snapshot.

Usage: SWARM_SIM=2d python scripts/critic_span.py --ckpt checkpoints/belief_best.pt --out results/critic_action_span.json
Reports the span of Q across nine root candidates (the mean action plus eight policy samples), the
ensemble spread at the mean action, their ratio r, how often the argmax picks the mean action, and
the distances of the chosen and the average candidate from the mean.
"""

from __future__ import annotations

import argparse
import os
import statistics as st
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import SIM, make_env, setup, write_json  # noqa: E402

from swarm.belief import BeliefConfig, beliefs  # noqa: E402
from swarm.buffer import Buffer  # noqa: E402
from swarm.rlpd import Agent  # noqa: E402
from swarm.rollout import Runner  # noqa: E402


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/belief_best.pt")
    p.add_argument("--out", default="results/critic_action_span.json")
    p.add_argument("--steps", type=int, default=120)
    args = p.parse_args()
    dev = setup(0)
    ag = Agent.load(args.ckpt, dev)
    n = ag.nets
    env = make_env(128, 7, False, device=dev)
    buf = Buffer(128, 32, env.types, dev)
    r = Runner(env, buf, BeliefConfig(1), n)
    acc: dict[str, list[float]] = {k: [] for k in ("q_span_across_candidates", "ensemble_spread_at_mean",
                                                  "argmax_picks_mean_fraction", "distance_of_chosen_to_mean",
                                                  "distance_of_average_candidate_to_mean")}
    torch.manual_seed(0)
    for _ in range(args.steps):
        r.step("mean")
        row = torch.full_like(r.env_idx, buf.t - 1)
        b = beliefs(n, buf, r.env_idx, row, False)["b"]
        mu = n.actor.mean(b)
        samp = n.actor.sample_n(b, 8)
        cand = torch.cat([mu.unsqueeze(2), samp], 2)
        q = n.critic(b.unsqueeze(2).expand(-1, -1, 9, -1), cand)
        qm = q.mean(0)
        best = qm.argmax(-1)
        chosen = cand.gather(2, best[..., None, None].expand(-1, -1, 1, 3)).squeeze(2)
        acc["q_span_across_candidates"].append((qm.max(-1).values - qm.min(-1).values).mean().item())
        acc["ensemble_spread_at_mean"].append(q.std(0)[..., 0].mean().item())
        acc["argmax_picks_mean_fraction"].append((best == 0).float().mean().item())
        acc["distance_of_chosen_to_mean"].append((chosen - mu).norm(dim=-1).mean().item())
        acc["distance_of_average_candidate_to_mean"].append((samp - mu.unsqueeze(2)).norm(dim=-1).mean().item())
    out = {k: st.mean(v) for k, v in acc.items()}
    out["ratio_r"] = out["q_span_across_candidates"] / out["ensemble_spread_at_mean"]
    out["sim"] = SIM
    out["ckpt"] = args.ckpt
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()})
    write_json(args.out, out)


if __name__ == "__main__":
    main()
