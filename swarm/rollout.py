"""The live env loop shared by data collection, training, and evaluation.

The runner owns the env, the message tables, and a buffer that doubles as the observation
history. One `step` call advances every env by one step with the chosen policy and writes the
row that training samples later. See docs/design.md sections 5, 6.2 and 7.
"""

from __future__ import annotations

import torch

from swarm.belief import BeliefConfig, MessageTable, beliefs, beliefs_full
from swarm.buffer import Buffer
from swarm.nets import Nets
from swarm.search import SearchConfig, search


class Runner:
    def __init__(
        self, env, buffer: Buffer, belief_cfg: BeliefConfig, nets: Nets | None = None,
        obs_mode: str = "belief",
    ) -> None:
        self.env = env
        self.obs_mode = obs_mode
        self.buf = buffer
        self.cfg = belief_cfg
        self.nets = nets
        self.E, self.K = buffer.E, buffer.K
        self.device = buffer.device
        self.table = MessageTable(self.E, self.K, self.device)
        self.ep_start = torch.full((self.E,), buffer.t, dtype=torch.long, device=self.device)
        self.obs = env.reset()
        self.env_idx = torch.arange(self.E, device=self.device)
        self.search_stats: dict[str, list[float]] = {}

    @torch.no_grad()
    def step(self, policy: str, scripted=None, search_cfg: SearchConfig | None = None) -> dict:
        """policy is one of scripted, sample, mean, search. Returns the env step results."""
        t = self.buf.t
        stamp = self.table.update(t, self.ep_start, self.obs["comm"], self.cfg)
        self.buf.begin(self.obs["local"], self.ep_start, stamp, self.obs["full"])
        unc = torch.zeros(self.E, self.K, device=self.device)
        if policy == "scripted":
            a = scripted.act()
        else:
            row = torch.full_like(self.env_idx, t)
            if self.obs_mode == "full":
                out = beliefs_full(self.nets, self.buf, self.env_idx, row, False)
            else:
                out = beliefs(self.nets, self.buf, self.env_idx, row, False, self.cfg.mask_messages)
            b = out["b"]
            unc = self.nets.critic(b, self.nets.actor.mean(b)).std(0)
            if policy == "sample":
                a = self.nets.actor.sample(b)[0]
            elif policy == "mean":
                a = self.nets.actor.mean(b)
            elif policy == "search":
                if self.obs_mode == "full":
                    raise ValueError("search needs beliefs, not the full observation")
                # Robot i knows its own uncertainty fresh and every teammate's as of the message.
                unc_table = out["unc_table"].clone()
                unc_table.diagonal(dim1=-2, dim2=-1).copy_(unc)
                a, stats = search(self.nets, out["table"], out["feat"], self.buf.types, unc_table, search_cfg, t)
                for k, v in stats.items():
                    self.search_stats.setdefault(k, []).append(v)
            else:
                raise ValueError(f"unknown policy {policy}")
        target = self.env.state()["decoder_target"]
        obs, reward, terminated, truncated, info = self.env.step(a)
        done = terminated | truncated
        # The next table is the table the robots would hold at t + 1 if the episode continued.
        # An ended env keeps its last comm mask because the new episode's mask is unrelated.
        next_comm = torch.where(done.view(-1, 1, 1), self.obs["comm"], obs["comm"])
        keep = self.table.stamp.clone()
        next_stamp = self.table.update(t + 1, self.ep_start, next_comm, self.cfg)
        self.table.stamp = keep
        self.buf.finish(
            action=a, unc=unc, reward=reward, terminated=terminated, truncated=truncated,
            next_local=info["final_local"], next_stamp=next_stamp, target=target,
            next_full=info["final_full"],
        )
        self.ep_start = torch.where(done, torch.full_like(self.ep_start, t + 1), self.ep_start)
        self.obs = obs
        return {"reward": reward, "terminated": terminated, "truncated": truncated, "info": info}
