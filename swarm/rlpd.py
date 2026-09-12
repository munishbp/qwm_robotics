"""RLPD update on per robot transitions. docs/design.md section 6.6.

One update samples half a batch from the offline buffer and half from the online buffer, builds
beliefs at t and t + 1 with the current encoders, and applies the SAC losses, the world model
loss, and the decoder loss. The RL losses shape the encoders. The world model chases them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F

from swarm.buffer import Buffer
from swarm.belief import beliefs, beliefs_full
from swarm.nets import ACT_DIM, ENC_IN, STACK, Nets


@dataclass
class RLPDConfig:
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    batch: int = 256
    num_critics: int = 10
    target_entropy: float = -float(ACT_DIM)
    utd: int = 4
    # Under a sparse terminal reward the entropy stream rewards a long episode, so a critic that
    # backs up the entropy term learns to stall. RLPD's backup_entropy=False leaves it out.
    backup_entropy: bool = False
    init_alpha: float = 0.1
    obs_mode: str = "belief"  # belief or full
    full_dim: int = 0


class Agent:
    def __init__(self, types: torch.Tensor, cfg: RLPDConfig, device: str) -> None:
        self.cfg = cfg
        self.device = device
        self.types = types.to(device)
        enc_in = STACK * cfg.full_dim + ACT_DIM if cfg.obs_mode == "full" else ENC_IN
        self.nets = Nets(cfg.num_critics, enc_in).to(device)
        with torch.no_grad():
            self.nets.log_alpha.fill_(math.log(cfg.init_alpha))
        self.belief_fn = beliefs_full if cfg.obs_mode == "full" else beliefs
        n = self.nets
        self.opt_critic = torch.optim.Adam(
            list(n.enc.parameters()) + list(n.fuse.parameters()) + list(n.critic.parameters()), lr=cfg.lr
        )
        self.opt_actor = torch.optim.Adam(n.actor.parameters(), lr=cfg.lr)
        self.opt_alpha = torch.optim.Adam([n.log_alpha], lr=cfg.lr)
        self.opt_wm = torch.optim.Adam(list(n.wm.parameters()) + list(n.dec.parameters()), lr=cfg.lr)
        self.eye = torch.eye(self.types.numel(), dtype=torch.bool, device=device)

    @property
    def alpha(self) -> torch.Tensor:
        return self.nets.log_alpha.exp()

    def _batch(self, buf: Buffer, n: int) -> dict[str, torch.Tensor]:
        env, row = buf.sample_rows(n)
        slot = buf._slot(row)
        now = self.belief_fn(self.nets, buf, env, row, use_next=False)
        with torch.no_grad():
            nxt = self.belief_fn(self.nets, buf, env, row, use_next=True)
        return {
            "b": now["b"], "e": now["e"], "b_next": nxt["b"], "e_next": nxt["e"],
            "a": buf.action[env, slot], "r": buf.reward[env, slot],
            "term": buf.terminated[env, slot], "target": buf.target[env, slot],
        }

    def update(self, offline: Buffer | None, online: Buffer) -> dict[str, float]:
        cfg, n = self.cfg, self.nets
        half = cfg.batch // 2
        parts = [self._batch(online, cfg.batch if offline is None else half)]
        if offline is not None:
            parts.append(self._batch(offline, half))
        d = {k: torch.cat([p[k] for p in parts], 0) for k in parts[0]}
        B, K = d["a"].shape[:2]

        # Critic. Target uses the minimum of two random heads, as in RLPD, and the entropy bonus.
        with torch.no_grad():
            a_next, logp_next = n.actor.sample(d["b_next"])
            q_t = n.critic_target(d["b_next"], a_next)
            pair = torch.randperm(cfg.num_critics, device=self.device)[:2]
            v_next = q_t[pair].min(0).values
            if cfg.backup_entropy:
                v_next = v_next - self.alpha * logp_next
            not_done = (~d["term"]).float().unsqueeze(-1)
            y = d["r"].unsqueeze(-1) + cfg.gamma * not_done * v_next  # [B, K]
        q = n.critic(d["b"], d["a"])  # [M, B, K]
        critic_loss = F.mse_loss(q, y.unsqueeze(0).expand_as(q))
        self.opt_critic.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.opt_critic.step()

        # Actor on detached beliefs, so the encoder follows the critic only.
        b = d["b"].detach()
        a_new, logp = n.actor.sample(b)
        q_new = n.critic(b, a_new).mean(0)
        actor_loss = (self.alpha.detach() * logp - q_new).mean()
        self.opt_actor.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.opt_actor.step()

        alpha_loss = -(n.log_alpha * (logp.detach() + cfg.target_entropy)).mean()
        self.opt_alpha.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.opt_alpha.step()

        # World model and decoder on stop gradient latents.
        e, e_next = d["e"].detach(), d["e_next"].detach()
        types = self.types.view(1, 1, K).expand(B, K, K)
        others = d["a"].unsqueeze(1).expand(B, K, K, ACT_DIM)
        ctx = n.wm.context(types, others, (~self.eye).expand(B, K, K))
        pred = n.wm(e, d["a"], ctx)
        wm_loss = F.mse_loss(pred, e_next)
        dec_loss = F.mse_loss(n.dec(e), d["target"])
        self.opt_wm.zero_grad(set_to_none=True)
        (wm_loss + dec_loss).backward()
        self.opt_wm.step()

        n.polyak(cfg.tau)
        return {
            "critic_loss": critic_loss.item(), "actor_loss": actor_loss.item(),
            "alpha": self.alpha.item(), "wm_loss": wm_loss.item(), "dec_loss": dec_loss.item(),
            "q_mean": q.mean().item(), "entropy": -logp.mean().item(),
        }

    def save(self, path: str, extra: dict | None = None) -> None:
        torch.save({"nets": self.nets.state_dict(), "cfg": asdict(self.cfg),
                    "types": self.types.cpu(), "extra": extra or {}}, path)

    @classmethod
    def load(cls, path: str, device: str) -> "Agent":
        d = torch.load(path, map_location="cpu")
        agent = cls(d["types"], RLPDConfig(**d["cfg"]), device)
        agent.nets.load_state_dict(d["nets"])
        return agent
