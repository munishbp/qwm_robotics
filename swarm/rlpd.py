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
    # Under a sparse reward the critic is flat in the action for a long time, so the actor has no
    # gradient to follow. A behavior cloning term on the offline half of every batch gives the
    # actor the scripted behavior while the critic learns the value the search needs.
    bc_weight: float = 1.0
    # The representation (encoders and fusion) is shaped by the critic loss and by two supervised
    # anchors, the cloning term and a decoder from the belief to the payload pose. The bootstrap
    # belief comes from target copies of the encoders and the fusion. Without the target copies
    # the belief collapsed to a constant between steps 3,500 and 5,000 in every run, and without
    # the critic gradient the critic could not fit the terminal rows.
    dec_b_weight: float = 1.0
    # The reward is one terminal unit, so every true value lies in [0, 1]. Clamping the target to
    # that range removes the offset that the minimum over noisy heads compounds through the
    # bootstrap (about -0.56 times the head spread divided by 1 - gamma).
    target_min: float = 0.0
    target_max: float = 1.0
    # The bootstrap uses the mean action. The sampled policy keeps a large standard deviation
    # because alpha decays to zero and a critic that is flat in the action never shrinks it, so
    # a sampled bootstrap values a noisy policy that fails, and that value decays by about 0.8
    # per step away from the goal. The search and the evaluation act with the mean policy.
    # The bootstrap uses the next action recorded in the buffer (a SARSA target on the behavior
    # data) and falls back to the policy mean where the next row is unavailable. The policy mean
    # lies about 0.45 from the data action under partial observability, and once the critic learns
    # action dependence that mean is an out of distribution query: the value chain collapsed to
    # zero at about step 5000 in every run that bootstrapped on the policy.
    target_policy: str = "data"  # data, mean, or sample
    # The target is the mean of two random heads, not the minimum. The heads disagree by 0.02 to
    # 0.03 near the goal, the minimum sits 0.56 of that below the mean on every bootstrap, and
    # over a 90 step horizon that compounds to a value of zero. The [0, 1] clamp bounds the
    # overestimation the minimum was there to prevent.
    target_reduce: str = "mean"  # mean or min
    # The critic target sums the rewards of `n_step` recorded rows and bootstraps after the last
    # one. It stays a SARSA target on the behavior data. 1 is the one step target. A longer window
    # attaches the terminal reward to the action in fewer backups. docs/results.md sections 3 to 20
    # used 1. Sections 22 and 23 show that 5 repairs the search on both simulators.
    n_step: int = 5
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
            list(n.critic.parameters()) + list(n.enc.parameters()) + list(n.fuse.parameters())
            + list(n.dec_b.parameters()), lr=cfg.lr
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
        # The return window. It ends at the last row of the episode, at the newest row of the
        # buffer, or after n_step rows, whichever comes first. `last` is its final row and `disc`
        # is the discount of the bootstrap that follows it.
        last, ret = row.clone(), buf.reward[env, slot].clone()
        disc = torch.full_like(ret, self.cfg.gamma)
        alive = torch.ones_like(row, dtype=torch.bool)
        for m in range(1, self.cfg.n_step):
            s_m = buf._slot((row + m).clamp(max=buf.t - 1))
            alive = alive & (row + m <= buf.t - 1) & (buf.ep_start[env, s_m] == buf.ep_start[env, slot])
            ret = ret + alive * disc * buf.reward[env, s_m]
            disc = torch.where(alive, disc * self.cfg.gamma, disc)
            last = torch.where(alive, row + m, last)
        # The world model target is one step after `row`. The critic bootstraps after `last`.
        with torch.no_grad():
            nxt = self.belief_fn(self.nets.target_view(), buf, env, row, use_next=True)
            boot = nxt if self.cfg.n_step == 1 else self.belief_fn(self.nets.target_view(), buf, env, last, use_next=True)
        # The next recorded action exists when the next row is in the buffer and in the same
        # episode. Otherwise the bootstrap uses the policy mean.
        nrow = (last + 1).clamp(max=buf.t - 1)
        has_next = (last + 1 <= buf.t - 1) & (buf.ep_start[env, buf._slot(nrow)] == buf.ep_start[env, slot])
        return {
            "b": now["b"], "e": now["e"], "b_next": boot["b"], "e_next": nxt["e"],
            "a": buf.action[env, slot], "r": ret, "disc": disc,
            "term": buf.terminated[env, buf._slot(last)], "target": buf.target[env, slot],
            "a_next": buf.action[env, buf._slot(nrow)], "has_next": has_next,
        }

    def update(self, offline: Buffer | None, online: Buffer) -> dict[str, float]:
        cfg, n = self.cfg, self.nets
        half = cfg.batch // 2
        parts = [self._batch(online, cfg.batch if offline is None else half)]
        if offline is not None:
            parts.append(self._batch(offline, half))
        d = {k: torch.cat([p[k] for p in parts], 0) for k in parts[0]}
        B, K = d["a"].shape[:2]
        n_online = parts[0]["a"].shape[0]

        # Critic. Target uses two random heads and the mean action.
        with torch.no_grad():
            a_next, logp_next = n.actor.sample(d["b_next"])
            if cfg.target_policy == "mean":
                a_next = n.actor.mean(d["b_next"])
            elif cfg.target_policy == "data":
                a_next = torch.where(d["has_next"].view(-1, 1, 1), d["a_next"], n.actor.mean(d["b_next"]))
            q_t = n.critic_target(d["b_next"], a_next)
            pair = torch.randperm(cfg.num_critics, device=self.device)[:2]
            v_next = q_t[pair].mean(0) if cfg.target_reduce == "mean" else q_t[pair].min(0).values
            if cfg.backup_entropy:
                v_next = v_next - self.alpha * logp_next
            not_done = (~d["term"]).float().unsqueeze(-1)
            y = d["r"].unsqueeze(-1) + d["disc"].unsqueeze(-1) * not_done * v_next  # [B, K]
            y = y.clamp(cfg.target_min, cfg.target_max)
        q = n.critic(d["b"], d["a"])  # [M, B, K]
        critic_loss = F.mse_loss(q, y.unsqueeze(0).expand_as(q))
        # The cloning term flows into the encoders and the fusion as well as the actor; it is what
        # teaches the fusion to read the payload position out of a teammate's message when the
        # own sensor cannot see it. The belief decoder anchors the belief to the payload pose.
        bc_loss = torch.zeros((), device=self.device)
        if offline is not None and cfg.bc_weight > 0:
            mu = n.actor.mean(d["b"][n_online:])
            bc_loss = F.mse_loss(mu, d["a"][n_online:])
        dec_b_loss = F.mse_loss(n.dec_b(d["b"]), d["target"])
        self.opt_critic.zero_grad(set_to_none=True)
        self.opt_actor.zero_grad(set_to_none=True)
        (critic_loss + cfg.bc_weight * bc_loss + cfg.dec_b_weight * dec_b_loss).backward()
        self.opt_critic.step()
        self.opt_actor.step()

        # SAC actor term on detached beliefs, so the RL objective shapes the actor only.
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
            "q_mean": q.mean().item(), "entropy": -logp.mean().item(), "bc_loss": bc_loss.item(),
            "dec_b_loss": dec_b_loss.item(),
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
