"""Test time tree search with imagined teammates. docs/design.md section 7.

A searching robot holds one estimate per team slot. It samples candidate own actions, imagines
every teammate's action from its estimate of that teammate's belief, rolls every estimate one
step through the world model with the imagined joint action, and scores the child with the
critic. A beam keeps the best paths. Teammate estimates roll forward one step per depth together
with the own estimate, so their staleness never changes with depth.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from swarm.belief import fuse_table
from swarm.nets import ACT_DIM, Nets


@dataclass
class SearchConfig:
    depth: int = 2
    candidates: int = 8
    beam: int = 4
    beta: float = 0.5
    mode: str = "independent"  # independent, leader, round_robin


def _roll_joint(nets: Nets, z: torch.Tensor, joint: torch.Tensor, types: torch.Tensor) -> torch.Tensor:
    """Roll every slot of z `[..., K, L]` one step with the joint action `[..., K, A]`."""
    k = z.shape[-2]
    lead = z.shape[:-2]
    eye = torch.eye(k, dtype=torch.bool, device=z.device)
    pooled_types = types.view(*([1] * (len(lead) + 1)), k).expand(*lead, k, k)
    pooled = joint.unsqueeze(-3).expand(*lead, k, k, ACT_DIM)
    ctx = nets.wm.context(pooled_types, pooled, (~eye).expand(*lead, k, k))
    return nets.wm(z, joint, ctx)


def _q(nets: Nets, b: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    return nets.critic(b, a).mean(0)


@torch.no_grad()
def search_rows(
    nets: Nets, z: torch.Tensor, feat: torch.Tensor, own: torch.Tensor, types: torch.Tensor,
    cfg: SearchConfig, joint_candidates: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the search for R independent rows.

    z `[R, K, L]` estimates, feat `[R, K, F]`, own `[R]` the slot of the searching robot. With
    `joint_candidates` the search samples every slot (the leader mode), otherwise only the own
    slot and imagines the rest. Returns the chosen joint action `[R, K, A]` and the root score.
    """
    R, K, L = z.shape
    rows = torch.arange(R, device=z.device)
    n_cand = cfg.candidates * (K if joint_candidates else 1)
    beam = cfg.beam
    D = cfg.depth
    scale = sum(cfg.beta**d for d in range(D + 1))

    # Depth 0: the root candidates. The mean action is always one of them, so the search can
    # never score below the plain policy under its own critic.
    b = fuse_table(nets, z, feat)  # [R, K, L]
    mean_joint = nets.actor.mean(b)  # [R, K, A]
    if joint_candidates:
        cand = nets.actor.sample_n(b, n_cand).transpose(-3, -2)  # [R, n, K, A]
    else:
        own_cand = nets.actor.sample_n(b[rows, own], n_cand)  # [R, n, A]
        cand = mean_joint.unsqueeze(1).repeat(1, n_cand, 1, 1)
        cand[rows, :, own] = own_cand
    joint = torch.cat([mean_joint.unsqueeze(1), cand], 1)  # [R, P, K, A]
    P = joint.shape[1]
    b_own = b[rows, own].unsqueeze(1).expand(R, P, L)
    score = _q(nets, b_own, joint[rows, :, own])  # [R, P]
    root_joint = joint
    root_idx = torch.arange(P, device=z.device).unsqueeze(0).expand(R, P)
    zp = z.unsqueeze(1).expand(R, P, K, L)
    featp = feat.unsqueeze(1).expand(R, P, K, feat.shape[-1])

    for d in range(1, D + 1):
        # Prune, then roll the survivors one step with their chosen joint action.
        keep = score.topk(min(beam, score.shape[1]), dim=1).indices  # [R, J]
        gather = lambda t: t.gather(1, keep.view(R, -1, *([1] * (t.dim() - 2))).expand(R, keep.shape[1], *t.shape[2:]))
        score, root_idx, joint, zp, featp = map(gather, (score, root_idx, joint, zp, featp))
        zp = _roll_joint(nets, zp, joint, types)
        bp = fuse_table(nets, zp, featp)  # [R, J, K, L]
        J = zp.shape[1]
        mean_joint = nets.actor.mean(bp)
        b_own = bp[rows, :, own]  # [R, J, L]
        if d == D:
            score = score + cfg.beta**d * _q(nets, b_own, mean_joint[rows, :, own])
            break
        # Expand: every survivor gets n_cand sampled children plus its mean child.
        if joint_candidates:
            cand = nets.actor.sample_n(bp, n_cand).transpose(-3, -2)  # [R, J, n, K, A]
        else:
            own_cand = nets.actor.sample_n(b_own, n_cand)  # [R, J, n, A]
            cand = mean_joint.unsqueeze(2).repeat(1, 1, n_cand, 1, 1)
            cand[rows, :, :, own] = own_cand
        child = torch.cat([mean_joint.unsqueeze(2), cand], 2)  # [R, J, n+1, K, A]
        C = child.shape[2]
        q_child = _q(nets, b_own.unsqueeze(2).expand(R, J, C, L), child[rows, :, :, own])
        score = (score.unsqueeze(2) + cfg.beta**d * q_child).reshape(R, J * C)
        root_idx = root_idx.unsqueeze(2).expand(R, J, C).reshape(R, J * C)
        joint = child.reshape(R, J * C, K, ACT_DIM)
        zp = zp.unsqueeze(2).expand(R, J, C, K, L).reshape(R, J * C, K, L)
        featp = featp.unsqueeze(2).expand(R, J, C, K, feat.shape[-1]).reshape(R, J * C, K, feat.shape[-1])

    best = score.argmax(1)
    chosen_root = root_idx[rows, best]
    return root_joint[rows, chosen_root], score[rows, best] / scale


@torch.no_grad()
def search(
    nets: Nets, table: torch.Tensor, feat: torch.Tensor, types: torch.Tensor, unc: torch.Tensor,
    cfg: SearchConfig, step: int,
) -> torch.Tensor:
    """Actions `[E, K, A]` for the whole team from each robot's table `[E, K, K, L]`."""
    E, K, _, L = table.shape
    if cfg.mode == "independent":
        z = table.reshape(E * K, K, L)
        f = feat.reshape(E * K, K, feat.shape[-1])
        own = torch.arange(K, device=table.device).repeat(E)
        joint, _ = search_rows(nets, z, f, own, types, cfg, joint_candidates=False)
        rows = torch.arange(E * K, device=table.device)
        return joint[rows, own].reshape(E, K, ACT_DIM)
    if cfg.mode == "leader":
        leader = unc.argmin(1)
    elif cfg.mode == "round_robin":
        leader = torch.full((E,), step % K, dtype=torch.long, device=table.device)
    else:
        raise ValueError(f"unknown search mode {cfg.mode}")
    envs = torch.arange(E, device=table.device)
    joint, _ = search_rows(nets, table[envs, leader], feat[envs, leader], leader, types, cfg, joint_candidates=True)
    return joint
