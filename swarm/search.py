"""Test time tree search with imagined teammates. docs/design.md section 7.

A searching robot holds one estimate per team slot. At the root it samples candidate own actions
and imagines every teammate's action from its estimate of that teammate's belief. Every path is
rolled one step through the world model with its joint action and then scored with the critic at
the policy mean of the rolled state. A beam keeps the best paths and expands them. Teammate
estimates roll forward together with the own estimate, so their staleness never changes with
depth.

Score of a path after D rolls: Q_0(b, a_0) + sum_{d=1}^{D} beta^d Q(b^(d), mu(b^(d))).
Every candidate is rolled before it is scored, so the number of world model calls grows with the
number of candidates. That is what makes the leader mode, which samples K times more candidates
in one search, cost the same as K independent searches.
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
    mode: str = "independent"  # independent, leader, leader_fresh, round_robin
    # Controls. "q" is the critic. "decoded" scores a belief by the decoded payload pose against
    # the goal, a hand built value that proves the search machinery without the critic. "random"
    # replaces every score by noise, so the search picks a random candidate.
    scorer: str = "q"
    # Gate. r is the span of Q over the root candidates divided by the ensemble spread at the mean
    # action (docs/next_steps.md, "The quantity that unites the two runs"). A row with r below
    # `gate` does not search and acts with `gate_fallback`: the mean action or one policy sample.
    # 0 disables the gate. `gate_shuffle` is the control: it permutes r across rows, which keeps
    # the searched fraction and destroys the selection.
    gate: float = 0.0
    gate_fallback: str = "sample"  # sample, mean
    gate_shuffle: bool = False
    # Pessimism. The critic score is the ensemble mean minus `lcb` ensemble standard deviations.
    # The argmax over noisy heads selects the largest head error, and the bound removes it.
    lcb: float = 0.0


def _roll_joint(nets: Nets, z: torch.Tensor, joint: torch.Tensor, types: torch.Tensor) -> torch.Tensor:
    """Roll every slot of z `[..., K, L]` one step with the joint action `[..., K, A]`."""
    k = z.shape[-2]
    lead = z.shape[:-2]
    eye = torch.eye(k, dtype=torch.bool, device=z.device)
    pooled_types = types.view(*([1] * (len(lead) + 1)), k).expand(*lead, k, k)
    pooled = joint.unsqueeze(-3).expand(*lead, k, k, ACT_DIM)
    ctx = nets.wm.context(pooled_types, pooled, (~eye).expand(*lead, k, k))
    return nets.wm(z, joint, ctx)


def _q(nets: Nets, b: torch.Tensor, a: torch.Tensor, cfg: SearchConfig | None = None,
       goal: torch.Tensor | None = None) -> torch.Tensor:
    """Node score. The critic by default; see SearchConfig.scorer for the controls."""
    if cfg is None or cfg.scorer == "q":
        q = nets.critic(b, a)
        return q.mean(0) if cfg is None or cfg.lcb == 0 else q.mean(0) - cfg.lcb * q.std(0)
    if cfg.scorer == "random":
        return torch.rand(b.shape[:-1], device=b.device)
    if cfg.scorer == "decoded":
        # dec_b(b) is (dx / A, dy / A, cos, sin, latched, visible) of the payload relative to the
        # robot. goal is (gx / A, gy / A, cos, sin) relative to the robot at the root. The score is
        # the negative pose error, in arena units plus a fraction of the angle error.
        d = nets.dec_b(b)
        pos = (d[..., :2] - goal[..., :2]).norm(dim=-1)
        ang = (d[..., 2:4] - goal[..., 2:4]).norm(dim=-1)
        return -(pos + 0.25 * ang)
    raise ValueError(f"unknown scorer {cfg.scorer}")


def _candidates(nets: Nets, b: torch.Tensor, own: torch.Tensor, n: int, joint_candidates: bool) -> torch.Tensor:
    """Joint actions `[R, ..., n + 1, K, A]` for beliefs `[R, ..., K, L]`: the mean joint action first,
    then n candidates. Independent mode samples the own slot only and keeps the teammates at their
    mean action. Joint mode samples every slot."""
    R = b.shape[0]
    rows = torch.arange(R, device=b.device)
    mean_joint = nets.actor.mean(b)  # [R, ..., K, A]
    if joint_candidates:
        cand = nets.actor.sample_n(b, n).transpose(-3, -2)  # [R, ..., n, K, A]
    else:
        b_own = b[rows, ..., own, :] if b.dim() == 3 else b[rows, :, own]
        own_cand = nets.actor.sample_n(b_own, n)  # [R, ..., n, A]
        cand = mean_joint.unsqueeze(-3).repeat(*([1] * (b.dim() - 2)), n, 1, 1)
        if b.dim() == 3:
            cand[rows, :, own] = own_cand
        else:
            cand[rows, :, :, own] = own_cand
    return torch.cat([mean_joint.unsqueeze(-3), cand], dim=-3)


def _gate(nets: Nets, b_own: torch.Tensor, a_mean: torch.Tensor, score: torch.Tensor, cfg: SearchConfig) -> torch.Tensor:
    """Rows `[R]` that search: r at or above the gate. `score` `[R, P]` is Q at the root candidates."""
    if cfg.scorer != "q":
        raise ValueError(f"the gate reads the critic, so it needs scorer q, got {cfg.scorer}")
    if cfg.gate_fallback not in ("sample", "mean"):
        raise ValueError(f"gate_fallback must be sample or mean, got {cfg.gate_fallback}")
    if cfg.gate_fallback == "sample" and score.shape[1] < 2:
        raise ValueError("gate_fallback sample needs at least one sampled candidate, got candidates=0")
    spread = nets.critic(b_own, a_mean).std(0).clamp_min(1e-8)
    r = (score.max(1).values - score.min(1).values) / spread
    if cfg.gate_shuffle:
        r = r[torch.randperm(r.shape[0], device=r.device)]
    return r >= cfg.gate


CHUNK = 512


@torch.no_grad()
def search_rows(
    nets: Nets, z: torch.Tensor, feat: torch.Tensor, own: torch.Tensor, types: torch.Tensor,
    cfg: SearchConfig, joint_candidates: bool, goal: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`_search_rows` over chunks of rows, which bounds peak memory.

    Memory per row grows with the square of the team size, so the chunk shrinks with it.
    """
    k = z.shape[1]
    chunk = max(32, int(CHUNK * (6 / k) ** 2))
    outs = [_search_rows(nets, z[i:i + chunk], feat[i:i + chunk], own[i:i + chunk], types, cfg, joint_candidates,
                         None if goal is None else goal[i:i + chunk])
            for i in range(0, z.shape[0], chunk)]
    return tuple(torch.cat([o[i] for o in outs]) for i in range(3))


@torch.no_grad()
def _search_rows(
    nets: Nets, z: torch.Tensor, feat: torch.Tensor, own: torch.Tensor, types: torch.Tensor,
    cfg: SearchConfig, joint_candidates: bool, goal: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the search for R independent rows.

    z `[R, K, L]` estimates, feat `[R, K, F]`, own `[R]` the slot of the searching robot. With
    `joint_candidates` the search samples every slot (the leader mode), otherwise only the own
    slot and imagines the rest. Returns the chosen joint action `[R, K, A]`, its normalized
    score `[R]`, and the rows that searched `[R]`. With a gate, a row below it returns its fallback
    candidate and leaves the search before the first world model call.
    """
    R, K, L = z.shape
    rows = torch.arange(R, device=z.device)
    n = cfg.candidates * (K if joint_candidates else 1)
    D = cfg.depth if cfg.beta > 0 else 0
    scale = sum(cfg.beta**d for d in range(D + 1))

    b = fuse_table(nets, z, feat)  # [R, K, L]
    joint = _candidates(nets, b, own, n, joint_candidates)  # [R, P, K, A]
    P = joint.shape[1]
    g = None if goal is None else goal.unsqueeze(1).expand(R, P, goal.shape[-1])
    score = _q(nets, b[rows, own].unsqueeze(1).expand(R, P, L), joint[rows, :, own], cfg, g)  # [R, P]
    went = torch.ones(R, dtype=torch.bool, device=z.device)
    if cfg.gate > 0:
        went = _gate(nets, b[rows, own], joint[rows, 0, own], score, cfg)
        # Candidate 0 is the mean joint action and candidate 1 is the first policy sample.
        fb = 0 if cfg.gate_fallback == "mean" else 1
        out_joint, out_score = joint[:, fb].clone(), score[:, fb] / scale
        if not went.any():
            return out_joint, out_score, went
        z, feat, own, joint, score = z[went], feat[went], own[went], joint[went], score[went]
        goal = None if goal is None else goal[went]
        R = z.shape[0]
        rows = torch.arange(R, device=z.device)
    root_joint = joint
    root_idx = torch.arange(P, device=z.device).unsqueeze(0).expand(R, P)
    zp = z.unsqueeze(1).expand(R, P, K, L)
    featp = feat.unsqueeze(1).expand(R, P, K, feat.shape[-1])

    for d in range(1, D + 1):
        zp = _roll_joint(nets, zp, joint, types)
        bp = fuse_table(nets, zp, featp)  # [R, P, K, L]
        mean_joint = nets.actor.mean(bp)
        gp = None if goal is None else goal.unsqueeze(1).expand(R, P, goal.shape[-1])
        score = score + cfg.beta**d * _q(nets, bp[rows, :, own], mean_joint[rows, :, own], cfg, gp)
        keep = score.topk(min(cfg.beam, P), dim=1).indices  # [R, J]
        gather = lambda t: t.gather(1, keep.view(R, -1, *([1] * (t.dim() - 2))).expand(R, keep.shape[1], *t.shape[2:]))
        score, root_idx, zp, featp, bp = map(gather, (score, root_idx, zp, featp, bp))
        if d == D:
            break
        # Expand every survivor: the mean joint action plus n sampled candidates.
        J = zp.shape[1]
        child = _candidates(nets, bp, own, n, joint_candidates)  # [R, J, n + 1, K, A]
        C = child.shape[2]
        P = J * C
        joint = child.reshape(R, P, K, ACT_DIM)
        score = score.unsqueeze(2).expand(R, J, C).reshape(R, P)
        root_idx = root_idx.unsqueeze(2).expand(R, J, C).reshape(R, P)
        zp = zp.unsqueeze(2).expand(R, J, C, K, L).reshape(R, P, K, L)
        featp = featp.unsqueeze(2).expand(R, J, C, K, feat.shape[-1]).reshape(R, P, K, feat.shape[-1])

    best = score.argmax(1)
    best_joint, best_score = root_joint[rows, root_idx[rows, best]], score[rows, best] / scale
    if cfg.gate <= 0:
        return best_joint, best_score, went
    out_joint[went], out_score[went] = best_joint, best_score
    return out_joint, out_score, went


@torch.no_grad()
def search(
    nets: Nets, table: torch.Tensor, feat: torch.Tensor, types: torch.Tensor, unc_table: torch.Tensor,
    cfg: SearchConfig, step: int, goal: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Actions `[E, K, A]` for the team from each robot's table `[E, K, K, L]`, plus statistics.

    `unc_table[e, i, j]` is the uncertainty of robot j as robot i knows it: fresh for j == i and as
    stale as the message otherwise. In leader mode every robot elects the lowest entry of its own
    row. A robot that elects itself runs a joint search and broadcasts. A robot that elected
    someone else executes its slot of that leader's broadcast, or its mean action when the elected
    robot did not elect itself. The disagreement rate is the share of robots whose choice differs
    from the election under fresh uncertainty.
    """
    E, K, _, L = table.shape
    dev = table.device
    stats: dict[str, float] = {}
    if cfg.mode == "independent":
        z = table.reshape(E * K, K, L)
        f = feat.reshape(E * K, K, feat.shape[-1])
        own = torch.arange(K, device=dev).repeat(E)
        g = None if goal is None else goal.reshape(E * K, -1)
        joint, _, went = search_rows(nets, z, f, own, types, cfg, joint_candidates=False, goal=g)
        if cfg.gate > 0:
            stats["searched_fraction"] = went.float().mean().item()
        rows = torch.arange(E * K, device=dev)
        return joint[rows, own].reshape(E, K, ACT_DIM), stats
    envs = torch.arange(E, device=dev)
    if cfg.mode == "round_robin":
        elected = torch.full((E, K), step % K, dtype=torch.long, device=dev)
    elif cfg.mode == "leader_fresh":
        # Control: one election per env from fresh uncertainty, every robot follows it, no
        # fallback. This removes the disagreement confound of the leader mode.
        fresh = unc_table.diagonal(dim1=-2, dim2=-1).argmin(-1)
        elected = fresh.unsqueeze(1).expand(E, K)
        stats["leader_disagreement"] = 0.0
    elif cfg.mode == "leader":
        elected = unc_table.argmin(-1)  # [E, K(i)] the leader that robot i elects
        fresh = unc_table.diagonal(dim1=-2, dim2=-1).argmin(-1)  # [E] election under fresh values
        stats["leader_disagreement"] = (elected != fresh.unsqueeze(1)).float().mean().item()
    else:
        raise ValueError(f"unknown search mode {cfg.mode}")
    self_elected = elected == torch.arange(K, device=dev).unsqueeze(0)  # [E, K]
    stats["searches_per_env"] = self_elected.float().sum(1).mean().item()
    e_idx, i_idx = self_elected.nonzero(as_tuple=True)
    # Default: every robot acts with its mean action, then broadcasts overwrite the slots.
    b_all = fuse_table(nets, table.reshape(E * K, K, L), feat.reshape(E * K, K, feat.shape[-1]))
    b_all = b_all.reshape(E, K, K, L)
    actions = nets.actor.mean(b_all.diagonal(dim1=1, dim2=2).transpose(1, 2))  # [E, K, A]
    if e_idx.numel() == 0:
        return actions, stats
    g = None if goal is None else goal[e_idx, i_idx]
    joint, _, went = search_rows(nets, table[e_idx, i_idx], feat[e_idx, i_idx], i_idx, types, cfg, joint_candidates=True, goal=g)
    if cfg.gate > 0:
        stats["searched_fraction"] = went.float().mean().item()
    broadcast = torch.zeros(E, K, K, ACT_DIM, device=dev)  # [e, leader, slot]
    broadcast[e_idx, i_idx] = joint
    leader_ok = self_elected[envs.unsqueeze(1), elected]  # [E, K] did my elected leader search
    chosen = broadcast[envs.unsqueeze(1), elected, torch.arange(K, device=dev).unsqueeze(0)]  # [E, K, A]
    actions = torch.where(leader_ok.unsqueeze(-1), chosen, actions)
    stats["robots_following_a_leader"] = leader_ok.float().mean().item()
    return actions, stats
