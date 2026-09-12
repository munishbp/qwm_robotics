"""Messages, forward correction, and fusion. docs/design.md sections 6.2 and 6.3.

Every robot i holds a table with one entry per teammate j: the absolute row of the last message
it received from j. `MessageTable` keeps that state for the live env loop. `beliefs` turns a
table read from a buffer into fused beliefs, and `roll` and `fuse_table` are the tensor level
pieces that the search reuses.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from swarm.buffer import AGE_MAX, Buffer
from swarm.nets import FEAT_DIM, NUM_TYPES, Nets


@dataclass
class BeliefConfig:
    lag: int = 1
    dropout: float = 0.0
    # Ablation: every teammate estimate is masked out of the fusion, so the belief is self only.
    mask_messages: bool = False
    # Ablation: the stale message is used as received, with no forward correction.
    roll_messages: bool = True


class MessageTable:
    """Per env, per receiver, per sender: the absolute row of the message held."""

    def __init__(self, num_envs: int, num_robots: int, device: str) -> None:
        self.stamp = torch.zeros(num_envs, num_robots, num_robots, dtype=torch.long, device=device)
        self.eye = torch.eye(num_robots, dtype=torch.bool, device=device)

    def update(
        self, row_now: int, ep_start: torch.Tensor, comm: torch.Tensor, cfg: BeliefConfig
    ) -> torch.Tensor:
        """Advance the table to `row_now` and return a copy of it.

        A message sent at `row_now - lag` arrives if the link `comm[i, j]` is up at `row_now` and
        it survives the dropout draw. A robot that hears nothing keeps its old entry. Every entry
        is at least `AGE_MAX` recent, which models a channel with a bounded delay. At an episode
        start every entry is the start row.
        """
        send = row_now - cfg.lag
        deliver = comm & (send >= ep_start).view(-1, 1, 1)
        if cfg.dropout > 0:
            deliver = deliver & (torch.rand_like(comm, dtype=torch.float32) >= cfg.dropout)
        stamp = torch.where(deliver, torch.full_like(self.stamp, send), self.stamp)
        stamp = torch.maximum(stamp, torch.full_like(stamp, row_now - AGE_MAX))
        stamp = torch.maximum(stamp, ep_start.view(-1, 1, 1))
        stamp = torch.where(self.eye, torch.full_like(stamp, row_now), stamp)
        fresh = (ep_start == row_now).view(-1, 1, 1)
        self.stamp = torch.where(fresh, torch.full_like(stamp, row_now), stamp)
        return self.stamp.clone()


def features(age: torch.Tensor, types: torch.Tensor) -> torch.Tensor:
    """Per estimate features `[..., FEAT_DIM]`: age fraction and type one hot.

    Uncertainty is not a fusion feature. The offline buffer has no critic, so a stored
    uncertainty would mark the data source inside every batch. Uncertainty serves the leader
    election only.
    """
    one_hot = F.one_hot(types.long(), NUM_TYPES).float()
    return torch.cat([(age.float() / AGE_MAX).unsqueeze(-1), one_hot], -1)


def fuse_table(nets: Nets, table: torch.Tensor, feat: torch.Tensor, mask_messages: bool = False) -> torch.Tensor:
    """Fuse every slot of a table as if it were the own encoding.

    table `[..., K, L]`, feat `[..., K, F]` -> beliefs `[..., K, L]`. Slot k is the query and the
    other K - 1 slots are its estimates. The own feature keeps age zero. With `mask_messages`
    every estimate is masked and the belief is self only.
    """
    k = table.shape[-2]
    eye = torch.eye(k, dtype=torch.bool, device=table.device)
    est = table.unsqueeze(-3).expand(*table.shape[:-2], k, k, table.shape[-1])
    est_feat = feat.unsqueeze(-3).expand(*feat.shape[:-2], k, k, feat.shape[-1])
    mask = (~eye).expand(*table.shape[:-2], k, k)
    if mask_messages:
        mask = torch.zeros_like(mask)
    own_feat = feat.clone()
    own_feat[..., 0] = 0.0
    return nets.fuse(table, own_feat, est, est_feat, mask)


def imagined_actions(nets: Nets, table: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
    """Mean policy action for every slot of a table, `[..., K, A]`."""
    return nets.actor.mean(fuse_table(nets, table, feat))


def self_only(nets: Nets, z: torch.Tensor, feat_self: torch.Tensor) -> torch.Tensor:
    """Belief of an estimate with no teammates, the input the policy sees during a roll."""
    empty = z.unsqueeze(-2)[..., :0, :]
    empty_feat = feat_self.unsqueeze(-2)[..., :0, :]
    empty_mask = torch.zeros(*z.shape[:-1], 0, dtype=torch.bool, device=z.device)
    return nets.fuse(z, feat_self, empty, empty_feat, empty_mask)


def roll(
    nets: Nets, z: torch.Tensor, a: torch.Tensor, ctx: torch.Tensor, steps: torch.Tensor,
    feat_self: torch.Tensor,
) -> torch.Tensor:
    """Roll each estimate forward by its own number of steps.

    z `[..., L]`, a `[..., A]` the recorded action at the stamp, ctx `[..., L]`, steps `[...]`
    integer up to AGE_MAX. Step one uses the recorded action. Every later step uses the mean
    action of the policy on the self only belief of the estimate, which is what the searching
    robot can compute. The context stays fixed because the receiver has no newer information.
    """
    for s in range(1, AGE_MAX + 1):
        active = (steps >= s).unsqueeze(-1)
        if not bool(active.any()):
            break
        z = torch.where(active, nets.wm(z, a, ctx), z)
        a = torch.where(active, nets.actor.mean(self_only(nets, z, feat_self)), a)
    return z


def table_context(nets: Nets, types: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Action context for every entry of a table from the other entries of the same table.

    types `[K]`, actions `[..., K, K, A]` (receiver i, sender j) -> ctx `[..., K, K, L]` where
    ctx[i, j] pools the recorded actions that i holds from every sender other than j.
    """
    k = types.numel()
    lead = actions.shape[:-3]
    eye = torch.eye(k, dtype=torch.bool, device=actions.device)
    pooled_types = types.view(*([1] * (len(lead) + 2)), k).expand(*lead, k, k, k)
    pooled_actions = actions.unsqueeze(-3).expand(*lead, k, k, k, actions.shape[-1])
    mask = (~eye).view(*([1] * (len(lead) + 1)), k, k).expand(*lead, k, k, k)
    return nets.wm.context(pooled_types, pooled_actions, mask)


@torch.no_grad()
def estimates(
    nets: Nets, buf: Buffer, env: torch.Tensor, row: torch.Tensor, use_next: bool,
    roll_messages: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rolled forward teammate estimates from a buffer row.

    Returns the table of estimates `[..., K, K, L]` with the diagonal left as the sender's
    stamped encoding (the caller replaces it with the fresh own encoding), the features
    `[..., K, K, F]`, the recorded actions `[..., K, K, A]`, and the stored uncertainties
    `[..., K, K]`. An entry stamped at the current row is not written yet when the robots act,
    so its action and uncertainty are zero. Its age is zero, so the action is never used.
    """
    k = buf.K
    slot = buf._slot(row)
    stamp = buf.next_stamp[env, slot] if use_next else buf.stamp[env, slot]
    row_now = (row + 1) if use_next else row
    tab = buf.table(env, stamp)
    sender_types = buf.types.view(*([1] * env.dim()), 1, k).expand(*stamp.shape)
    z = nets.enc(tab["obs"], sender_types)
    age = (row_now.view(*row.shape, 1, 1) - stamp).clamp(min=0, max=AGE_MAX)
    current = (age == 0).unsqueeze(-1)
    action = torch.where(current, torch.zeros_like(tab["action"]), tab["action"])
    unc = torch.where(current.squeeze(-1), torch.zeros_like(tab["unc"]), tab["unc"])
    feat = features(age, sender_types)
    ctx = table_context(nets, buf.types, action)
    feat_self = feat.clone()
    feat_self[..., 0] = 0.0
    if roll_messages:
        z = roll(nets, z, action, ctx, age, feat_self)
    return z, feat, action, unc


def beliefs(
    nets: Nets, buf: Buffer, env: torch.Tensor, row: torch.Tensor, use_next: bool,
    mask_messages: bool = False, roll_messages: bool = True,
) -> dict[str, torch.Tensor]:
    """Fused beliefs for every robot at a buffer row (or the row after it).

    The own encoding carries gradient. The teammate estimates do not, so the RL losses shape the
    encoder through the fresh observation only and never through the world model.
    """
    k = buf.K
    own_stack = buf.next_stack(env, row) if use_next else buf.stack(env, row)
    e = nets.enc(own_stack, buf.types.view(*([1] * env.dim()), k).expand(*env.shape, k))
    z, feat, _, unc = estimates(nets, buf, env, row, use_next, roll_messages)
    eye = torch.eye(k, dtype=torch.bool, device=e.device).view(*([1] * env.dim()), k, k, 1)
    table = torch.where(eye, e.unsqueeze(-2).expand_as(z), z)
    # Each receiver i fuses row i of the table. fuse_table treats every slot as own, so the
    # diagonal of its output is the belief of receiver i built from its own row.
    fused = fuse_table(nets, table, feat, mask_messages)  # [..., K(i), K(slot), L]
    b = fused.diagonal(dim1=-3, dim2=-2).transpose(-1, -2)
    return {"b": b, "e": e, "table": table, "feat": feat, "unc_table": unc}


def beliefs_full(
    nets: Nets, buf: Buffer, env: torch.Tensor, row: torch.Tensor, use_next: bool
) -> dict[str, torch.Tensor]:
    """Centralized baseline: the belief is the encoding of the full state stack. No messages."""
    k = buf.K
    stack = buf.next_stack(env, row, "full") if use_next else buf.stack(env, row, source="full")
    e = nets.enc(stack, buf.types.view(*([1] * env.dim()), k).expand(*env.shape, k))
    return {"b": e, "e": e}
