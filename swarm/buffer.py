"""Replay buffer with per env timelines.

Rows are indexed by (env, absolute step). A message table entry is an absolute row index, so a
training batch can read the encoding that a robot held from a teammate at any earlier step. See
docs/design.md section 5.

The live acting loop uses the same class as a history: `begin` writes the observation and the
message table of the current step before the robots act, and `finish` writes the rest after the
env step. Sampling skips the rows that a stack or a message table could reach past the ring.
"""

from __future__ import annotations

import torch

from swarm.nets import ACT_DIM, LOCAL_DIM, STACK

AGE_MAX = 8
GUARD = AGE_MAX + STACK
FIELDS_BEGIN = ("local", "ep_start", "stamp")
FIELDS_FINISH = (
    "action", "unc", "reward", "terminated", "truncated", "next_local", "next_stamp", "target",
)


class Buffer:
    def __init__(
        self, num_envs: int, capacity: int, types: torch.Tensor, device: str, full_dim: int = 0
    ) -> None:
        self.E, self.C, self.K = num_envs, capacity, int(types.numel())
        self.full_dim = full_dim
        self.device = device
        e, c, k = self.E, self.C, self.K
        f = torch.float32
        self.types = types.to(device).long()
        self.local = torch.zeros(e, c, k, LOCAL_DIM, device=device, dtype=f)
        self.next_local = torch.zeros(e, c, k, LOCAL_DIM, device=device, dtype=f)
        self.action = torch.zeros(e, c, k, ACT_DIM, device=device, dtype=f)
        self.unc = torch.zeros(e, c, k, device=device, dtype=f)
        self.reward = torch.zeros(e, c, device=device, dtype=f)
        self.terminated = torch.zeros(e, c, device=device, dtype=torch.bool)
        self.truncated = torch.zeros(e, c, device=device, dtype=torch.bool)
        self.ep_start = torch.zeros(e, c, device=device, dtype=torch.long)
        self.stamp = torch.zeros(e, c, k, k, device=device, dtype=torch.long)
        self.next_stamp = torch.zeros(e, c, k, k, device=device, dtype=torch.long)
        self.target = torch.zeros(e, c, k, 6, device=device, dtype=f)
        # The centralized baseline stores the full observation as well. Belief runs do not.
        self.full = torch.zeros(e, c, k, full_dim, device=device, dtype=f)
        self.next_full = torch.zeros(e, c, k, full_dim, device=device, dtype=f)
        # `t` is the absolute index of the row that `begin` writes next.
        self.t = 0

    @property
    def size(self) -> int:
        return min(self.t, self.C) * self.E

    def begin(
        self, local: torch.Tensor, ep_start: torch.Tensor, stamp: torch.Tensor,
        full: torch.Tensor | None = None,
    ) -> None:
        i = self.t % self.C
        self.local[:, i] = local
        if self.full_dim:
            self.full[:, i] = full
        self.ep_start[:, i] = ep_start
        self.stamp[:, i] = stamp

    def finish(self, **row: torch.Tensor) -> None:
        i = self.t % self.C
        for name in FIELDS_FINISH:
            getattr(self, name)[:, i] = row[name]
        if self.full_dim:
            self.next_full[:, i] = row["next_full"]
        self.t += 1

    def _slot(self, abs_row: torch.Tensor) -> torch.Tensor:
        return abs_row % self.C

    def sample_rows(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Uniform (env, absolute row) pairs whose stack and message gathers all land in the ring."""
        oldest = max(0, self.t - self.C) + GUARD
        newest = self.t - 1
        if newest < oldest:
            raise ValueError(f"buffer has {self.t} rows and needs more than {oldest}")
        env = torch.randint(0, self.E, (n,), device=self.device)
        row = torch.randint(oldest, newest + 1, (n,), device=self.device)
        return env, row

    def stack(
        self, env: torch.Tensor, row: torch.Tensor, robot: torch.Tensor | None = None,
        source: str = "local",
    ) -> torch.Tensor:
        """Encoder input at (env, row): the last STACK observations and the previous action.

        Rows before the episode start repeat the first observation of the episode. The previous
        action is zero at the episode start. With `robot` given, the result is for that robot
        only, `[..., ENC_IN]`. Otherwise it is `[..., K, ENC_IN]`. `source` selects the local
        or the full observation.
        """
        obs = getattr(self, source)
        start = self.ep_start[env, self._slot(row)]
        frames = []
        for s in range(STACK):
            r = self._slot(torch.maximum(row - s, start))
            frames.append(obs[env, r] if robot is None else obs[env, r, robot])
        prev_row = self._slot((row - 1).clamp(min=0))
        prev = self.action[env, prev_row] if robot is None else self.action[env, prev_row, robot]
        has_prev = (row > start).unsqueeze(-1)
        if robot is None:
            has_prev = has_prev.unsqueeze(-1)
        prev = torch.where(has_prev, prev, torch.zeros_like(prev))
        return torch.cat(frames + [prev], dim=-1)

    def next_stack(self, env: torch.Tensor, row: torch.Tensor, source: str = "local") -> torch.Tensor:
        """Encoder input one step after (env, row), from next_local, valid at a terminal row."""
        obs, nxt = getattr(self, source), getattr(self, "next_" + source)
        start = self.ep_start[env, self._slot(row)]
        frames = [nxt[env, self._slot(row)]]
        for s in range(STACK - 1):
            r = torch.maximum(row - s, start)
            frames.append(obs[env, self._slot(r)])
        prev = self.action[env, self._slot(row)]
        return torch.cat(frames + [prev], dim=-1)

    def table(self, env: torch.Tensor, stamp: torch.Tensor) -> dict[str, torch.Tensor]:
        """Message contents for a stamp table `[..., K, K]` with receiver i and sender j.

        Returns the sender's encoder input `[..., K, K, ENC_IN]`, action `[..., K, K, A]`, and
        uncertainty `[..., K, K]` at the stamped row.
        """
        k = self.K
        env_b = env.reshape(*env.shape, 1, 1).expand(*stamp.shape)
        sender = torch.arange(k, device=self.device).view(*([1] * env.dim()), 1, k).expand(*stamp.shape)
        obs = self.stack(env_b, stamp, sender)
        slot = self._slot(stamp)
        return {
            "obs": obs,
            "action": self.action[env_b, slot, sender],
            "unc": self.unc[env_b, slot, sender],
        }

    def state_dict(self) -> dict:
        if self.t > self.C:
            raise ValueError("saving a wrapped ring is not supported; size the buffer to the run")
        n = self.t
        out = {"t": n, "E": self.E, "types": self.types.cpu(), "full_dim": self.full_dim}
        for name in FIELDS_BEGIN + FIELDS_FINISH + ("full", "next_full"):
            out[name] = getattr(self, name)[:, :n].cpu()
        return out

    @classmethod
    def load(cls, path: str, device: str) -> "Buffer":
        d = torch.load(path, map_location="cpu")
        buf = cls(d["E"], d["t"], d["types"], device, d.get("full_dim", 0))
        for name in FIELDS_BEGIN + FIELDS_FINISH + ("full", "next_full"):
            getattr(buf, name).copy_(d[name].to(device))
        buf.t = d["t"]
        return buf
