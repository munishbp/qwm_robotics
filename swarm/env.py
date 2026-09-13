"""Batched 2D cooperative transport environment.

The world, the actions, the payload dynamics, and the observation layout follow docs/design.md
section 3. Every state tensor carries a leading env dimension E and lives on one device. The step
function runs one batched update with no python loop over envs or robots.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor

TYPE_PUSHER = 0
TYPE_GRIPPER = 1
TYPE_SCOUT = 2
NUM_TYPES = 3

LOCAL_DIM = 16

DEFAULT_TEAM: tuple[int, ...] = (
    TYPE_PUSHER,
    TYPE_PUSHER,
    TYPE_PUSHER,
    TYPE_GRIPPER,
    TYPE_GRIPPER,
    TYPE_SCOUT,
)


def full_dim(num_robots: int) -> int:
    """Return the width of the centralized observation for a team of this size."""
    return LOCAL_DIM + 4 + 3 * num_robots


@dataclass(frozen=True)
class EnvConfig:
    """Every physical constant of the task. The defaults are the values in docs/design.md."""

    arena_half: float = 5.0
    payload_hx: float = 0.8
    payload_hy: float = 0.4
    robot_radius: float = 0.2
    dt: float = 0.1
    horizon: int = 150
    v_max: float = 1.5
    # Measured from the robot center. The disc edge is then within 0.25 m of the boundary.
    contact_dist: float = 0.45
    push_force: float = 1.0
    grip_force: float = 0.3
    friction_f0: float = 3.5
    friction_df: float = 1.0
    friction_fmin: float = 0.5
    torque_frac: float = 0.5
    drag_lin: float = 2.0
    drag_ang: float = 2.0
    pos_tol: float = 0.3
    ang_tol: float = 0.2
    comm_range: float = 6.0
    sense_range: tuple[float, float, float] = (3.0, 3.0, 15.0)
    goal_dist_min: float = 1.5
    goal_dist_max: float = 3.0
    goal_angle_max: float = math.pi / 4
    start_clear: float = 1.0
    spawn_margin: float = 1.0
    # "quasistatic": velocity follows the excess force at once (the original task).
    # "momentum": the payload carries velocity; force accelerates it and Coulomb friction with the
    # same thresholds decelerates it. Mass and inertia below apply to the momentum model only.
    dynamics: str = "quasistatic"
    payload_mass: float = 2.0


def wrap_angle(angle: Tensor) -> Tensor:
    """Return the angle folded into [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _sign_nz(x: Tensor) -> Tensor:
    """Return +1 or -1. Plain sign returns 0 at the origin and that breaks a face choice."""
    return x.ge(0).to(x.dtype) * 2.0 - 1.0


def to_local(points: Tensor, pose: Tensor) -> Tensor:
    """Transform world points [E, N, 2] into the payload frame given the pose [E, 3]."""
    cos = torch.cos(pose[:, 2]).unsqueeze(-1)
    sin = torch.sin(pose[:, 2]).unsqueeze(-1)
    rel = points - pose[:, None, :2]
    return torch.stack(
        (rel[..., 0] * cos + rel[..., 1] * sin, -rel[..., 0] * sin + rel[..., 1] * cos), dim=-1
    )


def to_world_vec(vecs: Tensor, pose: Tensor) -> Tensor:
    """Rotate vectors [E, N, 2] from the payload frame into the world frame."""
    cos = torch.cos(pose[:, 2]).unsqueeze(-1)
    sin = torch.sin(pose[:, 2]).unsqueeze(-1)
    return torch.stack(
        (vecs[..., 0] * cos - vecs[..., 1] * sin, vecs[..., 0] * sin + vecs[..., 1] * cos), dim=-1
    )


class Contact(NamedTuple):
    """The result of one closest point query against the payload rectangle."""

    sdist: Tensor
    point: Tensor
    normal: Tensor
    local: Tensor


def rect_contact(points: Tensor, pose: Tensor, hx: float, hy: float) -> Contact:
    """Return the signed distance, the closest boundary point, and the outward normal.

    The query runs in the payload frame because a clamp to the half extents gives the closest point
    there. A point inside the rectangle leaves along the axis of least penetration, which keeps the
    normal continuous for the projection in step. The signed distance is negative inside.
    """
    local = to_local(points, pose)
    lx, ly = local[..., 0], local[..., 1]
    qx, qy = lx.clamp(-hx, hx), ly.clamp(-hy, hy)
    dx, dy = lx - qx, ly - qy
    dist = torch.sqrt(dx * dx + dy * dy)

    inside = (lx.abs() <= hx) & (ly.abs() <= hy)
    pen_x, pen_y = hx - lx.abs(), hy - ly.abs()
    use_x = pen_x <= pen_y
    sx, sy = _sign_nz(lx), _sign_nz(ly)
    zero = torch.zeros_like(lx)

    safe = dist.clamp(min=1e-9)
    close_x = torch.where(inside, torch.where(use_x, sx * hx, lx), qx)
    close_y = torch.where(inside, torch.where(use_x, ly, sy * hy), qy)
    norm_x = torch.where(inside, torch.where(use_x, sx, zero), dx / safe)
    norm_y = torch.where(inside, torch.where(use_x, zero, sy), dy / safe)
    sdist = torch.where(inside, -torch.where(use_x, pen_x, pen_y), dist)

    close_local = torch.stack((close_x, close_y), dim=-1)
    normal_local = torch.stack((norm_x, norm_y), dim=-1)
    point = pose[:, None, :2] + to_world_vec(close_local, pose)
    return Contact(sdist, point, to_world_vec(normal_local, pose), close_local)


def segment_hits_box(p0: Tensor, p1: Tensor, hx: float, hy: float) -> Tensor:
    """Return true where the segment p0 to p1 crosses the axis aligned box.

    Both endpoints are already in the payload frame. The test is the slab test. It requires a
    crossing of positive length so that a gripper that sits on the boundary does not occlude
    itself.
    """
    delta = p1 - p0
    neg_inf = torch.full_like(delta[..., 0], -math.inf)
    pos_inf = torch.full_like(delta[..., 0], math.inf)
    lo, hi = neg_inf, pos_inf
    for axis, half in ((0, hx), (1, hy)):
        d_axis, o_axis = delta[..., axis], p0[..., axis]
        parallel = d_axis.abs() < 1e-9
        safe = torch.where(parallel, torch.ones_like(d_axis), d_axis)
        t1, t2 = (-half - o_axis) / safe, (half - o_axis) / safe
        near, far = torch.minimum(t1, t2), torch.maximum(t1, t2)
        within = o_axis.abs() <= half
        near = torch.where(parallel, torch.where(within, neg_inf, pos_inf), near)
        far = torch.where(parallel, torch.where(within, pos_inf, neg_inf), far)
        lo, hi = torch.maximum(lo, near), torch.minimum(hi, far)
    return (hi.clamp(max=1.0) - lo.clamp(min=0.0)) > 1e-4


class TransportEnv:
    """A batch of transport episodes. See docs/design.md section 3.6 for the API."""

    def __init__(
        self,
        num_envs: int,
        team: tuple[int, ...] = DEFAULT_TEAM,
        device: str | torch.device = "cuda",
        cfg: EnvConfig = EnvConfig(),
        seed: int = 0,
    ) -> None:
        if num_envs < 1:
            raise ValueError(f"num_envs must be at least 1, got {num_envs}.")
        if len(team) < 1:
            raise ValueError("team must hold at least one robot type.")
        for entry in team:
            if entry not in (TYPE_PUSHER, TYPE_GRIPPER, TYPE_SCOUT):
                raise ValueError(f"team holds the unknown robot type {entry}.")
        device = torch.device(device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("device 'cuda' is not available. Pass device='cpu' instead.")

        self.num_envs = num_envs
        self.team = tuple(team)
        self.num_robots = len(team)
        self.device = device
        self.cfg = cfg

        self.types = torch.tensor(self.team, dtype=torch.long, device=device)
        self.type_onehot = torch.nn.functional.one_hot(self.types, NUM_TYPES).float()
        self.sense_range = torch.tensor(cfg.sense_range, device=device)[self.types]
        self.is_pusher = self.types == TYPE_PUSHER
        self.is_gripper = self.types == TYPE_GRIPPER
        self.rank = _type_rank(self.types)

        # The payload stays fully inside the arena, so its center stops one circumradius early.
        self.payload_radius = math.hypot(cfg.payload_hx, cfg.payload_hy)
        self.center_limit = cfg.arena_half - self.payload_radius
        self.robot_limit = cfg.arena_half - cfg.robot_radius
        self.spawn_limit = cfg.arena_half - cfg.spawn_margin

        shape = (num_envs, self.num_robots)
        self.payload = torch.zeros(num_envs, 3, device=device)
        self.goal = torch.zeros(num_envs, 3, device=device)
        self.robot_pos = torch.zeros(*shape, 2, device=device)
        self.latched = torch.zeros(*shape, dtype=torch.bool, device=device)
        self.latch_local = torch.zeros(*shape, 2, device=device)
        self.step_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.payload_vel = torch.zeros(num_envs, 3, device=device)  # momentum model only

        self._gen = torch.Generator(device=device)
        self.seed(seed)
        self.reset()

    def seed(self, seed: int) -> None:
        """Reseed the sampler so a run replicates."""
        self._gen.manual_seed(int(seed))

    def _rand(self, *shape: int) -> Tensor:
        return torch.rand(*shape, generator=self._gen, device=self.device)

    def reset(self, env_ids: Tensor | None = None) -> dict[str, Tensor]:
        """Reset the named envs, or every env, and return the observation for every env."""
        if env_ids is None:
            mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            if env_ids.dtype not in (torch.long, torch.int32):
                raise ValueError(f"env_ids must hold integers, got dtype {env_ids.dtype}.")
            env_ids = env_ids.to(self.device, torch.long)
            if env_ids.numel() and (env_ids.min() < 0 or env_ids.max() >= self.num_envs):
                raise ValueError(f"env_ids must lie in [0, {self.num_envs}).")
            mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            mask[env_ids] = True
        self._reset_masked(mask)
        return self._observe()

    def _reset_masked(self, mask: Tensor) -> None:
        """Sample a fresh episode for every env and keep it where the mask is true.

        The sampler runs on the whole batch instead of on an index list. That keeps the auto reset
        in step free of a device to host copy, and the cost of the extra samples is small.
        """
        cfg = self.cfg
        limit = self.spawn_limit
        num = self.num_envs

        angle = (self._rand(num) * 2.0 - 1.0) * math.pi
        direction = self._rand(num) * 2.0 * math.pi
        span = cfg.goal_dist_max - cfg.goal_dist_min
        reach = cfg.goal_dist_min + self._rand(num) * span
        offset = torch.stack((torch.cos(direction), torch.sin(direction)), dim=-1) * reach[:, None]

        # The payload center is drawn from the band that keeps both it and the goal in the arena.
        low = torch.clamp(-limit - offset, min=-limit)
        high = torch.clamp(limit - offset, max=limit)
        center = low + self._rand(num, 2) * (high - low)

        payload = torch.cat((center, angle[:, None]), dim=-1)
        goal_angle = wrap_angle(angle + (self._rand(num) * 2.0 - 1.0) * cfg.goal_angle_max)
        goal = torch.cat((center + offset, goal_angle[:, None]), dim=-1)

        pos = (self._rand(num, self.num_robots, 2) * 2.0 - 1.0) * self.robot_limit
        for _ in range(8):
            clear = rect_contact(pos, payload, cfg.payload_hx, cfg.payload_hy).sdist
            retry = (clear < cfg.start_clear).unsqueeze(-1)
            fresh = (self._rand(num, self.num_robots, 2) * 2.0 - 1.0) * self.robot_limit
            pos = torch.where(retry, fresh, pos)

        # The loop leaves the last draw unmeasured. This step puts any robot that is still too
        # close onto a ring around the payload, so the clearance rule holds for every reset.
        clear = rect_contact(pos, payload, cfg.payload_hx, cfg.payload_hy).sdist
        offset = pos - payload[:, None, :2]
        span = torch.linalg.vector_norm(offset, dim=-1, keepdim=True).clamp(min=1e-6)
        ring = payload[:, None, :2] + offset / span * (self.payload_radius + cfg.start_clear)
        pos = torch.where((clear < cfg.start_clear).unsqueeze(-1), ring, pos)
        pos = pos.clamp(-self.robot_limit, self.robot_limit)

        keep = mask[:, None]
        self.payload = torch.where(keep, payload, self.payload)
        self.goal = torch.where(keep, goal, self.goal)
        self.robot_pos = torch.where(keep[:, :, None], pos, self.robot_pos)
        self.latched = self.latched & ~keep
        self.latch_local = torch.where(keep[:, :, None], torch.zeros_like(pos), self.latch_local)
        self.step_count = torch.where(mask, torch.zeros_like(self.step_count), self.step_count)
        self.payload_vel = torch.where(mask[:, None], torch.zeros_like(self.payload_vel), self.payload_vel)

    def step(
        self, actions: Tensor
    ) -> tuple[dict[str, Tensor], Tensor, Tensor, Tensor, dict[str, Tensor]]:
        """Advance every env one step and auto reset the episodes that ended.

        The caller passes finite actions [E, K, 3] in [-1, 1]. The env clips them. A wrong shape
        raises ValueError, because a silent broadcast hides the bug in the caller.

        info carries the last observation and the last pose error of the ended episode. For an env
        that did not end they equal the values of the returned observation.
        """
        expected = (self.num_envs, self.num_robots, 3)
        if actions.shape != expected:
            raise ValueError(f"actions must have shape {expected}, got {tuple(actions.shape)}.")
        cfg = self.cfg
        actions = actions.to(self.device, torch.float32).clamp(-1.0, 1.0)
        vel_cmd, control = actions[..., :2], actions[..., 2]

        contact = rect_contact(self.robot_pos, self.payload, cfg.payload_hx, cfg.payload_hy)
        near = contact.sdist <= cfg.contact_dist
        hold = control > 0.0

        # A gripper latches before the force sum so that its own latch lowers the threshold now.
        fresh_latch = self.is_gripper & hold & near & ~self.latched
        self.latch_local = torch.where(fresh_latch.unsqueeze(-1), contact.local, self.latch_local)
        self.latched = (self.latched | fresh_latch) & self.is_gripper & hold

        center = self.payload[:, None, :2]
        pushing = (self.is_pusher & hold & near).unsqueeze(-1)
        push_force = -contact.normal * (control * cfg.push_force).unsqueeze(-1) * pushing
        push_arm = contact.point - center

        latch_world = center + to_world_vec(self.latch_local, self.payload)
        grip_force = vel_cmd * cfg.grip_force * self.latched.unsqueeze(-1)
        grip_arm = latch_world - center

        force = (push_force + grip_force).sum(dim=1)
        torque = (_cross(push_arm, push_force) + _cross(grip_arm, grip_force)).sum(dim=1)

        n_latched = self.latched.sum(dim=1).to(force.dtype)
        slip_force = torch.clamp(
            cfg.friction_f0 - cfg.friction_df * n_latched, min=cfg.friction_fmin
        )
        slip_torque = cfg.torque_frac * slip_force * cfg.payload_hx

        if cfg.dynamics == "momentum":
            # Semi implicit Euler with Coulomb friction: the force accelerates the payload, then the
            # friction removes up to slip_force * dt / m of speed. A force below the threshold
            # cannot build speed, so the friction facts of the design hold.
            m = cfg.payload_mass
            inertia = m * (cfg.payload_hx**2 + cfg.payload_hy**2) / 3.0
            v = self.payload_vel[:, :2] + force / m * cfg.dt
            vmag = torch.linalg.vector_norm(v, dim=-1)
            vmag_after = (vmag - slip_force * cfg.dt / m).clamp(min=0.0)
            velocity = v / vmag.clamp(min=1e-9)[:, None] * vmag_after[:, None]
            w = self.payload_vel[:, 2] + torque / inertia * cfg.dt
            spin = (w.abs() - slip_torque * cfg.dt / inertia).clamp(min=0.0) * _sign_nz(w)
            self.payload_vel = torch.cat((velocity, spin[:, None]), dim=-1)
        else:
            magnitude = torch.linalg.vector_norm(force, dim=-1)
            speed = (magnitude - slip_force).clamp(min=0.0) / cfg.drag_lin
            velocity = force / magnitude.clamp(min=1e-9)[:, None] * speed[:, None]
            spin = (torque.abs() - slip_torque).clamp(min=0.0) / cfg.drag_ang * _sign_nz(torque)

        moved = self.payload[:, :2] + velocity * cfg.dt
        self.payload = torch.cat(
            (
                moved.clamp(-self.center_limit, self.center_limit),
                wrap_angle(self.payload[:, 2] + spin * cfg.dt)[:, None],
            ),
            dim=-1,
        )

        # The arena clamp runs before the projection, never after it. A clamp after the projection
        # pushes a robot back into a payload that stands against a wall. A robot inside the payload
        # breaks the contact model and the occlusion test at the same time, so the projection is
        # the last word. A robot squeezed between the payload and a wall can pass the robot limit
        # by up to one radius.
        pos = self.robot_pos + vel_cmd * cfg.v_max * cfg.dt
        pos = pos.clamp(-self.robot_limit, self.robot_limit)
        after = rect_contact(pos, self.payload, cfg.payload_hx, cfg.payload_hy)
        overlap = (after.sdist < cfg.robot_radius).unsqueeze(-1)
        pos = torch.where(overlap, after.point + after.normal * cfg.robot_radius, pos)
        latch_world = self.payload[:, None, :2] + to_world_vec(self.latch_local, self.payload)
        self.robot_pos = torch.where(self.latched.unsqueeze(-1), latch_world, pos)

        self.step_count = self.step_count + 1
        gap = torch.linalg.vector_norm(self.payload[:, :2] - self.goal[:, :2], dim=-1)
        angle_error = wrap_angle(self.payload[:, 2] - self.goal[:, 2]).abs()
        terminated = (gap < cfg.pos_tol) & (angle_error < cfg.ang_tol)
        truncated = (self.step_count >= cfg.horizon) & ~terminated
        reward = terminated.to(torch.float32)

        ended = self._observe()
        info = {
            "success": terminated,
            "final_local": ended["local"],
            "final_full": ended["full"],
            "final_pos_error": gap,
            "final_angle_error": angle_error,
        }
        self._reset_masked(terminated | truncated)
        return self._observe(), reward, terminated, truncated, info

    def _visible(self) -> Tensor:
        """Return the payload visibility flag [E, K] from the range of each robot type."""
        offset = self.payload[:, None, :2] - self.robot_pos
        return torch.linalg.vector_norm(offset, dim=-1) <= self.sense_range

    def _local_obs(self) -> Tensor:
        """Build the per robot observation [E, K, 16]. The layout is docs/design.md section 3.5."""
        cfg = self.cfg
        scale = cfg.arena_half
        visible = self._visible().to(torch.float32)
        goal_rel = (self.goal[:, None, :2] - self.robot_pos) / scale
        goal_dir = torch.stack((torch.cos(self.goal[:, 2]), torch.sin(self.goal[:, 2])), dim=-1)
        payload_rel = (self.payload[:, None, :2] - self.robot_pos) / scale
        payload_dir = torch.stack(
            (torch.cos(self.payload[:, 2]), torch.sin(self.payload[:, 2])), dim=-1
        )
        shape = (self.num_envs, self.num_robots, 2)
        payload_block = torch.cat(
            (payload_rel, payload_dir[:, None, :].expand(shape)), dim=-1
        ) * visible.unsqueeze(-1)
        time_frac = (self.step_count.to(torch.float32) / cfg.horizon)[:, None, None]
        return torch.cat(
            (
                self.robot_pos / scale,
                self.type_onehot.expand(self.num_envs, -1, -1),
                goal_rel,
                goal_dir[:, None, :].expand(shape),
                visible.unsqueeze(-1),
                payload_block,
                self.latched.to(torch.float32).unsqueeze(-1),
                time_frac.expand(self.num_envs, self.num_robots, 1),
            ),
            dim=-1,
        )

    def _comm_mask(self) -> Tensor:
        """Return the message mask [E, K, K]. The payload occludes a link that crosses it."""
        offset = self.robot_pos[:, :, None, :] - self.robot_pos[:, None, :, :]
        in_range = torch.linalg.vector_norm(offset, dim=-1) < self.cfg.comm_range
        local = to_local(self.robot_pos, self.payload)
        blocked = segment_hits_box(
            local[:, :, None, :], local[:, None, :, :], self.cfg.payload_hx, self.cfg.payload_hy
        )
        self_link = torch.eye(self.num_robots, dtype=torch.bool, device=self.device)
        return in_range & ~blocked & ~self_link

    def _observe(self) -> dict[str, Tensor]:
        local = self._local_obs()
        scale = self.cfg.arena_half
        pose = torch.cat(
            (
                self.payload[:, :2] / scale,
                torch.cos(self.payload[:, 2])[:, None],
                torch.sin(self.payload[:, 2])[:, None],
            ),
            dim=-1,
        )
        team = torch.cat(
            (self.robot_pos / scale, self.latched.to(torch.float32).unsqueeze(-1)), dim=-1
        ).flatten(1)
        shared = torch.cat((pose, team), dim=-1)[:, None, :]
        full = torch.cat((local, shared.expand(-1, self.num_robots, -1)), dim=-1)
        return {"local": local, "full": full, "comm": self._comm_mask()}

    def state(self) -> dict[str, Tensor]:
        """Return the privileged state and the decoder targets of docs/design.md section 5."""
        scale = self.cfg.arena_half
        rel = (self.payload[:, None, :2] - self.robot_pos) / scale
        pose_dir = torch.stack(
            (torch.cos(self.payload[:, 2]), torch.sin(self.payload[:, 2])), dim=-1
        )
        shape = (self.num_envs, self.num_robots, 2)
        decoder_target = torch.cat(
            (
                rel,
                pose_dir[:, None, :].expand(shape),
                self.latched.to(torch.float32).unsqueeze(-1),
                self._visible().to(torch.float32).unsqueeze(-1),
            ),
            dim=-1,
        )
        return {
            "payload": self.payload,
            "goal": self.goal,
            "robot_pos": self.robot_pos,
            "latched": self.latched,
            "decoder_target": decoder_target,
        }


def _cross(arm: Tensor, force: Tensor) -> Tensor:
    """Return the 2D cross product used for the torque about the payload center."""
    return arm[..., 0] * force[..., 1] - arm[..., 1] * force[..., 0]


def _type_rank(types: Tensor) -> Tensor:
    """Return the index of each robot inside its own type group.

    The scripted controller gives a different job to the first and the second gripper, so it needs
    a stable order inside a type.
    """
    rank = torch.zeros_like(types)
    for value in (TYPE_PUSHER, TYPE_GRIPPER, TYPE_SCOUT):
        same = types == value
        rank = torch.where(same, torch.cumsum(same.long(), dim=0) - 1, rank)
    return rank
