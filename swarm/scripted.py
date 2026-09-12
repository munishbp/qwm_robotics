"""Scripted centralized controller for the transport task. See docs/design.md section 4.

The controller reads the privileged state and drives every robot toward a target point in the
payload frame. It fills the offline buffer, so it has to succeed often. It does not have to be
optimal.

The plan has two phases. The transport phase puts every pusher on the face opposite the goal and
shifts them along that face to turn the payload while it travels. The turning phase starts near the
goal and splits the pushers onto the two long faces, where they form a force couple. A couple turns
the payload and does not move it off the goal.
"""

from __future__ import annotations

import torch
from torch import Tensor

from swarm.env import Contact, TransportEnv, rect_contact, to_local, to_world_vec, wrap_angle

# Gains. The env clips the action, so a move gain is a normalized command per meter of error.
MOVE_GAIN = 5.0
ANGLE_GAIN = 4.0
OFFSET_FRAC = 0.9
PRESS = 0.05
RING_RADIUS = 1.5
ALIGN_TOL = 0.4
ORBIT_STEP = 0.9
PUSH_SCALE = 0.5
GRIP_ROT_GAIN = 3.0
SCOUT_STANDOFF = 2.0
ENDGAME_DIST = 0.6
ENDGAME_ANGLE = 0.12
COUPLE_FRAC = 0.85


class ScriptedController:
    """Produce actions [E, K, 3] for one batched env from its privileged state."""

    def __init__(self, env: TransportEnv, noise: float = 0.0) -> None:
        if noise < 0.0:
            raise ValueError(f"noise must not be negative, got {noise}.")
        self.env = env
        self.noise = float(noise)

    def act(self) -> Tensor:
        """Return one action per robot. The caller passes it straight to env.step."""
        env = self.env
        cfg = env.cfg
        state = env.state()
        payload, goal = state["payload"], state["goal"]
        pos, latched = state["robot_pos"], state["latched"]

        to_goal = goal[:, :2] - payload[:, :2]
        distance = torch.linalg.vector_norm(to_goal, dim=-1)
        heading = to_goal / distance.clamp(min=1e-6)[:, None]
        angle_error = wrap_angle(goal[:, 2] - payload[:, 2])
        turn = _sign(angle_error)
        turning = (distance < ENDGAME_DIST) & (angle_error.abs() > ENDGAME_ANGLE)

        # The goal direction in the payload frame picks the back face and the shift that turns it.
        local_dir = to_local(payload[:, None, :2] + heading[:, None, :], payload)[:, 0]
        use_x = local_dir[:, 0].abs() >= local_dir[:, 1].abs()
        face = torch.where(use_x, -_sign(local_dir[:, 0]), -_sign(local_dir[:, 1]))
        tan_half = torch.where(use_x, cfg.payload_hy, cfg.payload_hx)
        torque_sign = torch.where(use_x, face, -face)
        shift = torque_sign * (ANGLE_GAIN * angle_error).clamp(-1.0, 1.0) * tan_half * OFFSET_FRAC

        target, normal, busy = self._targets(use_x, face, shift, turn, turning)
        contact = rect_contact(pos, payload, cfg.payload_hx, cfg.payload_hy)
        command = self._approach(pos, payload, target, normal, busy)
        control = self._effort(contact, target, normal, busy, latched, distance, turning)
        command = self._gripper_force(
            command, pos, payload, latched, heading, distance, angle_error, turn, turning
        )
        command = self._scout_move(command, pos, payload, heading)

        action = torch.cat((command, control.unsqueeze(-1)), dim=-1)
        if self.noise > 0.0:
            action = action + torch.randn_like(action) * self.noise
        return action.clamp(-1.0, 1.0)

    def _targets(
        self, use_x: Tensor, face: Tensor, shift: Tensor, turn: Tensor, turning: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return the target point, the face normal, and the busy flag for every robot."""
        env = self.env
        cfg = env.cfg
        shape = (env.num_envs, env.num_robots)
        rank = env.rank[None, :].expand(shape)
        pusher = env.is_pusher[None, :].expand(shape)
        gripper = env.is_gripper[None, :].expand(shape)
        flat = use_x[:, None].expand(shape)
        zero = torch.zeros(shape, device=env.device)

        back_point = torch.stack(
            (
                torch.where(use_x, face * cfg.payload_hx, shift),
                torch.where(use_x, shift, face * cfg.payload_hy),
            ),
            dim=-1,
        )[:, None, :].expand(*shape, 2)
        back_normal = torch.stack(
            (
                torch.where(use_x, face, torch.zeros_like(face)),
                torch.where(use_x, torch.zeros_like(face), face),
            ),
            dim=-1,
        )[:, None, :].expand(*shape, 2)

        # The couple pushes the two long faces in opposite directions, so the net force is zero.
        arm = turn[:, None].expand(shape) * COUPLE_FRAC * cfg.payload_hx
        side = torch.where(rank == 0, 1.0, -1.0)
        couple_point = torch.stack((-arm * side, side * cfg.payload_hy), dim=-1)
        couple_normal = torch.stack((zero, side), dim=-1)

        # A gripper works a side face, which is the face that the back face does not use.
        grip_point = torch.stack(
            (
                torch.where(flat, zero, side * cfg.payload_hx),
                torch.where(flat, side * cfg.payload_hy, zero),
            ),
            dim=-1,
        )
        grip_normal = torch.stack(
            (torch.where(flat, zero, side), torch.where(flat, side, zero)), dim=-1
        )

        couple = (turning[:, None] & pusher & (rank < 2)).unsqueeze(-1)
        target = torch.where(couple, couple_point, back_point)
        normal = torch.where(couple, couple_normal, back_normal)
        target = torch.where(gripper.unsqueeze(-1), grip_point, target)
        normal = torch.where(gripper.unsqueeze(-1), grip_normal, normal)

        spare = turning[:, None] & pusher & (rank >= 2)
        return target, normal, (pusher | gripper) & ~spare

    def _approach(
        self, pos: Tensor, payload: Tensor, target: Tensor, normal: Tensor, busy: Tensor
    ) -> Tensor:
        """Drive every robot to its face. A robot behind the payload orbits it first.

        The orbit keeps a pusher off the wrong faces on the way in. A pusher never pushes the
        payload sideways while it travels, because it also holds its effort at zero until it
        stands on the face that its job names.
        """
        local = to_local(pos, payload)
        radius = torch.linalg.vector_norm(local, dim=-1, keepdim=True).clamp(min=1e-6)
        unit = local / radius
        cross = unit[..., 0] * normal[..., 1] - unit[..., 1] * normal[..., 0]
        bearing = torch.atan2(cross, (unit * normal).sum(dim=-1))
        step = bearing.clamp(-ORBIT_STEP, ORBIT_STEP)
        cos, sin = torch.cos(step), torch.sin(step)
        orbit = (
            torch.stack(
                (unit[..., 0] * cos - unit[..., 1] * sin, unit[..., 0] * sin + unit[..., 1] * cos),
                dim=-1,
            )
            * RING_RADIUS
        )
        aligned = (bearing.abs() < ALIGN_TOL).unsqueeze(-1)
        goto = torch.where(aligned, target + normal * PRESS, orbit)
        world = payload[:, None, :2] + to_world_vec(goto, payload)
        return _unit_command(world - pos) * busy.unsqueeze(-1)

    def _effort(
        self,
        contact: Contact,
        target: Tensor,
        normal: Tensor,
        busy: Tensor,
        latched: Tensor,
        distance: Tensor,
        turning: Tensor,
    ) -> Tensor:
        """Return the third action element for every robot.

        A pusher pushes only from the face that its job names, so a stray touch on the way in
        applies no force. A gripper holds its latch for the whole episode, because a release raises
        the friction threshold and helps nobody.
        """
        env = self.env
        cfg = env.cfg
        drift = ((contact.local - target) * normal.abs()).sum(dim=-1).abs()
        touching = contact.sdist <= cfg.contact_dist
        on_face = (drift < 1e-3) & touching
        near = (distance / PUSH_SCALE).clamp(0.0, 1.0)[:, None]
        effort = torch.where(turning[:, None], torch.ones_like(near), near)
        push = effort * (on_face & busy & env.is_pusher[None, :]).to(contact.sdist.dtype)
        grip = (latched | touching).to(contact.sdist.dtype)
        return torch.where(env.is_gripper[None, :], grip, push)

    def _gripper_force(
        self,
        command: Tensor,
        pos: Tensor,
        payload: Tensor,
        latched: Tensor,
        heading: Tensor,
        distance: Tensor,
        angle_error: Tensor,
        turn: Tensor,
        turning: Tensor,
    ) -> Tensor:
        """Replace the move command of a latched gripper with the force that it applies.

        A latched gripper sits at its latch point, so its first two action elements are a force
        direction and not a velocity.
        """
        arm = pos - payload[:, None, :2]
        span = torch.linalg.vector_norm(arm, dim=-1, keepdim=True).clamp(min=1e-6)
        # A force along the tangent at the latch point turns the payload the positive way.
        tangent = torch.stack((-arm[..., 1], arm[..., 0]), dim=-1) / span
        pull = heading[:, None, :] * (distance / PUSH_SCALE).clamp(0.0, 1.0)[:, None, None]
        spin = tangent * (GRIP_ROT_GAIN * angle_error).clamp(-1.0, 1.0)[:, None, None]
        force = torch.where(turning[:, None, None], tangent * turn[:, None, None], pull + spin)
        force = force / torch.linalg.vector_norm(force, dim=-1, keepdim=True).clamp(min=1.0)
        hold = (self.env.is_gripper[None, :] & latched).unsqueeze(-1)
        return torch.where(hold, force, command)

    def _scout_move(
        self, command: Tensor, pos: Tensor, payload: Tensor, heading: Tensor
    ) -> Tensor:
        """Keep the scout on the goal side of the payload. It applies no force."""
        world = payload[:, None, :2] + heading[:, None, :] * SCOUT_STANDOFF
        scout = (self.env.types == 2)[None, :, None]
        return torch.where(scout, _unit_command(world - pos), command)


def _unit_command(error: Tensor) -> Tensor:
    """Return a proportional command whose length never passes one."""
    scaled = error * MOVE_GAIN
    return scaled / torch.linalg.vector_norm(scaled, dim=-1, keepdim=True).clamp(min=1.0)


def _sign(x: Tensor) -> Tensor:
    """Return +1 or -1. Plain sign returns 0 at the origin and that stalls a face choice."""
    return x.ge(0).to(x.dtype) * 2.0 - 1.0
