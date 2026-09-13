"""The transport task on mjlab (MuJoCo Warp).

The class keeps every non physical part of the 2D task (observation layout, sensing ranges,
occlusion, message mask, success test, reset sampling, decoder target) by inheriting from
`TransportEnv`, and replaces the quasi static dynamics with a MuJoCo Warp simulation through
`mjlab.sim.Simulation`. See docs/design.md section 3 for the task and docs/mjlab_port.md for
the mapping.

Physics mapping, in the units of the 2D task times ten:

- The payload is a free box of 8 kg on a plane with friction 0.5, so the static threshold is
  about 39 N. Three pushers at 10 N each cannot move it. One latched gripper unloads it by 25 N
  (12.5 N of threshold), so three pushers can against 27 N; two latched grippers unload it by
  50 N, so two pushers can against 14 N. Two grippers alone pull 6 N against 14 N and cannot.
- A pusher is a cylinder on two slide joints with velocity actuators limited to 10 N per axis.
  It pushes by driving into the payload, so the contact force is real. With `u > 0` in contact
  the env adds an inward velocity command so the robot presses on the face it touches.
- A gripper latches kinematically: while `u > 0` and in contact its position is written to the
  latch point every substep, its pull is applied to the payload at that point as an external
  force of at most 3 N, and a 20 N upward force at the payload center unloads the friction.
- The scout is a weak robot that cannot move the payload.
- Walls are static boxes. Robots are limited to the arena by joint ranges. Robots do not
  collide with each other or with the floor.

`MjlabConfig` carries four fidelity options. Every one defaults to the behavior above, so an
existing result stays reproducible. See docs/mjlab_fidelity.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import torch
from torch import Tensor

from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.sim.sim import Simulation
from swarm.env import (
    TYPE_GRIPPER,
    TYPE_PUSHER,
    TYPE_SCOUT,
    DEFAULT_TEAM,
    EnvConfig,
    TransportEnv,
    rect_contact,
    to_world_vec,
    wrap_angle,
)


@dataclass
class MjlabConfig:
    physics_dt: float = 0.01
    decimation: int = 10
    payload_mass: float = 8.0
    payload_hz: float = 0.15
    friction: float = 0.5
    robot_height: float = 0.3
    # Force limits per type in newtons (pusher, gripper, scout), ten times the 2D values.
    force_limit: tuple[float, float, float] = (10.0, 3.0, 3.0)
    velocity_gain: float = 60.0
    grip_force: float = 3.0
    lift_force: float = 25.0
    nconmax: int = 24
    njmax: int = 96
    # Fidelity options. Each default reproduces the physics of docs/mjlab_port.md.
    robot_collision: bool = False
    lift: str = "central"
    latch: str = "kinematic"
    pusher_shape: str = "cylinder"

    def __post_init__(self) -> None:
        """Reject a bad option here, because a typo deep in the build gives a worse message."""
        for name, value, allowed in (
            ("lift", self.lift, ("central", "at_latch")),
            ("latch", self.latch, ("kinematic", "connect")),
            ("pusher_shape", self.pusher_shape, ("cylinder", "box")),
        ):
            if value not in allowed:
                raise ValueError(f"MjlabConfig.{name} must be one of {allowed}, got {value!r}.")


def build_spec(team: tuple[int, ...], cfg: EnvConfig, mj: MjlabConfig) -> mujoco.MjSpec:
    """One world: floor, four walls, the payload box, and one body per robot."""
    spec = mujoco.MjSpec()
    spec.option.timestep = mj.physics_dt
    spec.compiler.autolimits = True
    a = cfg.arena_half
    spec.worldbody.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[a + 1, a + 1, 0.1],
        contype=4, conaffinity=4, friction=[mj.friction, 0.005, 0.0001],
    )
    for name, pos, size in (
        ("wall_px", [a + 0.1, 0, 0.3], [0.1, a + 0.2, 0.3]),
        ("wall_nx", [-a - 0.1, 0, 0.3], [0.1, a + 0.2, 0.3]),
        ("wall_py", [0, a + 0.1, 0.3], [a + 0.2, 0.1, 0.3]),
        ("wall_ny", [0, -a - 0.1, 0.3], [a + 0.2, 0.1, 0.3]),
    ):
        spec.worldbody.add_geom(
            name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size, contype=4, conaffinity=4
        )
    payload = spec.worldbody.add_body(name="payload", pos=[0, 0, mj.payload_hz])
    payload.add_freejoint(name="payload_free")
    payload.add_geom(
        name="payload_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[cfg.payload_hx, cfg.payload_hy, mj.payload_hz], mass=mj.payload_mass,
        contype=5, conaffinity=5, friction=[mj.friction, 0.005, 0.0001],
    )
    limit = a - cfg.robot_radius
    # Bit 0 and bit 1 carry the robot, bit 2 the floor and the walls. A robot pair matches only
    # when both robot masks hold both bits, so one flag turns robot to robot collision on and
    # leaves the floor and the walls out.
    robot_type, robot_aff = (3, 3) if mj.robot_collision else (2, 1)
    for k, t in enumerate(team):
        body = spec.worldbody.add_body(name=f"robot{k}", pos=[0, 0, mj.robot_height / 2])
        body.add_joint(name=f"r{k}x", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0], range=[-limit, limit])
        body.add_joint(name=f"r{k}y", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 1, 0], range=[-limit, limit])
        if t == TYPE_PUSHER and mj.pusher_shape == "box":
            # A box against the payload box takes the multicontact path, so the pusher gets more
            # than one contact point and can carry a moment through its own patch.
            geom_type = mujoco.mjtGeom.mjGEOM_BOX
            geom_size = [cfg.robot_radius, cfg.robot_radius, mj.robot_height / 2]
        else:
            geom_type = mujoco.mjtGeom.mjGEOM_CYLINDER
            geom_size = [cfg.robot_radius, mj.robot_height / 2, 0]
        body.add_geom(
            name=f"robot{k}_geom", type=geom_type, size=geom_size, mass=2.0,
            contype=robot_type, conaffinity=robot_aff,
        )
        if t == TYPE_GRIPPER and mj.latch == "connect":
            # The latch is inactive at compile. A latch writes `eq_active` and `eq_data` per world.
            spec.add_equality(
                name=f"latch{k}", type=mujoco.mjtEq.mjEQ_CONNECT,
                objtype=mujoco.mjtObj.mjOBJ_BODY, name1=f"robot{k}", name2="payload", active=False,
            )
        f = mj.force_limit[t]
        for axis in ("x", "y"):
            act = spec.add_actuator(name=f"a{k}{axis}", target=f"r{k}{axis}", trntype=mujoco.mjtTrn.mjTRN_JOINT)
            # A velocity servo: force = kv (ctrl - qvel), saturating at the force limit.
            act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            act.gainprm[0] = mj.velocity_gain
            act.biasprm[0] = 0.0
            act.biasprm[1] = 0.0
            act.biasprm[2] = -mj.velocity_gain
            act.forcerange = [-f, f]
            act.forcelimited = True
    return spec


class MjlabTransportEnv(TransportEnv):
    """The transport task with MuJoCo Warp dynamics and the 2D task's interface."""

    def __init__(
        self,
        num_envs: int,
        team: tuple[int, ...] = DEFAULT_TEAM,
        device: str | torch.device = "cuda",
        cfg: EnvConfig = EnvConfig(),
        seed: int = 0,
        mj: MjlabConfig = MjlabConfig(),
    ) -> None:
        self.mj = mj
        self._sim: Simulation | None = None
        # The outward normal at the latch point in the payload frame. A latched body sits one
        # radius outside the boundary along it, because unlike the 2D task the body collides.
        self.latch_normal_local = torch.zeros(num_envs, len(team), 2, device=device)
        super().__init__(num_envs, team, device, cfg, seed)

    # Simulation setup.

    def _solver_capacity(self) -> tuple[int, int]:
        """Return the contact and the constraint row budget for the options in use.

        Warning: MuJoCo Warp drops a contact or a row above the budget without an error, and the
        solver then reports a wrong force. So the budget grows with the options that add rows.
        """
        mj = self.mj
        nconmax, njmax = mj.nconmax, mj.njmax
        if mj.robot_collision:
            pairs = self.num_robots * (self.num_robots - 1) // 2
            # A box pair takes the multicontact path and reaches four points; a convex pair one.
            per_pair = 4 if mj.pusher_shape == "box" else 1
            nconmax += pairs * per_pair
            njmax += 3 * pairs * per_pair
        if mj.latch == "connect":
            njmax += 3 * int(self.is_gripper.sum())
        return nconmax, njmax

    def _build_sim(self) -> None:
        mj = self.mj
        spec = build_spec(self.team, self.cfg, mj)
        nconmax, njmax = self._solver_capacity()
        sim_cfg = SimulationCfg(
            nconmax=nconmax, njmax=njmax,
            mujoco=MujocoCfg(timestep=mj.physics_dt, integrator="implicitfast", iterations=30, ls_iterations=20, cone="elliptic", impratio=10.0),
        )
        self._sim = Simulation(self.num_envs, sim_cfg, spec=spec, device=str(self.device))
        model = self._sim.mj_model
        self._payload_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "payload")
        self._payload_qpos = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "payload_free")]
        self._payload_qvel = model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "payload_free")]
        k = self.num_robots
        jx = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"r{i}x") for i in range(k)]
        self._robot_qpos = torch.tensor([model.jnt_qposadr[j] for j in jx], device=self.device)
        self._robot_qvel = torch.tensor([model.jnt_dofadr[j] for j in jx], device=self.device)
        self._robot_ctrl = torch.tensor(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"a{i}x") for i in range(k)], device=self.device
        )
        self._robot_body = torch.tensor(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"robot{i}") for i in range(k)], device=self.device
        )
        self.step_dt = mj.physics_dt * mj.decimation
        if mj.latch == "connect":
            self._build_latch_equalities()

    def _build_latch_equalities(self) -> None:
        """Expand `eq_data` to one row per world and record the equality id of every gripper.

        `expand_model_fields` tiles the field and recaptures the CUDA graph once, here. Every
        equality kernel then reads `eq_data[worldid % eq_data.shape[0]]`, so each world holds its
        own anchor. The gripper side anchor stays at the body origin, so only the payload side
        anchor changes at a latch.
        """
        model = self._sim.mj_model
        gripper = [k for k in range(self.num_robots) if self.team[k] == TYPE_GRIPPER]
        ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, f"latch{k}") for k in gripper]
        if any(i < 0 for i in ids):
            raise RuntimeError("build_spec did not add one latch equality per gripper.")
        self._latch_robot = torch.tensor(gripper, device=self.device)
        self._latch_eq = torch.tensor(ids, device=self.device)
        self._sim.expand_model_fields(("eq_data",))
        rows = int(self._sim.model.eq_data.shape[0])
        if rows != self.num_envs:
            raise RuntimeError(
                f"eq_data has {rows} world rows for {self.num_envs} envs, so a latch would "
                "share one anchor across envs. Use latch='kinematic' with this mjlab version."
            )
        eq_data = self._sim.model.eq_data[:].clone()
        eq_data[:, self._latch_eq, 0:3] = 0.0
        self._sim.model.eq_data[:] = eq_data

    def _write_state(self, mask: Tensor) -> None:
        """Write the payload pose and the robot positions of the masked envs into the simulation."""
        d = self._sim.data
        qpos = d.qpos[:].clone()
        qvel = d.qvel[:].clone()
        p = self._payload_qpos
        half = self.payload[:, 2] / 2
        quat = torch.stack([torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)], -1)
        pose = torch.cat([self.payload[:, :2], torch.full_like(half, self.mj.payload_hz).unsqueeze(-1), quat], -1)
        qpos[:, p:p + 7] = torch.where(mask[:, None], pose, qpos[:, p:p + 7])
        qvel[:, self._payload_qvel:self._payload_qvel + 6] = torch.where(
            mask[:, None], torch.zeros(self.num_envs, 6, device=self.device), qvel[:, self._payload_qvel:self._payload_qvel + 6]
        )
        for axis in range(2):
            idx = self._robot_qpos + axis
            qpos[:, idx] = torch.where(mask[:, None], self.robot_pos[..., axis], qpos[:, idx])
            vidx = self._robot_qvel + axis
            qvel[:, vidx] = torch.where(mask[:, None], torch.zeros_like(qvel[:, vidx]), qvel[:, vidx])
        d.qpos[:] = qpos
        d.qvel[:] = qvel
        d.qacc_warmstart[:] = torch.where(mask[:, None], torch.zeros_like(d.qacc_warmstart[:]), d.qacc_warmstart[:])
        d.xfrc_applied[:] = torch.zeros_like(d.xfrc_applied[:])
        d.ctrl[:] = torch.zeros_like(d.ctrl[:])
        if self.mj.latch == "connect":
            # `_reset_masked` already cleared `latched` for the masked envs. The env owns
            # `eq_active`, because it never calls `Simulation.reset`, which is what restores
            # `eq_active` from `eq_active0`. Without this write a latch leaks into the next
            # episode and `forward` solves against a stale anchor.
            self._write_latch(self.latched)
        self._sim.forward()

    def _write_latch(self, latched: Tensor) -> None:
        """Write the per world anchor and the active flag of every gripper equality.

        The anchor sits one robot radius outside the latch point, so the constraint holds the
        gripper where the kinematic latch used to write it. The write is idempotent: the anchor
        only changes on the substep that latches.
        """
        d, model = self._sim.data, self._sim.model
        anchor = self.latch_local + self.latch_normal_local * (self.cfg.robot_radius + 0.01)
        eq_data = model.eq_data[:].clone()
        eq_data[:, self._latch_eq, 3] = anchor[:, self._latch_robot, 0]
        eq_data[:, self._latch_eq, 4] = anchor[:, self._latch_robot, 1]
        eq_data[:, self._latch_eq, 5] = 0.0
        model.eq_data[:] = eq_data
        active = d.eq_active[:].clone()
        active[:, self._latch_eq] = latched[:, self._latch_robot]
        d.eq_active[:] = active

    def _read_state(self) -> None:
        """Read the payload pose and the robot positions back from the simulation."""
        d = self._sim.data
        xpos = d.xpos[:, self._payload_body]
        q = d.xquat[:, self._payload_body]  # (w, x, y, z)
        yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
        self.payload = torch.stack([xpos[:, 0], xpos[:, 1], yaw], -1)
        qpos = d.qpos[:]
        self.robot_pos = torch.stack([qpos[:, self._robot_qpos], qpos[:, self._robot_qpos + 1]], -1)

    # Overrides.

    def _reset_masked(self, mask: Tensor) -> None:
        super()._reset_masked(mask)
        self.latch_normal_local = torch.where(mask[:, None, None], torch.zeros_like(self.latch_normal_local), self.latch_normal_local)
        if self._sim is None:
            self._build_sim()
            mask = torch.ones_like(mask)
        self._write_state(mask)
        self._read_state()

    def step(
        self, actions: Tensor
    ) -> tuple[dict[str, Tensor], Tensor, Tensor, Tensor, dict[str, Tensor]]:
        expected = (self.num_envs, self.num_robots, 3)
        if actions.shape != expected:
            raise ValueError(f"actions must have shape {expected}, got {tuple(actions.shape)}.")
        cfg, mj, d = self.cfg, self.mj, self._sim.data
        actions = actions.to(self.device, torch.float32).clamp(-1.0, 1.0)
        vel_cmd, control = actions[..., :2], actions[..., 2]
        hold = control > 0.0

        for _ in range(mj.decimation):
            self._read_state()
            contact = rect_contact(self.robot_pos, self.payload, cfg.payload_hx, cfg.payload_hy)
            near = contact.sdist <= cfg.contact_dist
            # Latch decisions follow the 2D rule, evaluated at every substep.
            fresh_latch = self.is_gripper & hold & near & ~self.latched
            self.latch_local = torch.where(fresh_latch.unsqueeze(-1), contact.local, self.latch_local)
            c, sn = torch.cos(self.payload[:, 2])[:, None], torch.sin(self.payload[:, 2])[:, None]
            normal_local = torch.stack(
                [c * contact.normal[..., 0] + sn * contact.normal[..., 1], -sn * contact.normal[..., 0] + c * contact.normal[..., 1]], -1
            )
            self.latch_normal_local = torch.where(fresh_latch.unsqueeze(-1), normal_local, self.latch_normal_local)
            self.latched = (self.latched | fresh_latch) & self.is_gripper & hold
            # Velocity commands. A pusher in contact with u > 0 presses into the face.
            press = (self.is_pusher & hold & near).unsqueeze(-1)
            cmd = vel_cmd * cfg.v_max + torch.where(press, -contact.normal * cfg.v_max, torch.zeros_like(vel_cmd))
            if mj.latch == "kinematic":
                # A kinematically latched gripper is written, not driven, so it takes no command.
                cmd = torch.where(self.latched.unsqueeze(-1), torch.zeros_like(cmd), cmd)
            ctrl = d.ctrl[:].clone()
            ctrl[:, self._robot_ctrl] = cmd[..., 0]
            ctrl[:, self._robot_ctrl + 1] = cmd[..., 1]
            d.ctrl[:] = ctrl
            # Latched grippers sit at their latch point and pull on the payload there.
            center = self.payload[:, None, :2]
            latch_world = center + to_world_vec(self.latch_local, self.payload)
            if mj.latch == "kinematic":
                body_world = center + to_world_vec(
                    self.latch_local + self.latch_normal_local * (cfg.robot_radius + 0.01), self.payload
                )
                if bool(self.latched.any()):
                    qpos = d.qpos[:].clone()
                    qvel = d.qvel[:].clone()
                    for axis in range(2):
                        idx = self._robot_qpos + axis
                        qpos[:, idx] = torch.where(self.latched, body_world[..., axis], qpos[:, idx])
                        vidx = self._robot_qvel + axis
                        qvel[:, vidx] = torch.where(self.latched, torch.zeros_like(qvel[:, vidx]), qvel[:, vidx])
                    d.qpos[:] = qpos
                    d.qvel[:] = qvel
                grip = vel_cmd * mj.grip_force * self.latched.unsqueeze(-1)  # [E, K, 2]
            else:
                # The constraint carries the pull, so the gripper keeps its command and its own
                # actuator supplies the force. A second external pull would double count it.
                self._write_latch(self.latched)
                grip = torch.zeros_like(vel_cmd)
            arm = latch_world - center
            torque_z = (arm[..., 0] * grip[..., 1] - arm[..., 1] * grip[..., 0]).sum(1)
            # The unloading force is bounded so the payload never leaves the floor: four latched
            # grippers would otherwise lift 100 N against a 78.5 N weight.
            weight = mj.payload_mass * 9.81
            count = self.latched.sum(1).to(torch.float32)
            lift = (mj.lift_force * count).clamp(max=0.8 * weight)
            wrench = torch.zeros(self.num_envs, 6, device=self.device)
            wrench[:, 0] = grip[..., 0].sum(1)
            wrench[:, 1] = grip[..., 1].sum(1)
            wrench[:, 2] = lift
            wrench[:, 5] = torque_z
            if mj.lift == "at_latch":
                # `xfrc_applied` acts at the body centre of mass, so the lift at a latch point
                # becomes the same total force plus its moment r cross F about that centre.
                share = (lift / count.clamp(min=1.0)).unsqueeze(-1) * self.latched
                wrench[:, 3] = (arm[..., 1] * share).sum(1)
                wrench[:, 4] = -(arm[..., 0] * share).sum(1)
            xfrc = torch.zeros_like(d.xfrc_applied[:])
            xfrc[:, self._payload_body] = wrench
            d.xfrc_applied[:] = xfrc
            self._sim.step()

        self._sim.forward()
        self._read_state()
        # Keep the payload state in the 2D convention (center inside the arena band).
        self.payload = torch.cat(
            [self.payload[:, :2].clamp(-self.center_limit, self.center_limit), wrap_angle(self.payload[:, 2])[:, None]], -1
        )

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
