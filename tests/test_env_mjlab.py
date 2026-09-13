"""Physics checks for the mjlab port. Slow: each test builds a MuJoCo Warp simulation."""

import math

import pytest
import torch

from swarm.compute import limit_memory
from swarm.env import TYPE_GRIPPER, TYPE_PUSHER, TYPE_SCOUT, EnvConfig, to_world_vec

pytest.importorskip("mujoco_warp")
from swarm.env_mjlab import MjlabConfig, MjlabTransportEnv  # noqa: E402

limit_memory()
DEV = "cuda"
CFG = EnvConfig()


def place(env: MjlabTransportEnv, payload, robots) -> None:
    """Put the payload at a pose and the robots at positions in every env, then sync the sim."""
    env.payload[:] = torch.tensor(payload, device=DEV)
    env.robot_pos[:] = torch.tensor(robots, device=DEV)
    env.latched[:] = False
    env.step_count[:] = 0
    env._write_state(torch.ones(env.num_envs, dtype=torch.bool, device=DEV))
    env._read_state()


def drive(env: MjlabTransportEnv, actions, steps: int) -> torch.Tensor:
    a = torch.tensor(actions, device=DEV).unsqueeze(0).expand(env.num_envs, -1, -1).clone()
    start = env.payload[:, :2].clone()
    for _ in range(steps):
        env.step(a)
    return (env.payload[:, :2] - start).norm(dim=-1)


def behind(k: int, spread: float = 0.5):
    """k robots on the -x face of a payload at the origin, ready to push toward +x."""
    x = -(CFG.payload_hx + CFG.robot_radius + 0.05)
    ys = [(i - (k - 1) / 2) * spread for i in range(k)]
    return [[x, y] for y in ys]


def test_one_pusher_cannot_move_the_payload():
    env = MjlabTransportEnv(2, team=(TYPE_PUSHER,), device=DEV)
    place(env, [0.0, 0.0, 0.0], behind(1))
    assert drive(env, [[1.0, 0.0, 1.0]], 20).max() < 0.05


def test_three_pushers_cannot_move_the_payload():
    env = MjlabTransportEnv(2, team=(TYPE_PUSHER,) * 3, device=DEV)
    place(env, [0.0, 0.0, 0.0], behind(3))
    assert drive(env, [[1.0, 0.0, 1.0]] * 3, 20).max() < 0.05


def test_two_latched_grippers_alone_cannot_move_the_payload():
    env = MjlabTransportEnv(2, team=(TYPE_GRIPPER,) * 2, device=DEV)
    x = -(CFG.payload_hx + CFG.robot_radius + 0.02)
    place(env, [0.0, 0.0, 0.0], [[x, 0.2], [x, -0.2]])
    env.step(torch.tensor([[[0.0, 0.0, 1.0]] * 2] * 2, device=DEV))  # latch
    assert bool(env.latched.all())
    assert drive(env, [[-1.0, 0.0, 1.0]] * 2, 20).max() < 0.05


def test_two_pushers_and_two_latched_grippers_move_the_payload():
    env = MjlabTransportEnv(2, team=(TYPE_PUSHER, TYPE_PUSHER, TYPE_GRIPPER, TYPE_GRIPPER), device=DEV)
    x = -(CFG.payload_hx + CFG.robot_radius + 0.05)
    xg = CFG.payload_hx + CFG.robot_radius + 0.02
    place(env, [0.0, 0.0, 0.0], [[x, 0.25], [x, -0.25], [xg, 0.2], [xg, -0.2]])
    env.step(torch.tensor([[[0.0, 0.0, 0.0]] * 2 + [[0.0, 0.0, 1.0]] * 2] * 2, device=DEV))
    assert bool(env.latched[:, 2:].all())
    moved = drive(env, [[1.0, 0.0, 1.0]] * 2 + [[1.0, 0.0, 1.0]] * 2, 20)
    assert moved.min() > 0.3, moved


def test_scripted_controller_succeeds_often():
    from swarm.scripted import ScriptedController

    import dataclasses

    env = MjlabTransportEnv(32, device=DEV, seed=1, cfg=dataclasses.replace(CFG, v_max=2.5))
    ctrl = ScriptedController(env, noise=0.0)
    done = torch.zeros(32, dtype=torch.bool, device=DEV)
    success = torch.zeros(32, dtype=torch.bool, device=DEV)
    for _ in range(env.cfg.horizon):
        _, _, term, trunc, _ = env.step(ctrl.act())
        success |= term & ~done
        done |= term | trunc
    assert success.float().mean().item() >= 0.6, success.float().mean().item()


# Fidelity options. Each one defaults to the behavior the tests above measure, so the checks
# below construct an env with the option on. See docs/mjlab_fidelity.md.


def closest_gap(env: MjlabTransportEnv, actions, steps: int) -> float:
    """Drive the team and return the smallest centre to centre gap of robots 0 and 1."""
    a = torch.tensor(actions, device=DEV).unsqueeze(0).expand(env.num_envs, -1, -1).clone()
    gap = float("inf")
    for _ in range(steps):
        env.step(a)
        gap = min(gap, (env.robot_pos[:, 0] - env.robot_pos[:, 1]).norm(dim=-1).min().item())
    return gap


def drive_apart(collide: bool) -> float:
    """Two pushers 2 m apart drive into each other, far from the payload."""
    env = MjlabTransportEnv(2, team=(TYPE_PUSHER,) * 2, device=DEV,
                            mj=MjlabConfig(robot_collision=collide))
    place(env, [0.0, 3.5, 0.0], [[-1.0, -2.0], [1.0, -2.0]])
    return closest_gap(env, [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], 20)


def test_robot_collision_stops_robots_at_contact():
    # Two discs of radius 0.2 touch at a centre gap of 0.4 m. Without the option they pass
    # through each other, so the gap falls to nearly zero.
    assert drive_apart(False) < 0.10
    assert drive_apart(True) > 0.30


def payload_attitude(env: MjlabTransportEnv) -> tuple[float, float, float]:
    """Return the payload height and the absolute roll and pitch in degrees."""
    d = env._sim.data
    z = d.xpos[:, env._payload_body][:, 2]
    q = d.xquat[:, env._payload_body]
    roll = torch.atan2(2 * (q[:, 0] * q[:, 1] + q[:, 2] * q[:, 3]), 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2))
    pitch = torch.asin((2 * (q[:, 0] * q[:, 2] - q[:, 3] * q[:, 1])).clamp(-1.0, 1.0))
    return (z.max().item(), math.degrees(roll.abs().max().item()),
            math.degrees(pitch.abs().max().item()))


def latch_one_end(mj: MjlabConfig) -> MjlabTransportEnv:
    """Latch one gripper at the middle of the +x face of a payload at the origin."""
    env = MjlabTransportEnv(2, team=(TYPE_GRIPPER,), device=DEV, mj=mj)
    place(env, [0.0, 0.0, 0.0], [[CFG.payload_hx + CFG.robot_radius + 0.02, 0.0]])
    hold = torch.tensor([[[0.0, 0.0, 1.0]]], device=DEV).expand(2, -1, -1).clone()
    env.step(hold)
    assert bool(env.latched.all())
    for _ in range(20):
        env.step(hold)
    return env


def test_lift_at_latch_keeps_the_payload_flat_on_the_floor():
    env = latch_one_end(MjlabConfig(lift="at_latch"))
    z, roll, pitch = payload_attitude(env)
    assert abs(z - env.mj.payload_hz) < 0.02, z
    assert roll < 5.0 and pitch < 5.0, (roll, pitch)


def test_lift_at_latch_carries_a_moment_that_the_central_lift_does_not():
    # At 45 N the moment about the far edge beats the restoring moment, so the box tips. The
    # check proves the moment reaches the payload; the default 25 N stays below this threshold.
    _, _, central = payload_attitude(latch_one_end(MjlabConfig(lift_force=45.0)))
    _, _, at_latch = payload_attitude(latch_one_end(MjlabConfig(lift="at_latch", lift_force=45.0)))
    assert central < 0.1, central
    assert at_latch > 5.0, at_latch


def anchor_error(env: MjlabTransportEnv) -> torch.Tensor:
    """Return the distance from each gripper body to the payload point its equality anchors."""
    d = env._sim.data
    grip = d.xpos[:, env._robot_body[env._latch_robot]]
    payload = d.xpos[:, env._payload_body]
    local = (env.latch_local + env.latch_normal_local * (CFG.robot_radius + 0.01))[:, env._latch_robot]
    world = payload[:, None, :2] + to_world_vec(local, env.payload)
    return torch.cat([grip[..., :2] - world, grip[..., 2:] - payload[:, None, 2:]], -1).norm(dim=-1)


def test_connect_latch_holds_its_anchor_while_the_payload_moves():
    env = MjlabTransportEnv(2, team=(TYPE_PUSHER, TYPE_PUSHER, TYPE_GRIPPER, TYPE_GRIPPER),
                            device=DEV, mj=MjlabConfig(latch="connect"))
    x = -(CFG.payload_hx + CFG.robot_radius + 0.05)
    xg = CFG.payload_hx + CFG.robot_radius + 0.02
    place(env, [0.0, 0.0, 0.0], [[x, 0.25], [x, -0.25], [xg, 0.2], [xg, -0.2]])
    env.step(torch.tensor([[[0.0, 0.0, 0.0]] * 2 + [[0.0, 0.0, 1.0]] * 2] * 2, device=DEV))
    assert bool(env.latched[:, 2:].all())
    start = env.payload[:, :2].clone()
    push = torch.tensor([[[1.0, 0.0, 1.0]] * 4] * 2, device=DEV)
    worst = 0.0
    for _ in range(20):  # 20 control steps are 200 physics steps.
        env.step(push)
        worst = max(worst, anchor_error(env).max().item())
    moved = (env.payload[:, :2] - start).norm(dim=-1).min().item()
    assert moved > 0.3, moved
    assert worst < 0.02, worst


def test_connect_latch_keeps_the_friction_facts():
    mj = MjlabConfig(latch="connect")
    alone = MjlabTransportEnv(2, team=(TYPE_GRIPPER,) * 2, device=DEV, mj=mj)
    x = -(CFG.payload_hx + CFG.robot_radius + 0.02)
    place(alone, [0.0, 0.0, 0.0], [[x, 0.2], [x, -0.2]])
    alone.step(torch.tensor([[[0.0, 0.0, 1.0]] * 2] * 2, device=DEV))
    assert bool(alone.latched.all())
    assert drive(alone, [[-1.0, 0.0, 1.0]] * 2, 20).max() < 0.05

    team = MjlabTransportEnv(2, team=(TYPE_PUSHER, TYPE_PUSHER, TYPE_GRIPPER, TYPE_GRIPPER),
                             device=DEV, mj=mj)
    xp = -(CFG.payload_hx + CFG.robot_radius + 0.05)
    xg = CFG.payload_hx + CFG.robot_radius + 0.02
    place(team, [0.0, 0.0, 0.0], [[xp, 0.25], [xp, -0.25], [xg, 0.2], [xg, -0.2]])
    team.step(torch.tensor([[[0.0, 0.0, 0.0]] * 2 + [[0.0, 0.0, 1.0]] * 2] * 2, device=DEV))
    assert bool(team.latched[:, 2:].all())
    moved = drive(team, [[1.0, 0.0, 1.0]] * 4, 20)
    assert moved.min() > 0.3, moved


def pusher_contacts(shape: str, yaw: float) -> int:
    """Return the largest contact count between the pusher geom and the payload geom."""
    import mujoco

    env = MjlabTransportEnv(2, team=(TYPE_PUSHER,), device=DEV,
                           mj=MjlabConfig(pusher_shape=shape, payload_mass=2.0))
    model = env._sim.mj_model
    pg = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "payload_geom")
    rg = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "robot0_geom")
    place(env, [0.0, 0.0, yaw], [[-(CFG.payload_hx + CFG.robot_radius + 0.05), 0.1]])
    push = torch.tensor([[[1.0, 0.0, 1.0]]] * 2, device=DEV)
    best = 0
    for _ in range(20):
        env.step(push)
        d = env._sim.data
        n = int(d.nacon[:][0])
        geom, world = d.contact.geom[:n], d.contact.worldid[:n]
        pair = ((geom[:, 0] == pg) & (geom[:, 1] == rg)) | ((geom[:, 0] == rg) & (geom[:, 1] == pg))
        for w in range(env.num_envs):
            best = max(best, int((pair & (world == w)).sum()))
    return best


def test_box_pusher_gets_more_than_one_contact_point():
    # A cylinder against a box takes the convex path, which mujoco_warp caps at one point. A box
    # against a box takes the multicontact path: four points on a parallel face, two on a
    # rotated one.
    assert pusher_contacts("cylinder", 0.0) == 1
    assert pusher_contacts("cylinder", 0.25) == 1
    assert pusher_contacts("box", 0.0) > 1
    assert pusher_contacts("box", 0.25) > 1


def off_centre_yaw(shape: str) -> float:
    """Two pushers press the same half of the long face, so the push is off centre."""
    env = MjlabTransportEnv(2, team=(TYPE_PUSHER,) * 2, device=DEV,
                            mj=MjlabConfig(pusher_shape=shape, payload_mass=2.0))
    y = -(CFG.payload_hy + CFG.robot_radius + 0.05)
    place(env, [0.0, 0.0, 0.0], [[0.45, y], [0.0, y]])
    push = torch.tensor([[[0.0, 1.0, 1.0]] * 2] * 2, device=DEV)
    for _ in range(20):
        env.step(push)
    return env.payload[:, 2].abs().min().item()


def test_box_pusher_resists_yaw_that_a_cylinder_pusher_cannot():
    # The contact patch of a box carries a couple, so an off centre push turns the payload less,
    # not more. A cylinder touches at one point and cannot resist the turn at all.
    cylinder, box = off_centre_yaw("cylinder"), off_centre_yaw("box")
    assert cylinder > 0.5, cylinder
    assert box < 0.2 * cylinder, (cylinder, box)
