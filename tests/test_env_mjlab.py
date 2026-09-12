"""Physics checks for the mjlab port. Slow: each test builds a MuJoCo Warp simulation."""

import math

import pytest
import torch

from swarm.compute import limit_memory
from swarm.env import TYPE_GRIPPER, TYPE_PUSHER, TYPE_SCOUT, EnvConfig

pytest.importorskip("mujoco_warp")
from swarm.env_mjlab import MjlabTransportEnv  # noqa: E402

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
