"""Tests for the transport env and the scripted controller. See docs/design.md sections 3 and 4."""

from __future__ import annotations

import math

import pytest
import torch

from swarm import compute
from swarm.env import (
    DEFAULT_TEAM,
    LOCAL_DIM,
    TYPE_GRIPPER,
    TYPE_PUSHER,
    TYPE_SCOUT,
    EnvConfig,
    TransportEnv,
    full_dim,
    rect_contact,
    wrap_angle,
)
from swarm.scripted import ScriptedController

compute.limit_memory()

CFG = EnvConfig()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# The robot center stops one radius short of the face, so this gap is inside the contact range.
CONTACT_GAP = 0.22


def _place(team, payload, robots, goal=(3.0, 0.0, 0.0)):
    """Return an env with a known pose, so a test can state the exact force that it applies."""
    env = TransportEnv(1, team=team, device="cpu", seed=0)
    env.payload = torch.tensor([list(payload)])
    env.goal = torch.tensor([list(goal)])
    env.robot_pos = torch.tensor([robots])
    env.latched = torch.zeros(1, len(team), dtype=torch.bool)
    env.latch_local = torch.zeros(1, len(team), 2)
    env.step_count = torch.zeros(1, dtype=torch.long)
    return env


def _drive(env, actions, steps):
    """Run the same action for several steps and return the payload displacement."""
    start = env.payload.clone()
    for _ in range(steps):
        env.step(torch.tensor([actions]))
    return env.payload - start


def test_observation_shapes():
    env = TransportEnv(4, device=DEVICE, seed=0)
    obs = env.reset()
    assert obs["local"].shape == (4, 6, LOCAL_DIM)
    assert obs["full"].shape == (4, 6, full_dim(6))
    assert obs["comm"].shape == (4, 6, 6)
    assert obs["comm"].dtype == torch.bool
    assert full_dim(6) == LOCAL_DIM + 4 + 18


def test_local_observation_layout():
    env = TransportEnv(3, device=DEVICE, seed=1)
    obs, _, _, _, _ = env.step(torch.zeros(3, 6, 3, device=DEVICE))
    local = obs["local"]
    scale = CFG.arena_half
    pos, payload, goal = env.robot_pos, env.payload, env.goal

    torch.testing.assert_close(local[..., 0:2], pos / scale)
    torch.testing.assert_close(local[..., 2:5], env.type_onehot.expand(3, -1, -1))
    torch.testing.assert_close(local[..., 5:7], (goal[:, None, :2] - pos) / scale)
    torch.testing.assert_close(local[..., 7], torch.cos(goal[:, 2])[:, None].expand(3, 6))
    torch.testing.assert_close(local[..., 8], torch.sin(goal[:, 2])[:, None].expand(3, 6))

    gap = torch.linalg.vector_norm(payload[:, None, :2] - pos, dim=-1)
    visible = (gap <= env.sense_range).float()
    torch.testing.assert_close(local[..., 9], visible)
    relative = (payload[:, None, :2] - pos) / scale * visible[..., None]
    torch.testing.assert_close(local[..., 10:12], relative)
    torch.testing.assert_close(local[..., 12], torch.cos(payload[:, 2])[:, None] * visible)
    torch.testing.assert_close(local[..., 13], torch.sin(payload[:, 2])[:, None] * visible)
    torch.testing.assert_close(local[..., 14], env.latched.float())
    # One step ran, so the time fraction is one horizon step.
    tick = torch.full((3, 6), 1.0 / CFG.horizon, device=local.device)
    torch.testing.assert_close(local[..., 15], tick)


def test_full_observation_layout():
    env = TransportEnv(2, device=DEVICE, seed=2)
    obs, _, _, _, _ = env.step(torch.zeros(2, 6, 3, device=DEVICE))
    full, scale = obs["full"], CFG.arena_half
    torch.testing.assert_close(full[..., :LOCAL_DIM], obs["local"])
    pose = (env.payload[:, :2] / scale)[:, None, :].expand(2, 6, 2)
    torch.testing.assert_close(full[..., 16:18], pose)
    torch.testing.assert_close(full[..., 18], torch.cos(env.payload[:, 2])[:, None].expand(2, 6))
    torch.testing.assert_close(full[..., 19], torch.sin(env.payload[:, 2])[:, None].expand(2, 6))
    team = torch.cat((env.robot_pos / scale, env.latched.float().unsqueeze(-1)), dim=-1).flatten(1)
    torch.testing.assert_close(full[..., 20:], team[:, None, :].expand(2, 6, 18))


def test_the_sensing_range_follows_the_robot_type():
    """A pusher reaches 2.0 m and a scout reaches 8.0 m. See docs/design.md section 3.5."""
    env = _place(
        (TYPE_PUSHER, TYPE_PUSHER, TYPE_SCOUT),
        (0.0, 0.0, 0.0),
        [[1.5, 0.0], [4.0, 0.0], [4.0, 0.0]],
    )
    obs, _, _, _, _ = env.step(torch.zeros(1, 3, 3))
    visible = obs["local"][0, :, 9]
    torch.testing.assert_close(visible, torch.tensor([1.0, 0.0, 1.0]))
    # The payload block carries the visible flag, so it reads zero for the robot that is too far.
    assert torch.all(obs["local"][0, 1, 10:14] == 0.0)
    assert torch.any(obs["local"][0, 2, 10:14] != 0.0)


def test_three_pushers_cannot_move_the_payload():
    """F_net is 3.0 and the threshold is 3.5. See docs/design.md section 3.3."""
    face = -(CFG.payload_hx + CONTACT_GAP)
    env = _place(
        (TYPE_PUSHER,) * 3, (0.0, 0.0, 0.0), [[face, -0.2], [face, 0.0], [face, 0.2]]
    )
    moved = _drive(env, [[1.0, 0.0, 1.0]] * 3, 5)
    assert torch.allclose(moved, torch.zeros_like(moved), atol=1e-6)


def test_two_latched_grippers_cannot_move_the_payload():
    """F_net is 0.6 and the threshold drops to 1.5. See docs/design.md section 3.3."""
    face = CFG.payload_hy + CONTACT_GAP
    env = _place((TYPE_GRIPPER,) * 2, (0.0, 0.0, 0.0), [[0.0, face], [0.0, -face]])
    env.step(torch.tensor([[[0.0, -1.0, 1.0], [0.0, 1.0, 1.0]]]))
    assert torch.all(env.latched)
    moved = _drive(env, [[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]], 5)
    assert torch.allclose(moved, torch.zeros_like(moved), atol=1e-6)


def test_two_pushers_and_two_latched_grippers_move_the_payload():
    """F_net is 2.6 against a threshold of 1.5, so the payload slides along x."""
    back = -(CFG.payload_hx + CONTACT_GAP)
    side = CFG.payload_hy + CONTACT_GAP
    env = _place(
        (TYPE_PUSHER, TYPE_PUSHER, TYPE_GRIPPER, TYPE_GRIPPER),
        (0.0, 0.0, 0.0),
        [[back, -0.2], [back, 0.2], [0.0, side], [0.0, -side]],
    )
    env.step(torch.tensor([[[1.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, -1.0, 1.0], [0.0, 1.0, 1.0]]]))
    assert int(env.latched.sum()) == 2
    moved = _drive(env, [[1.0, 0.0, 1.0]] * 4, 5)
    expected = (2.6 - 1.5) / CFG.drag_lin * CFG.dt * 5
    assert moved[0, 0].item() == pytest.approx(expected, abs=1e-3)
    assert abs(moved[0, 1].item()) < 1e-6
    assert abs(moved[0, 2].item()) < 1e-6


def test_one_pusher_cannot_move_the_payload():
    face = -(CFG.payload_hx + CONTACT_GAP)
    env = _place((TYPE_PUSHER,), (0.0, 0.0, 0.0), [[face, 0.0]])
    moved = _drive(env, [[1.0, 0.0, 1.0]], 5)
    assert torch.allclose(moved, torch.zeros_like(moved), atol=1e-6)


def test_payload_occludes_the_comm_mask():
    """Robot 0 and robot 1 sit on opposite faces, so the payload blocks the link."""
    env = _place(
        (TYPE_PUSHER,) * 4, (0.0, 0.0, 0.0), [[0.0, 1.0], [0.0, -1.0], [0.0, 1.5], [4.5, 1.0]]
    )
    obs, _, _, _, _ = env.step(torch.zeros(1, 4, 3))
    comm = obs["comm"][0]
    assert not comm[0, 1] and not comm[1, 0]
    assert comm[0, 2] and comm[2, 0]
    assert not comm[0, 3], "robot 3 is beyond the comm range"
    assert not torch.any(torch.diagonal(comm))


def test_comm_mask_is_clear_beside_the_payload():
    """Two robots that stand on the same side of the payload keep their link."""
    env = _place((TYPE_PUSHER,) * 2, (0.0, 0.0, 0.0), [[1.0, 0.0], [-1.0, 0.0]])
    obs, _, _, _, _ = env.step(torch.zeros(1, 2, 3))
    assert not obs["comm"][0, 0, 1]
    env.robot_pos = torch.tensor([[[1.0, 1.0], [1.0, -1.0]]])
    env.payload = torch.tensor([[0.0, 0.0, math.pi / 2]])
    obs, _, _, _, _ = env.step(torch.zeros(1, 2, 3))
    assert obs["comm"][0, 0, 1]


def test_success_terminates_and_auto_resets():
    env = TransportEnv(4, team=DEFAULT_TEAM, device="cpu", seed=5)
    env.payload = env.goal.clone()
    obs, reward, terminated, truncated, info = env.step(torch.zeros(4, 6, 3))

    assert torch.all(terminated)
    assert not torch.any(truncated)
    torch.testing.assert_close(reward, torch.ones(4))
    torch.testing.assert_close(info["success"], terminated)
    assert torch.all(info["final_pos_error"] < CFG.pos_tol)
    assert torch.all(info["final_angle_error"] < CFG.ang_tol)

    # The ended episode ran one step. The fresh episode starts the clock again.
    assert info["final_local"].shape == (4, 6, LOCAL_DIM)
    assert info["final_full"].shape == (4, 6, full_dim(6))
    torch.testing.assert_close(info["final_full"][..., :LOCAL_DIM], info["final_local"])
    torch.testing.assert_close(
        info["final_local"][..., 15], torch.full((4, 6), 1.0 / CFG.horizon)
    )
    assert torch.all(obs["local"][..., 15] == 0.0)
    assert torch.all(env.step_count == 0)


def test_truncation_at_the_horizon():
    env = TransportEnv(2, device="cpu", seed=6)
    for _ in range(CFG.horizon - 1):
        _, _, terminated, truncated, _ = env.step(torch.zeros(2, 6, 3))
        assert not torch.any(truncated)
    _, reward, terminated, truncated, info = env.step(torch.zeros(2, 6, 3))
    assert torch.all(truncated)
    assert not torch.any(terminated)
    assert torch.all(reward == 0.0)
    torch.testing.assert_close(info["final_local"][..., 15], torch.ones(2, 6))


def test_info_matches_the_observation_when_no_episode_ends():
    env = TransportEnv(4, device="cpu", seed=7)
    obs, _, terminated, truncated, info = env.step(torch.zeros(4, 6, 3))
    assert not torch.any(terminated | truncated)
    torch.testing.assert_close(info["final_local"], obs["local"])
    torch.testing.assert_close(info["final_full"], obs["full"])


def test_decoder_target_layout():
    env = TransportEnv(3, device="cpu", seed=8)
    state = env.state()
    target, scale = state["decoder_target"], CFG.arena_half
    assert target.shape == (3, 6, 6)
    relative = (env.payload[:, None, :2] - env.robot_pos) / scale
    torch.testing.assert_close(target[..., 0:2], relative)
    torch.testing.assert_close(target[..., 2], torch.cos(env.payload[:, 2])[:, None].expand(3, 6))
    torch.testing.assert_close(target[..., 3], torch.sin(env.payload[:, 2])[:, None].expand(3, 6))
    torch.testing.assert_close(target[..., 4], env.latched.float())
    gap = torch.linalg.vector_norm(env.payload[:, None, :2] - env.robot_pos, dim=-1)
    torch.testing.assert_close(target[..., 5], (gap <= env.sense_range).float())


def test_a_robot_never_ends_inside_the_payload_at_the_wall():
    """The arena clamp must not push a robot back into a payload that stands against a wall."""
    env = TransportEnv(1, device="cpu", seed=13)
    env.payload = torch.tensor([[env.center_limit, 0.0, 0.0]])
    env.goal = torch.tensor([[0.0, 0.0, 0.0]])
    # This gap between the payload face and the wall is narrower than the robot limit.
    env.robot_pos = torch.full((1, 6, 2), 4.95)
    env.robot_pos[..., 1] = 0.0
    env.latched = torch.zeros(1, 6, dtype=torch.bool)
    obs, _, _, _, _ = env.step(torch.zeros(1, 6, 3))

    clear = rect_contact(env.robot_pos, env.payload, CFG.payload_hx, CFG.payload_hy).sdist
    assert torch.all(clear >= CFG.robot_radius - 1e-4)
    # A robot inside the payload would lose every comm link to the slab test.
    assert int(obs["comm"][0, 0].sum()) == 5


def test_reset_keeps_the_robots_clear_of_the_payload():
    env = TransportEnv(64, device=DEVICE, seed=9)
    clear = rect_contact(env.robot_pos, env.payload, CFG.payload_hx, CFG.payload_hy).sdist
    assert torch.all(clear >= CFG.start_clear - 1e-4)
    gap = torch.linalg.vector_norm(env.goal[:, :2] - env.payload[:, :2], dim=-1)
    assert torch.all(gap >= CFG.goal_dist_min - 1e-4)
    assert torch.all(gap <= CFG.goal_dist_max + 1e-4)
    turn = wrap_angle(env.goal[:, 2] - env.payload[:, 2]).abs()
    assert torch.all(turn <= CFG.goal_angle_max + 1e-4)


def test_reset_of_a_subset_leaves_the_other_envs_alone():
    env = TransportEnv(4, device="cpu", seed=10)
    keep = env.payload[1:3].clone()
    env.reset(torch.tensor([0, 3]))
    torch.testing.assert_close(env.payload[1:3], keep)


def test_step_rejects_a_wrong_action_shape():
    env = TransportEnv(2, device="cpu", seed=11)
    with pytest.raises(ValueError, match="actions must have shape"):
        env.step(torch.zeros(2, 5, 3))


def test_the_seed_replicates_a_rollout():
    first = TransportEnv(8, device="cpu", seed=12)
    second = TransportEnv(8, device="cpu", seed=12)
    action = torch.zeros(8, 6, 3)
    for _ in range(5):
        left, _, _, _, _ = first.step(action)
        right, _, _, _, _ = second.step(action)
    torch.testing.assert_close(left["local"], right["local"])


def test_scripted_controller_reaches_the_target_success_rate():
    """The offline buffer needs a controller above 60 percent. See docs/design.md section 4."""
    count = 64
    env = TransportEnv(count, device=DEVICE, seed=0)
    control = ScriptedController(env)
    done = torch.zeros(count, dtype=torch.bool, device=env.device)
    success = torch.zeros_like(done)
    for _ in range(CFG.horizon):
        _, _, terminated, truncated, _ = env.step(control.act())
        success |= terminated & ~done
        done |= terminated | truncated
    assert success.float().mean().item() >= 0.6
