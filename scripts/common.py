"""Shared argument handling for the scripts."""

from __future__ import annotations

import json
import os
import random

import numpy as np
import torch

from swarm.compute import clamp_envs, limit_memory
from swarm.env import DEFAULT_TEAM, TYPE_GRIPPER, TYPE_PUSHER, TYPE_SCOUT, EnvConfig, TransportEnv


def team(pushers: int, grippers: int, scouts: int = 1) -> tuple[int, ...]:
    return (TYPE_PUSHER,) * pushers + (TYPE_GRIPPER,) * grippers + (TYPE_SCOUT,) * scouts


TEAMS = {
    "default": tuple(DEFAULT_TEAM),
    "p2g1s1": team(2, 1),
    "p4g4s1": team(4, 4),
    "p6g5s1": team(6, 5),
    "p8g7s1": team(8, 7),
}


def setup(seed: int) -> str:
    limit_memory()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return "cuda" if torch.cuda.is_available() else "cpu"


SIM = os.environ.get("SWARM_SIM", "2d")


def env_config() -> EnvConfig:
    """The task constants for the selected simulator.

    On mjlab the robots are velocity servos that accelerate, so a higher top speed gives the same
    reach per episode as the 2D task's instantaneous motion. Every other constant is shared.
    """
    if SIM == "mjlab":
        return EnvConfig(v_max=2.5)
    if SIM == "2d_momentum":
        return EnvConfig(dynamics="momentum")
    return EnvConfig()


def make_env(num_envs: int, seed: int, training: bool, team: str = "default", device: str = "cuda"):
    """Build the task on the simulator named by the SWARM_SIM environment variable (2d or mjlab)."""
    n = clamp_envs(num_envs, training)
    if SIM == "mjlab":
        from swarm.env_mjlab import MjlabTransportEnv

        return MjlabTransportEnv(n, team=TEAMS[team], device=device, cfg=env_config(), seed=seed)
    if SIM not in ("2d", "2d_momentum"):
        raise ValueError(f"SWARM_SIM must be 2d, 2d_momentum, or mjlab, got {SIM}")
    return TransportEnv(n, team=TEAMS[team], device=device, cfg=env_config(), seed=seed)


def write_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"wrote {path}")
