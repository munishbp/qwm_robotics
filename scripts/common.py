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


def make_env(num_envs: int, seed: int, training: bool, team: str = "default", device: str = "cuda") -> TransportEnv:
    return TransportEnv(clamp_envs(num_envs, training), team=TEAMS[team], device=device, cfg=EnvConfig(), seed=seed)


def write_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"wrote {path}")
