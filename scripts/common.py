"""Shared argument handling for the scripts."""

from __future__ import annotations

import json
import os
import random

import numpy as np
import torch

from swarm.compute import clamp_envs, limit_memory
from swarm.env import DEFAULT_TEAM, EnvConfig, TransportEnv

TEAMS = {
    "default": DEFAULT_TEAM,
    "p2g1s1": ("pusher", "pusher", "gripper", "scout"),
    "p4g4s1": ("pusher",) * 4 + ("gripper",) * 4 + ("scout",),
    "p6g5s1": ("pusher",) * 6 + ("gripper",) * 5 + ("scout",),
    "p8g7s1": ("pusher",) * 8 + ("gripper",) * 7 + ("scout",),
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
