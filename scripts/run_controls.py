"""Run the four environment controls and write results/controls.json.

The controls check that the task needs the whole team. The scripted controller has to succeed often
with the full team, and a single robot of one type has to fail every time. See docs/design.md
section 9.

Usage:
    python scripts/run_controls.py            # 256 envs
    python scripts/run_controls.py --smoke    # 16 envs
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

from swarm import compute

compute.limit_memory()

import torch  # noqa: E402  the memory cap has to come first

from swarm.env import DEFAULT_TEAM, TYPE_GRIPPER, TYPE_PUSHER, TransportEnv  # noqa: E402
from swarm.scripted import ScriptedController  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results" / "controls.json"
WARMUP_STEPS = 20
TIMED_STEPS = 300


def rollout(team: tuple[int, ...], num_envs: int, device: str, seed: int) -> dict[str, float]:
    """Run one episode per env under the scripted controller and report the first outcome.

    The env auto resets, so the run keeps a done mask and records each env only once. A later
    episode in the same env would bias the mean toward the short episodes.
    """
    env = TransportEnv(num_envs, team=team, device=device, seed=seed)
    control = ScriptedController(env)
    done = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    success = torch.zeros_like(done)
    length = torch.zeros(num_envs, device=env.device)
    pos_error = torch.zeros(num_envs, device=env.device)
    angle_error = torch.zeros(num_envs, device=env.device)

    for step in range(env.cfg.horizon):
        _, _, terminated, truncated, info = env.step(control.act())
        ending = (terminated | truncated) & ~done
        success |= terminated & ~done
        length = torch.where(ending, torch.full_like(length, step + 1.0), length)
        pos_error = torch.where(ending, info["final_pos_error"], pos_error)
        angle_error = torch.where(ending, info["final_angle_error"], angle_error)
        done |= terminated | truncated

    won = success.float().mean().item()
    return {
        "num_envs": num_envs,
        "success_rate": won,
        "mean_success_length": length[success].mean().item() if won > 0.0 else None,
        "mean_pos_error": pos_error.mean().item(),
        "mean_angle_error": angle_error.mean().item(),
    }


def throughput(num_envs: int, device: str, seed: int) -> dict[str, float]:
    """Measure env steps per second with random actions after a warmup."""
    env = TransportEnv(num_envs, team=DEFAULT_TEAM, device=device, seed=seed)
    shape = (num_envs, env.num_robots, 3)
    for _ in range(WARMUP_STEPS):
        env.step(torch.rand(shape, device=env.device) * 2.0 - 1.0)
    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(TIMED_STEPS):
        env.step(torch.rand(shape, device=env.device) * 2.0 - 1.0)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return {
        "num_envs": num_envs,
        "steps": TIMED_STEPS,
        "seconds": elapsed,
        "batched_steps_per_second": TIMED_STEPS / elapsed,
        "env_steps_per_second": TIMED_STEPS * num_envs / elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="run 16 envs instead of 256")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    num_envs = compute.clamp_envs(16 if args.smoke else 256, training=False)
    report = {
        "num_envs": num_envs,
        "device": args.device,
        "seed": args.seed,
        "scripted_full_team": rollout(DEFAULT_TEAM, num_envs, args.device, args.seed),
        "single_pusher": rollout((TYPE_PUSHER,), num_envs, args.device, args.seed),
        "single_gripper": rollout((TYPE_GRIPPER,), num_envs, args.device, args.seed),
        "throughput": throughput(num_envs, args.device, args.seed),
    }

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(report, indent=2) + "\n")

    print(f"controls at {num_envs} envs on {args.device}, seed {args.seed}")
    print(f"{'control':<22}{'success':>9}{'len':>8}{'pos err':>9}{'ang err':>9}")
    for name in ("scripted_full_team", "single_pusher", "single_gripper"):
        row = report[name]
        length = "-" if row["mean_success_length"] is None else f"{row['mean_success_length']:.1f}"
        print(
            f"{name:<22}{row['success_rate']:>9.3f}{length:>8}"
            f"{row['mean_pos_error']:>9.3f}{row['mean_angle_error']:>9.3f}"
        )
    rate = report["throughput"]
    print(f"throughput            {rate['env_steps_per_second']:>9.0f} env steps per second")
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    main()
