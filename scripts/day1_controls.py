"""The first day controls of docs/next_steps.md, on one snapshot per simulator.

Usage: SWARM_SIM=2d python scripts/day1_controls.py --ckpt checkpoints/belief_best.pt --out results/day1_controls.json
Cells: forward correction off across lags (2D question), random world model against the trained
one, the decoded state scorer and the random scorer as controls on the search machinery, the
leader mode with one fresh election and no fallback, and on mjlab the re baselined H1 against
the sampled policy at 256 envs and 5 batches.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from common import SIM, make_env, setup, write_json  # noqa: E402

from swarm.belief import BeliefConfig  # noqa: E402
from swarm.evaluate import evaluate  # noqa: E402
from swarm.nets import WorldModel  # noqa: E402
from swarm.rlpd import Agent  # noqa: E402
from swarm.search import SearchConfig  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/belief_best.pt")
    p.add_argument("--out", default="results/day1_controls.json")
    p.add_argument("--seed", type=int, default=1000)
    args = p.parse_args()
    dev = setup(args.seed)
    agent = Agent.load(args.ckpt, dev)
    trained_wm = agent.nets.wm
    rows = []

    def cell(name, policy, bcfg, scfg=None, envs=128, batches=3, **tags):
        t0 = time.time()
        res = evaluate(agent, lambda s: make_env(envs, s, False, "default", dev), bcfg, policy, scfg, batches, args.seed)
        res.update({"name": name, "sim": SIM, "envs": envs, "batches": batches, **tags})
        rows.append(res)
        print(f"{name}: success={res['success']:.3f}+-{res['success_se']:.3f} ({time.time() - t0:.0f}s)", flush=True)

    # F5a: forward correction off (the 2D staleness rise question), both simulators for symmetry.
    for lag in [0, 1, 2, 4]:
        cell(f"no search, no forward correction, lag {lag}", "mean", BeliefConfig(lag=lag, roll_messages=False), group="F5a", lag=lag)
        cell(f"no search, forward correction, lag {lag}", "mean", BeliefConfig(lag=lag), group="F5a", lag=lag)
    # F3: random world model against the trained one, lag 1.
    for depth in [2, 6]:
        cell(f"trained world model, depth {depth}", "search", BeliefConfig(lag=1), SearchConfig(depth=depth), group="F3", depth=depth, wm="trained")
        agent.nets.wm = WorldModel().to(dev)
        cell(f"random world model, depth {depth}", "search", BeliefConfig(lag=1), SearchConfig(depth=depth), group="F3", depth=depth, wm="random")
        agent.nets.wm = trained_wm
    # F2: scorer controls at lag 1.
    for depth in [0, 2]:
        cell(f"decoded scorer, depth {depth}", "search", BeliefConfig(lag=1), SearchConfig(depth=depth, scorer="decoded"), group="F2", depth=depth, scorer="decoded")
    cell("random scorer, depth 0", "search", BeliefConfig(lag=1), SearchConfig(depth=0, scorer="random"), group="F2", depth=0, scorer="random")
    cell("critic scorer, depth 0", "search", BeliefConfig(lag=1), SearchConfig(depth=0), group="F2", depth=0, scorer="q")
    cell("sampled policy", "sample", BeliefConfig(lag=1), group="F2")
    # F7: leader with one fresh election and no fallback.
    for lag in [1, 4]:
        cell(f"independent, lag {lag}", "search", BeliefConfig(lag=lag), SearchConfig(depth=2), group="F7", lag=lag, mode="independent")
        cell(f"leader fresh election no fallback, lag {lag}", "search", BeliefConfig(lag=lag), SearchConfig(depth=2, mode="leader_fresh"), group="F7", lag=lag, mode="leader_fresh")
    # F1: the re baselined H1 on mjlab at 256 envs and 5 batches.
    if SIM == "mjlab":
        for lag in [0, 1, 4]:
            cell(f"H1 sampled policy, lag {lag}", "sample", BeliefConfig(lag=lag), envs=256, batches=5, group="F1", lag=lag, arm="sampled")
            cell(f"H1 random candidate, lag {lag}", "search", BeliefConfig(lag=lag), SearchConfig(depth=0, scorer="random"), envs=256, batches=5, group="F1", lag=lag, arm="random_candidate")
            cell(f"H1 depth 0, lag {lag}", "search", BeliefConfig(lag=lag), SearchConfig(depth=0), envs=256, batches=5, group="F1", lag=lag, arm="depth0")
            cell(f"H1 depth 6 beta 0.9, lag {lag}", "search", BeliefConfig(lag=lag), SearchConfig(depth=6, beta=0.9), envs=256, batches=5, group="F1", lag=lag, arm="depth6")
    write_json(args.out, {"rows": rows, "args": vars(args), "sim": SIM})


if __name__ == "__main__":
    main()
