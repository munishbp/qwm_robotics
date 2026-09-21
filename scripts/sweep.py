"""Test time sweeps on one snapshot. docs/design.md section 9.

Usage: python scripts/sweep.py --which h2 h3 h4 robust transfer gate lcb [--smoke]
Every cell is an evaluation of the same checkpoint at a different search or channel setting.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from common import make_env, setup, write_json  # noqa: E402
from evaluate import add_eval_args  # noqa: E402

from swarm.belief import BeliefConfig  # noqa: E402
from swarm.evaluate import evaluate  # noqa: E402
from swarm.rlpd import Agent  # noqa: E402
from swarm.search import SearchConfig  # noqa: E402

DEPTHS = [-1, 0, 1, 2, 4, 6]
LAGS = [0, 1, 2, 4]
BETAS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
DROPOUTS = [0.0, 0.1, 0.25, 0.5]
TEAMS = ["p2g1s1", "default", "p4g4s1", "p6g5s1", "p8g7s1"]
GATES = [0.25, 0.5, 1.0, 2.0, 4.0]


def cell(agent, args, dev, lag, depth, beta=0.5, mode="independent", dropout=0.0, team="default",
         gate=0.0, fallback="sample", shuffle=False, policy="mean", lcb=None) -> dict:
    """`policy` is the no search policy of a depth -1 cell: mean or sample. `lcb` defaults to --lcb."""
    lcb = args.lcb if lcb is None else lcb
    scfg = SearchConfig(depth=depth, beta=beta, mode=mode, candidates=args.candidates, beam=args.beam,
                        gate=gate, gate_fallback=fallback, gate_shuffle=shuffle, lcb=lcb)
    policy = "search" if depth >= 0 else policy
    t0 = time.time()
    res = evaluate(agent, lambda s: make_env(args.envs, s, False, team, dev), BeliefConfig(lag, dropout),
                   policy, scfg if depth >= 0 else None, args.batches, args.seed)
    res.update({"lag": lag, "depth": depth, "beta": beta, "mode": mode, "dropout": dropout, "team": team,
                "gate": gate, "fallback": fallback, "shuffle": shuffle, "policy": policy, "lcb": lcb})
    print(f"lag={lag} depth={depth} policy={policy} beta={beta} mode={mode} drop={dropout} team={team} "
          f"lcb={lcb} gate={gate} fallback={fallback} shuffle={shuffle} searched={res.get('searched_fraction', 1.0):.2f}: "
          f"success={res['success']:.3f}+-{res['success_se']:.3f} ms/step={res['ms_per_step']:.1f} ({time.time()-t0:.0f}s)")
    return res


def main() -> None:
    p = argparse.ArgumentParser()
    add_eval_args(p)
    p.add_argument("--which", nargs="+", default=["h2", "h3", "h4", "robust", "transfer"])
    p.add_argument("--candidates", type=int, default=8)
    p.add_argument("--beam", type=int, default=4)
    p.add_argument("--h3-depth", type=int, default=2)
    p.add_argument("--h4-depth", type=int, default=2)
    p.add_argument("--gates", type=float, nargs="+", default=GATES, help="gate grid of --which gate")
    p.add_argument("--lcb", type=float, default=2.0, help="pessimistic score of every group but lcb; 0 reproduces results.md sections 3 to 20")
    p.add_argument("--lcbs", type=float, nargs="+", default=[0.5, 1.0, 2.0], help="lcb grid of --which lcb")
    p.add_argument("--tag", default="", help="suffix of the output file names, so a second grid keeps the first")
    p.add_argument("--smoke", action="store_true")
    # 128 envs per batch keeps the 76 cells of the sweeps under two hours.
    p.set_defaults(envs=128)
    args = p.parse_args()
    if args.smoke:
        args.envs, args.batches = 16, 1
    dev = setup(args.seed)
    agent = Agent.load(args.ckpt, dev)
    tag = "_smoke" if args.smoke else args.tag
    if "h2" in args.which:
        rows = [cell(agent, args, dev, lag, d) for lag in LAGS for d in DEPTHS]
        write_json(f"results/h2{tag}.json", {"rows": rows, "args": vars(args)})
    if "h3" in args.which:
        rows = [cell(agent, args, dev, lag, args.h3_depth, beta=b) for lag in LAGS for b in BETAS]
        write_json(f"results/h3{tag}.json", {"rows": rows, "args": vars(args)})
    if "h4" in args.which:
        rows = [cell(agent, args, dev, lag, args.h4_depth, mode=m)
                for lag in [1, 4] for m in ["independent", "leader", "round_robin"]]
        write_json(f"results/h4{tag}.json", {"rows": rows, "args": vars(args)})
    if "robust" in args.which:
        rows = [cell(agent, args, dev, 1, d, dropout=q) for q in DROPOUTS for d in [-1, args.h4_depth]]
        write_json(f"results/robust{tag}.json", {"rows": rows, "args": vars(args)})
    if "transfer" in args.which:
        rows = [cell(agent, args, dev, 1, d, team=tm) for tm in TEAMS for d in [-1, args.h4_depth]]
        write_json(f"results/transfer{tag}.json", {"rows": rows, "args": vars(args)})
    if "gate" in args.which:
        # The two no search baselines, the ungated search, the gate grid, and the shuffled control.
        rows = [cell(agent, args, dev, 1, -1, policy=pol) for pol in ["mean", "sample"]]
        for d in [0, args.h4_depth]:
            rows.append(cell(agent, args, dev, 1, d))
            rows += [cell(agent, args, dev, 1, d, gate=g, fallback=fb) for fb in ["sample", "mean"] for g in args.gates]
            rows += [cell(agent, args, dev, 1, d, gate=g, fallback="sample", shuffle=True)
                     for g in args.gates if g in (0.5, 1.0, 2.0)]
        write_json(f"results/gate{tag}.json", {"rows": rows, "args": vars(args)})
    if "lcb" in args.which:
        # The pessimistic score alone, then with the real gate and the shuffled gate. lcb 0 is in gate.json.
        rows = []
        for lcb in args.lcbs:
            for d in [0, args.h4_depth]:
                rows.append(cell(agent, args, dev, 1, d, lcb=lcb))
                rows += [cell(agent, args, dev, 1, d, gate=g, shuffle=sh, lcb=lcb) for g in args.gates for sh in [False, True]]
        write_json(f"results/lcb{tag}.json", {"rows": rows, "args": vars(args)})


if __name__ == "__main__":
    main()
