"""Print markdown tables and hypothesis verdicts from the result files.

Usage: python scripts/summarize.py [--results results] > docs/results_tables.md
The decision rules are those of docs/methodology.md section 7.
"""

from __future__ import annotations

import argparse
import json
import os


def load(path: str):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def pct(x: float, se: float | None = None) -> str:
    return f"{100 * x:.1f}" + (f" ± {100 * se:.1f}" if se is not None else "")


def controls(d: dict) -> None:
    print("### Controls (256 envs)\n")
    print("| Control | Success (%) | Length on success | Position error (m) | Angle error (rad) |")
    print("|---|---|---|---|---|")
    for k, v in d.items():
        if not isinstance(v, dict) or "success_rate" not in v:
            continue
        ln = v.get("mean_success_length")
        print(f"| {k} | {pct(v['success_rate'])} | {ln:.1f} | {v['mean_pos_error']:.3f} | {v['mean_angle_error']:.3f} |"
              if ln is not None else
              f"| {k} | {pct(v['success_rate'])} | - | {v['mean_pos_error']:.3f} | {v['mean_angle_error']:.3f} |")
    t = d.get("throughput", {})
    if t:
        print(f"\nThroughput: {t.get('env_steps_per_second', 0):,.0f} env steps per second at {t.get('num_envs')} envs.")
    print()


def training(results: str) -> None:
    print("### Training\n")
    print("| Run | Transitions | Final eval success | Best eval success | Wall minutes |")
    print("|---|---|---|---|---|")
    for name in sorted(os.listdir(results)):
        if not name.startswith("train_") or not name.endswith(".jsonl") or "smoke" in name:
            continue
        rows = []
        for line in open(os.path.join(results, name)):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        if not rows:
            continue
        last = rows[-1]
        best = max(r["eval_success"] for r in rows)
        print(f"| {name[6:-6]} | {last['transitions']:,} | {pct(last['eval_success'])} | {pct(best)} | {last['wall_min']:.0f} |")
    print()


def wm_error(d: dict) -> None:
    print("### World model open loop error\n")
    print("| Horizon | Latent MSE | Copy latent MSE | Decoded position error (m) | Copy position error (m) | Decoded angle error (rad) |")
    print("|---|---|---|---|---|---|")
    for k, v in d["horizons"].items():
        print(f"| {k} | {v['latent_mse']:.4f} | {v['copy_latent_mse']:.4f} | {v['decoded_pos_error_m']:.3f} | {v['copy_decoded_pos_error_m']:.3f} | {v['decoded_angle_error_rad']:.3f} |")
    print()


def h1(results: str) -> None:
    rows = {}
    for d in (-1, 0, 2):
        r = load(os.path.join(results, f"h1_depth{d}.json"))
        if r:
            rows[d] = r
    if not rows:
        return
    print("### H1: search against no search at lag 1 (256 envs, 3 batches)\n")
    print("| Setting | Success (%) | Length on success | ms per step |")
    print("|---|---|---|---|")
    names = {-1: "no search (mean action)", 0: "depth 0 (critic argmax)", 2: "depth 2 search"}
    for d, r in rows.items():
        print(f"| {names[d]} | {pct(r['success'], r['success_se'])} | {r['length_on_success']:.1f} | {r['ms_per_step']:.1f} |")
    if -1 in rows and 2 in rows:
        a, b = rows[-1], rows[2]
        gap = b["success"] - a["success"]
        se = (a["success_se"] ** 2 + b["success_se"] ** 2) ** 0.5
        verdict = "confirmed" if gap > 2 * se else ("refuted (search is worse)" if gap < -2 * se else "not resolved (within noise)")
        print(f"\nH1 verdict: depth 2 minus no search = {100 * gap:+.1f} points, 2 SE = {200 * se:.1f} points, **{verdict}**.\n")


def grid(rows: list[dict], row_key: str, col_key: str) -> tuple[list, list, dict]:
    rs = sorted({r[row_key] for r in rows})
    cs = sorted({r[col_key] for r in rows})
    cell = {(r[row_key], r[col_key]): r for r in rows}
    return rs, cs, cell


def h2(d: dict) -> None:
    rows = d["rows"]
    lags, depths, cell = grid(rows, "lag", "depth")
    print("### H2: success (%) by staleness (rows) and depth (columns), beta 0.5\n")
    print("| Lag | " + " | ".join("no search" if c == -1 else f"D={c}" for c in depths) + " | best depth |")
    print("|---|" + "---|" * (len(depths) + 1))
    best_by_lag = {}
    for lag in lags:
        cells = []
        best_d, best_s = None, -1
        for c in depths:
            r = cell.get((lag, c))
            cells.append(pct(r["success"], r["success_se"]) if r else "-")
            if r and c >= 0 and r["success"] > best_s:
                best_d, best_s = c, r["success"]
        best_by_lag[lag] = (best_d, best_s)
        print(f"| {lag} | " + " | ".join(cells) + f" | {best_d} |")
    seq = [best_by_lag[l][0] for l in lags]
    mono = all(seq[i] >= seq[i + 1] for i in range(len(seq) - 1))
    drop = seq[0] is not None and seq[-1] is not None and seq[0] > seq[-1]
    # Note the noise level: the spread of the best cell against its neighbors.
    verdict = "confirmed" if mono and drop else ("refuted" if seq[0] is not None and seq[-1] is not None and seq[-1] > seq[0] else "not resolved")
    print(f"\nBest depth per lag: {dict(zip(lags, seq))}. H2 verdict: **{verdict}** (non increasing: {mono}, drop from lag {lags[0]} to {lags[-1]}: {drop}).\n")
    gaps = {lag: cell[(lag, 1)]["success"] - cell[(lag, 0)]["success"] for lag in lags if (lag, 1) in cell and (lag, 0) in cell}
    print("Cost of one step of imagination (depth 1 minus depth 0, points): "
          + ", ".join(f"lag {l}: {100 * g:+.1f}" for l, g in gaps.items()) + ".\n")
    below = {lag: cell[(lag, best_by_lag[lag][0])]["success"] - cell[(lag, -1)]["success"] for lag in lags if (lag, -1) in cell and best_by_lag[lag][0] is not None}
    print("Best search cell minus no search (points): "
          + ", ".join(f"lag {l}: {100 * g:+.1f}" for l, g in below.items()) + ".\n")
    print("Cost: ms per step by depth at lag 1: " + ", ".join(f"D={c}: {cell[(1, c)]['ms_per_step']:.0f}" for c in depths if (1, c) in cell) + ".\n")


def h3(d: dict) -> None:
    rows = d["rows"]
    lags, betas, cell = grid(rows, "lag", "beta")
    depth = rows[0]["depth"]
    print(f"### H3: success (%) by staleness (rows) and tree search discount beta (columns), depth {depth}\n")
    print("| Lag | " + " | ".join(f"beta={c}" for c in betas) + " | best beta |")
    print("|---|" + "---|" * (len(betas) + 1))
    seq = []
    for lag in lags:
        cells, best_b, best_s = [], None, -1
        for c in betas:
            r = cell.get((lag, c))
            cells.append(pct(r["success"], r["success_se"]) if r else "-")
            if r and r["success"] > best_s:
                best_b, best_s = c, r["success"]
        seq.append(best_b)
        print(f"| {lag} | " + " | ".join(cells) + f" | {best_b} |")
    mono = all(seq[i] >= seq[i + 1] for i in range(len(seq) - 1))
    print(f"\nBest beta per lag: {dict(zip(lags, seq))}. H3 verdict: **{'confirmed' if mono and seq[0] > seq[-1] else ('refuted' if seq[-1] > seq[0] else 'not resolved')}** (non increasing: {mono}).\n")


def h4(d: dict) -> None:
    rows = d["rows"]
    lags, modes, cell = grid(rows, "lag", "mode")
    print(f"### H4: leader election at matched compute, depth {rows[0]['depth']}\n")
    print("| Lag | Mode | Success (%) | ms per step | Searches per env | Robots following | Leader disagreement |")
    print("|---|---|---|---|---|---|---|")
    verdict = "not resolved"
    for lag in lags:
        for m in ["independent", "leader", "round_robin"]:
            r = cell.get((lag, m))
            if not r:
                continue
            print(f"| {lag} | {m} | {pct(r['success'], r['success_se'])} | {r['ms_per_step']:.0f} | {r.get('searches_per_env', 6 if m == 'independent' else 1):.2f} | {r.get('robots_following_a_leader', 1.0):.2f} | {r.get('leader_disagreement', 0.0):.2f} |")
        a, b = cell.get((lag, "independent")), cell.get((lag, "leader"))
        if a and b:
            gap = b["success"] - a["success"]
            se = (a["success_se"] ** 2 + b["success_se"] ** 2) ** 0.5
            if gap > 2 * se:
                verdict = "confirmed"
            elif gap < -2 * se and verdict != "confirmed":
                verdict = "refuted (leader is worse)"
    print(f"\nH4 verdict: **{verdict}**.\n")


def simple(d: dict, key: str, title: str) -> None:
    rows = d["rows"]
    print(f"### {title}\n")
    print(f"| {key} | no search (%) | depth {max(r['depth'] for r in rows)} search (%) |")
    print("|---|---|---|")
    vals = sorted({r[key] for r in rows}, key=lambda x: (str(type(x)), x))
    for v in vals:
        ns = next((r for r in rows if r[key] == v and r["depth"] == -1), None)
        s = next((r for r in rows if r[key] == v and r["depth"] >= 0), None)
        print(f"| {v} | {pct(ns['success'], ns['success_se']) if ns else '-'} | {pct(s['success'], s['success_se']) if s else '-'} |")
    print()


def ablations(d: dict) -> None:
    print("### Ablations on the snapshot (128 envs, 3 batches)\n")
    print("| Setting | Success (%) | Length on success |")
    print("|---|---|---|")
    for r in d["rows"]:
        print(f"| {r['name']} | {pct(r['success'], r['success_se'])} | {r['length_on_success']:.1f} |")
    print()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    args = p.parse_args()
    R = args.results
    if (d := load(os.path.join(R, "controls.json"))):
        controls(d)
    training(R)
    if (d := load(os.path.join(R, "wm_error.json"))):
        wm_error(d)
    h1(R)
    if (d := load(os.path.join(R, "h2.json"))):
        h2(d)
    if (d := load(os.path.join(R, "h3.json"))):
        h3(d)
    if (d := load(os.path.join(R, "h4.json"))):
        h4(d)
    if (d := load(os.path.join(R, "robust.json"))):
        simple(d, "dropout", "Robustness: message dropout at lag 1")
    if (d := load(os.path.join(R, "transfer.json"))):
        simple(d, "team", "Transfer: team composition at lag 1 (trained on the default team)")
    if (d := load(os.path.join(R, "ablations.json"))):
        ablations(d)


if __name__ == "__main__":
    main()
