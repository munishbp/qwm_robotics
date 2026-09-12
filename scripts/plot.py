"""Make the figures from the result files. docs/design.md section 9.

Usage: python scripts/plot.py --results results --figures figures
The script skips a figure whose input file is missing. It prints one line per figure.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3})

DPI = 150
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
MODES = ["independent", "leader", "round_robin"]
# Robot count per team name. scripts/common.py defines the teams.
TEAM_SIZE = {"p2g1s1": 4, "default": 6, "p4g4s1": 9, "p6g5s1": 12, "p8g7s1": 16}


def load_json(path: str):
    """Return the parsed file. Return None when the file is missing or broken."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        # A half written file must not stop the other figures.
        return None


def load_rows(path: str):
    """Return the rows of a sweep result file. Return None when the file gives no row."""
    data = load_json(path)
    if not isinstance(data, dict) or not data.get("rows"):
        return None
    return data["rows"]


def load_jsonl(path: str) -> list[dict]:
    """Return the parsed lines. A run in progress leaves a partial last line, so drop a bad line."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def skip(name: str, path: str) -> str:
    reason = "not found" if not os.path.exists(path) else "holds no usable row"
    return f"skip {name}: {path} {reason}"


def save(fig, figures_dir: str, name: str) -> str:
    path = os.path.join(figures_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return f"wrote {path}"


def xy(rows: list[dict], xkey: str, ykey: str):
    """Return the x values, the y values and the standard errors, sorted by x."""
    rows = sorted(rows, key=lambda r: r[xkey])
    return ([r[xkey] for r in rows], [r[ykey] for r in rows],
            [r.get(ykey + "_se", 0.0) for r in rows])


def fig_learning_curves(results_dir: str, figures_dir: str) -> str:
    name = "learning_curves.png"
    paths = sorted(p for p in glob.glob(os.path.join(results_dir, "train_*.jsonl"))
                   if "smoke" not in os.path.basename(p))
    if not paths:
        return f"skip {name}: no train_*.jsonl in {results_dir}"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    drawn = 0
    for i, path in enumerate(paths):
        rows = [r for r in load_jsonl(path) if "transitions" in r and "eval_success" in r]
        if not rows:
            continue
        rows.sort(key=lambda r: r["transitions"])
        run = os.path.basename(path)[len("train_"):-len(".jsonl")]
        color = COLORS[i % len(COLORS)]
        ax1.plot([r["transitions"] for r in rows], [r["eval_success"] for r in rows],
                 marker="o", ms=3, color=color, label=run)
        # train.py writes an empty metrics dict before the update starts, so a row can miss a loss.
        for key, style in (("wm_loss", "-"), ("dec_loss", "--")):
            pts = [(r["transitions"], r[key]) for r in rows if r.get(key, 0.0) > 0.0]
            if pts:
                ax2.plot([p[0] for p in pts], [p[1] for p in pts], style, color=color,
                         label=f"{run} {key}")
        drawn += 1
    if drawn == 0:
        plt.close(fig)
        return f"skip {name}: no rows in train_*.jsonl"
    ax1.set_xlabel("transitions")
    ax1.set_ylabel("eval success")
    ax1.set_title("Evaluation success")
    ax1.legend()
    ax2.set_xlabel("transitions")
    ax2.set_ylabel("loss")
    ax2.set_yscale("log")
    ax2.set_title("World model and decoder loss")
    ax2.legend(fontsize=8)
    return save(fig, figures_dir, name)


def fig_wm_error(results_dir: str, figures_dir: str) -> str:
    name = "wm_error.png"
    path = os.path.join(results_dir, "wm_error.json")
    data = load_json(path)
    horizons = data.get("horizons") if isinstance(data, dict) else None
    if not horizons:
        return skip(name, path)
    by_k = {int(k): v for k, v in horizons.items()}
    ks = sorted(by_k)

    def series(key):
        keep = [k for k in ks if key in by_k[k]]
        return keep, [by_k[k][key] for k in keep]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    panels = [
        (ax1, "latent_mse", "copy_latent_mse", "latent MSE", "Latent error"),
        (ax2, "decoded_pos_error_m", "copy_decoded_pos_error_m", "position error (m)",
         "Decoded position error"),
    ]
    for ax, model_key, copy_key, ylabel, title in panels:
        for key, label in ((model_key, "world model"), (copy_key, "copy baseline")):
            x, y = series(key)
            if x:
                ax.plot(x, y, marker="o", label=label)
        ax.set_xlabel("horizon (steps)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(ks)
        ax.legend()
    return save(fig, figures_dir, name)


def fig_h2_depth_vs_lag(results_dir: str, figures_dir: str) -> str:
    name = "h2_depth_vs_lag.png"
    path = os.path.join(results_dir, "h2.json")
    rows = load_rows(path)
    if not rows:
        return skip(name, path)
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, lag in enumerate(sorted({r["lag"] for r in rows})):
        color = COLORS[i % len(COLORS)]
        searched = [r for r in rows if r["lag"] == lag and r["depth"] >= 0]
        if searched:
            x, y, e = xy(searched, "depth", "success")
            ax.errorbar(x, y, yerr=e, marker="o", capsize=3, color=color, label=f"lag {lag}")
        plain = [r for r in rows if r["lag"] == lag and r["depth"] < 0]
        if plain:
            ax.axhline(plain[0]["success"], ls="--", lw=1, color=color,
                       label="no search" if i == 0 else "_nolegend_")
    ax.set_xlabel("search depth")
    ax.set_ylabel("success")
    ax.set_title("H2: search depth against message lag")
    ax.legend(fontsize=8)
    return save(fig, figures_dir, name)


def fig_h2_heatmap(results_dir: str, figures_dir: str) -> str:
    name = "h2_heatmap.png"
    path = os.path.join(results_dir, "h2.json")
    rows = load_rows(path)
    if not rows:
        return skip(name, path)
    lags = sorted({r["lag"] for r in rows})
    depths = sorted({r["depth"] for r in rows})
    grid = np.full((len(lags), len(depths)), np.nan)
    for r in rows:
        grid[lags.index(r["lag"]), depths.index(r["depth"])] = r["success"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(grid, cmap="viridis", aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(len(depths)), [str(d) for d in depths])
    ax.set_yticks(range(len(lags)), [str(v) for v in lags])
    ax.set_xlabel("search depth (-1 is no search)")
    ax.set_ylabel("message lag")
    ax.set_title("H2: success rate")
    fig.colorbar(im, ax=ax, label="success")
    for i in range(len(lags)):
        for j in range(len(depths)):
            v = grid[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                    color="black" if im.norm(v) > 0.5 else "white")
    return save(fig, figures_dir, name)


def fig_h2_best_depth(results_dir: str, figures_dir: str) -> str:
    name = "h2_best_depth.png"
    path = os.path.join(results_dir, "h2.json")
    rows = load_rows(path)
    if not rows:
        return skip(name, path)
    best = []
    for lag in sorted({r["lag"] for r in rows}):
        searched = sorted([r for r in rows if r["lag"] == lag and r["depth"] >= 0],
                          key=lambda r: r["depth"])
        if not searched:
            continue
        # The rows rise in depth, so a strict test gives the smaller depth on a tie.
        top = searched[0]
        for r in searched[1:]:
            if r["success"] > top["success"]:
                top = r
        best.append((lag, top["depth"], top["success"]))
    if not best:
        return f"skip {name}: no row with a search depth in {path}"
    fig, ax = plt.subplots(figsize=(6, 4))
    pos = range(len(best))
    ax.bar(pos, [b[1] for b in best], width=0.6, color=COLORS[0])
    for p, (_, depth, success) in zip(pos, best):
        ax.text(p, depth + 0.1, f"d={depth}\n{success:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(list(pos), [str(b[0]) for b in best])
    ax.set_ylim(0, max(b[1] for b in best) + 1.6)
    ax.set_xlabel("message lag")
    ax.set_ylabel("best search depth")
    ax.set_title("H2: best depth per lag")
    return save(fig, figures_dir, name)


def fig_h3_beta_vs_lag(results_dir: str, figures_dir: str) -> str:
    name = "h3_beta_vs_lag.png"
    path = os.path.join(results_dir, "h3.json")
    rows = load_rows(path)
    if not rows:
        return skip(name, path)
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, lag in enumerate(sorted({r["lag"] for r in rows})):
        sub = [r for r in rows if r["lag"] == lag]
        x, y, e = xy(sub, "beta", "success")
        ax.errorbar(x, y, yerr=e, marker="o", capsize=3, color=COLORS[i % len(COLORS)],
                    label=f"lag {lag}")
    ax.set_xlabel("beta")
    ax.set_ylabel("success")
    ax.set_title("H3: tree search discount against message lag")
    ax.legend(fontsize=8)
    return save(fig, figures_dir, name)


def fig_h4_leader(results_dir: str, figures_dir: str) -> str:
    name = "h4_leader.png"
    path = os.path.join(results_dir, "h4.json")
    rows = load_rows(path)
    if not rows:
        return skip(name, path)
    lags = sorted({r["lag"] for r in rows})
    present = {r["mode"] for r in rows}
    modes = [m for m in MODES if m in present] + sorted(present - set(MODES))
    cell = {(r["lag"], r["mode"]): r for r in rows}
    width = 0.8 / len(modes)
    base = np.arange(len(lags))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key, ylabel, title in ((ax1, "success", "success", "H4: leader mode"),
                                   (ax2, "ms_per_step", "ms per step", "H4: compute cost")):
        for j, mode in enumerate(modes):
            vals, errs = [], []
            for lag in lags:
                r = cell.get((lag, mode))
                vals.append(r[key] if r else np.nan)
                errs.append(r.get(key + "_se", 0.0) if r else 0.0)
            ax.bar(base + j * width, vals, width, yerr=errs, capsize=3, label=mode,
                   color=COLORS[j % len(COLORS)])
        ax.set_xticks(base + width * (len(modes) - 1) / 2, [str(v) for v in lags])
        ax.set_xlabel("message lag")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
    return save(fig, figures_dir, name)


def fig_robust_dropout(results_dir: str, figures_dir: str) -> str:
    name = "robust_dropout.png"
    path = os.path.join(results_dir, "robust.json")
    rows = load_rows(path)
    if not rows:
        return skip(name, path)
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (label, searched) in enumerate((("search", True), ("no search", False))):
        sub = [r for r in rows if (r["depth"] >= 0) == searched]
        if not sub:
            continue
        x, y, e = xy(sub, "dropout", "success")
        ax.errorbar(x, y, yerr=e, marker="o", capsize=3, color=COLORS[i % len(COLORS)], label=label)
    ax.set_xlabel("message dropout")
    ax.set_ylabel("success")
    ax.set_title("Robustness to message dropout")
    ax.legend()
    return save(fig, figures_dir, name)


def fig_transfer_team(results_dir: str, figures_dir: str) -> str:
    name = "transfer_team.png"
    path = os.path.join(results_dir, "transfer.json")
    rows = load_rows(path)
    if not rows:
        return skip(name, path)
    known = [r for r in rows if r.get("team") in TEAM_SIZE]
    if not known:
        return f"skip {name}: no known team name in {path}"
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (label, searched) in enumerate((("search", True), ("no search", False))):
        pts = sorted((TEAM_SIZE[r["team"]], r["success"], r.get("success_se", 0.0))
                     for r in known if (r["depth"] >= 0) == searched)
        if not pts:
            continue
        ax.errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=[p[2] for p in pts],
                    marker="o", capsize=3, color=COLORS[i % len(COLORS)], label=label)
    ax.set_xticks(sorted({TEAM_SIZE[r["team"]] for r in known}))
    ax.set_xlabel("team size K")
    ax.set_ylabel("success")
    ax.set_title("Transfer to a new team size")
    ax.legend()
    msg = save(fig, figures_dir, name)
    dropped = len(rows) - len(known)
    return msg + (f" ({dropped} rows dropped, unknown team name)" if dropped else "")


def fig_search_cost(results_dir: str, figures_dir: str) -> str:
    name = "search_cost.png"
    path = os.path.join(results_dir, "h2.json")
    rows = load_rows(path)
    if not rows:
        return skip(name, path)
    at_lag1 = [r for r in rows if r["lag"] == 1]
    searched = [r for r in at_lag1 if r["depth"] >= 0]
    if not searched:
        return f"skip {name}: no row at lag 1 with a search depth in {path}"
    fig, ax = plt.subplots(figsize=(6, 4))
    x, y, e = xy(searched, "depth", "ms_per_step")
    ax.errorbar(x, y, yerr=e, marker="o", capsize=3, color=COLORS[0], label="search")
    plain = [r for r in at_lag1 if r["depth"] < 0]
    if plain:
        ax.axhline(plain[0]["ms_per_step"], ls="--", lw=1, color=COLORS[1], label="no search")
    ax.set_xlabel("search depth")
    ax.set_ylabel("ms per step")
    ax.set_title("Search cost at lag 1")
    ax.legend()
    return save(fig, figures_dir, name)


FIGURES = [
    fig_learning_curves,
    fig_wm_error,
    fig_h2_depth_vs_lag,
    fig_h2_heatmap,
    fig_h2_best_depth,
    fig_h3_beta_vs_lag,
    fig_h4_leader,
    fig_robust_dropout,
    fig_transfer_team,
    fig_search_cost,
]


def main() -> None:
    p = argparse.ArgumentParser(description="Make the figures from the result files.")
    p.add_argument("--results", default="results")
    p.add_argument("--figures", default="figures")
    args = p.parse_args()
    if not os.path.isdir(args.results):
        sys.exit(f"error: the results directory {args.results} does not exist. "
                 f"Run the experiment scripts first, or pass --results.")
    os.makedirs(args.figures, exist_ok=True)
    for make in FIGURES:
        print(make(args.results, args.figures))


if __name__ == "__main__":
    main()
