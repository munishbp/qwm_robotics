"""Make the paper figures from the result files.

Usage: python scripts/plot_paper.py --root . --figures figures/paper
The pipeline writes the result files while it runs. The script skips a figure whose input is
missing and prints one line for each figure.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3})

DPI = 200
ARMS = ("sampled", "depth6")
ARM_LABEL = {"sampled": "sampled policy", "depth6": "depth 6"}
ARM_COLOR = {"sampled": "tab:blue", "depth6": "tab:orange"}

# One H1 run per entry. The last entry is the agent that uses search during collection.
SEED_DIRS = [
    ("mjlab", "runs/mjlab"),
    ("mjlab_seed1", "runs/mjlab_seed1"),
    ("mjlab_seed2", "runs/mjlab_seed2"),
    ("search in collection", "runs/mjlab_collect"),
]
SWEEP_GROUPS = ("h2", "h3", "h4", "robust", "transfer")

# Every sweep group repeats this cell. The figure compares the repeats.
REPEATED_CELL = {
    "depth": 2,
    "lag": 1,
    "beta": 0.5,
    "mode": "independent",
    "dropout": 0.0,
    "team": "default",
}


def load_json(path: str):
    """Return the parsed file. Return None when the file is missing or broken."""
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        # The pipeline can leave a half written file. One bad file must not stop the other figures.
        return None


def load_rows(path: str):
    """Return the rows of a result file. Return None when the file gives no row."""
    data = load_json(path)
    if not isinstance(data, dict):
        return None
    rows = data.get("rows")
    return rows if isinstance(rows, list) and rows else None


def number(value):
    """Return the value as a float. Return None when the value is not a number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def point(row: dict):
    """Return the success and its standard error. Return None when the row has no success."""
    success = number(row.get("success"))
    if success is None:
        return None
    error = number(row.get("success_se"))
    return success, 0.0 if error is None else error


def run_label(root: str, path: str) -> str:
    """Return the run name of a result file path."""
    parts = os.path.relpath(path, root).split(os.sep)
    if parts[0] == "runs" and len(parts) > 2:
        return parts[1]
    return parts[0]


def save(fig, out_dir: str, name: str) -> None:
    """Write one figure and print the path."""
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def add_arm_point(points: dict, row: dict) -> None:
    """Store the row when it holds one arm of interest at one lag."""
    arm = row.get("arm")
    lag = number(row.get("lag"))
    value = point(row)
    if arm in ARMS and lag is not None and value is not None:
        points[(arm, int(lag))] = value


def load_arm_points(root: str, run_dir: str) -> dict:
    """Return {(arm, lag): (success, error)} for the H1 arms of one run directory."""
    points: dict = {}
    # The rerun with per episode outcomes replaces the first file where it exists.
    f1 = os.path.join(root, run_dir, "results", "day1_controls_f1.json")
    first = f1 if os.path.exists(f1) else os.path.join(root, run_dir, "results", "day1_controls.json")
    for row in load_rows(first) or []:
        if row.get("group") == "F1":
            add_arm_point(points, row)
    for row in load_rows(os.path.join(root, run_dir, "results", "lag_curve.json")) or []:
        add_arm_point(points, row)
    return points


def fig_h1_by_seed(seeds: dict, out_dir: str) -> None:
    """Draw the sampled policy against depth 6 per seed. One panel shows one lag."""
    lags = sorted({lag for points in seeds.values() for _, lag in points})
    if not lags:
        print("skip fair_h1_by_seed.png: no day1_controls.json or lag_curve.json row for the H1 arms")
        return
    labels = list(seeds)
    positions = np.arange(len(labels))
    width = 0.36
    columns = min(len(lags), 3)
    rows = (len(lags) + columns - 1) // columns
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(0.95 * len(labels) * columns + 2.5, 4.6 * rows),
        sharey=True,
        squeeze=False,
    )
    panels = axes.ravel()
    for axis, lag in zip(panels, lags):
        for index, arm in enumerate(ARMS):
            values = [seeds[label].get((arm, lag)) for label in labels]
            heights = np.array([v[0] if v else np.nan for v in values])
            errors = np.array([v[1] if v else np.nan for v in values])
            axis.bar(
                positions + (index - 0.5) * width,
                heights,
                width,
                yerr=errors,
                capsize=3,
                color=ARM_COLOR[arm],
                label=ARM_LABEL[arm],
            )
        annotate_pairs(axis, seeds, labels, positions, lag)
        axis.set_title(f"lag {lag}")
        axis.set_xlabel("run")
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=20, ha="right")
        axis.set_ylim(0.0, 1.18)
    for axis in panels[len(lags):]:
        axis.axis("off")
    for index in range(0, len(panels), columns):
        panels[index].set_ylabel("success rate")
    panels[0].legend(loc="upper left")
    fig.suptitle("H1 under a fair budget: sampled policy against depth 6 search, per run")
    save(fig, out_dir, "fair_h1_by_seed.png")


def annotate_pairs(axis, seeds: dict, labels: list, positions, lag: int) -> None:
    """Write the depth 6 minus sampled difference above each pair of bars."""
    for label, position in zip(labels, positions):
        low = seeds[label].get(("sampled", lag))
        high = seeds[label].get(("depth6", lag))
        if low is None or high is None:
            continue
        top = max(low[0] + low[1], high[0] + high[1])
        axis.text(
            position,
            top + 0.03,
            f"{high[0] - low[0]:+.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def fig_gain_vs_staleness(seeds: dict, out_dir: str) -> None:
    """Draw the depth 6 minus sampled difference against lag, with the mean across runs."""
    gains = {}
    for label, points in seeds.items():
        lags = sorted({lag for arm, lag in points if arm == "sampled"} & {lag for arm, lag in points if arm == "depth6"})
        if lags:
            gains[label] = {lag: points[("depth6", lag)][0] - points[("sampled", lag)][0] for lag in lags}
    if not gains:
        print("skip gain_vs_staleness.png: no run holds both the sampled policy and depth 6 at one lag")
        return
    lags = sorted({lag for series in gains.values() for lag in series})
    table = np.array([[series.get(lag, np.nan) for lag in lags] for series in gains.values()])
    fig, axis = plt.subplots(figsize=(7.5, 5.0))
    axis.axhline(0.0, color="black", linewidth=1.0)
    for label, series in gains.items():
        axis.plot(list(series), list(series.values()), linewidth=1.0, marker="o", markersize=4, alpha=0.75, label=label)
    mean = np.nanmean(table, axis=0)
    spread = np.nanstd(table, axis=0)
    axis.fill_between(lags, mean - spread, mean + spread, color="gray", alpha=0.25, label="mean +/- SD")
    axis.plot(lags, mean, color="black", linewidth=2.8, marker="s", label="mean across runs")
    axis.set_title("Gain of depth 6 search over the sampled policy against staleness")
    axis.set_xlabel("lag (steps)")
    axis.set_ylabel("success rate difference")
    axis.set_xticks(lags)
    axis.legend()
    save(fig, out_dir, "gain_vs_staleness.png")


def gain_over_no_search(rows: list) -> dict:
    """Return {lag: best search success minus no search success} for one h2 sweep."""
    cells: dict = {}
    for row in rows:
        lag = number(row.get("lag"))
        depth = number(row.get("depth"))
        value = point(row)
        if lag is None or depth is None or value is None:
            continue
        cell = cells.setdefault(int(lag), {"none": None, "best": None})
        if depth < 0:
            cell["none"] = value[0]
        elif cell["best"] is None or value[0] > cell["best"]:
            cell["best"] = value[0]
    return {
        lag: cell["best"] - cell["none"]
        for lag, cell in cells.items()
        if cell["none"] is not None and cell["best"] is not None
    }


def fig_gain_over_no_search(root: str, out_dir: str) -> None:
    """Draw the gain of the best search cell over the no search cell, one line per run."""
    paths = sorted(glob.glob(os.path.join(root, "runs", "*", "results", "h2.json")))
    if os.path.exists(os.path.join(root, "results", "h2.json")):
        paths.insert(0, os.path.join(root, "results", "h2.json"))
    panels = {"2D": {}, "mjlab": {}}
    for path in paths:
        rows = load_rows(path)
        if rows is None:
            continue
        label = run_label(root, path)
        gains = gain_over_no_search(rows)
        if gains:
            panels["mjlab" if "mjlab" in label else "2D"][label] = gains
    if not panels["2D"] and not panels["mjlab"]:
        print("skip gain_over_no_search_by_seed.png: no h2.json holds both a search cell and a no search cell")
        return
    ticks = sorted({lag for runs in panels.values() for gains in runs.values() for lag in gains})
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), sharey=True)
    for axis, (sim, runs) in zip(axes, panels.items()):
        axis.axhline(0.0, color="black", linewidth=1.0)
        for label, gains in sorted(runs.items()):
            lags = sorted(gains)
            axis.plot(lags, [gains[lag] for lag in lags], marker="o", label=label)
        axis.set_xticks(ticks)
        if not runs:
            axis.text(0.5, 0.5, "no run found", ha="center", va="center", transform=axis.transAxes)
        else:
            axis.legend()
        axis.set_title(sim)
        axis.set_xlabel("lag (steps)")
    axes[0].set_ylabel("success rate difference")
    fig.suptitle("Gain of the best search cell over the no search cell, per run")
    save(fig, out_dir, "gain_over_no_search_by_seed.png")


def matches_cell(row: dict) -> bool:
    """Return True when the row is the cell that every sweep group repeats."""
    for key, want in REPEATED_CELL.items():
        got = row.get(key)
        if isinstance(want, float):
            value = number(got)
            if value is None or abs(value - want) > 1e-9:
                return False
        elif got != want:
            return False
    return True


def find_repeated_cell(path: str):
    """Return the success and error of the repeated cell. Return None when the file has no such row."""
    for row in load_rows(path) or []:
        if matches_cell(row):
            return point(row)
    return None


def fig_repeated_cell(root: str, out_dir: str) -> None:
    """Draw the repeated cell of each mjlab sweep group, before and after per batch seeding."""
    series = {"before per batch seeding": {}, "after per batch seeding": {}}
    for group in SWEEP_GROUPS:
        before = find_repeated_cell(os.path.join(root, "runs", "mjlab", "results", f"{group}_preseed.json"))
        after = find_repeated_cell(os.path.join(root, "runs", "mjlab", "results", f"{group}.json"))
        if before:
            series["before per batch seeding"][group] = before
        if after:
            series["after per batch seeding"][group] = after
    groups = [g for g in SWEEP_GROUPS if g in series["before per batch seeding"] or g in series["after per batch seeding"]]
    if not groups:
        print("skip repeated_cell_floor.png: no mjlab sweep file holds the repeated cell")
        return
    positions = np.arange(len(groups))
    fig, axis = plt.subplots(figsize=(8.0, 5.0))
    for offset, (name, values) in zip((-0.08, 0.08), series.items()):
        heights = np.array([values[g][0] if g in values else np.nan for g in groups])
        errors = np.array([values[g][1] if g in values else np.nan for g in groups])
        axis.errorbar(positions + offset, heights, yerr=errors, fmt="o", capsize=4, markersize=8, label=name)
    axis.set_title("The repeated cell in each mjlab sweep group (depth 2, lag 1, beta 0.5, independent)")
    axis.set_xlabel("sweep group")
    axis.set_ylabel("success rate")
    axis.set_xticks(positions)
    axis.set_xticklabels(groups)
    axis.legend()
    save(fig, out_dir, "repeated_cell_floor.png")


def critic_entries(label: str, path: str) -> list:
    """Return [(label, metrics)] for one critic span file. The top level file holds one dict per sim."""
    data = load_json(path)
    if not isinstance(data, dict):
        return []
    if number(data.get("ratio_r")) is not None:
        sim = data.get("sim")
        return [(f"{label} ({sim})" if isinstance(sim, str) else label, data)]
    entries = []
    for key, value in data.items():
        if isinstance(value, dict) and number(value.get("ratio_r")) is not None:
            entries.append((f"{label} ({key})", value))
    return entries


def fig_critic_span(root: str, out_dir: str) -> None:
    """Draw the critic span ratio and the Q span across candidates, one bar per run."""
    paths = sorted(glob.glob(os.path.join(root, "runs", "*", "results", "critic_action_span.json")))
    if os.path.exists(os.path.join(root, "results", "critic_action_span.json")):
        paths.insert(0, os.path.join(root, "results", "critic_action_span.json"))
    entries = []
    for path in paths:
        entries.extend(critic_entries(run_label(root, path), path))
    if not entries:
        print("skip critic_span.png: no critic_action_span.json holds ratio_r")
        return
    labels = [label for label, _ in entries]
    positions = np.arange(len(labels))
    ratios = np.array([number(metrics.get("ratio_r")) for _, metrics in entries], dtype=float)
    spans = np.array(
        [np.nan if number(m.get("q_span_across_candidates")) is None else number(m.get("q_span_across_candidates")) for _, m in entries]
    )
    fig, axes = plt.subplots(2, 1, figsize=(1.6 * len(labels) + 4.0, 8.0), sharex=True)
    axes[0].bar(positions, ratios, color="tab:blue")
    axes[0].axhline(1.0, color="black", linewidth=1.2, label="ratio 1")
    axes[0].set_title("Critic action span per run")
    axes[0].set_ylabel("ratio r")
    axes[0].legend()
    axes[1].bar(positions, spans, color="tab:green")
    axes[1].set_ylabel("Q span across candidates")
    axes[1].set_xlabel("run")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    save(fig, out_dir, "critic_span.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Make the paper figures.")
    parser.add_argument("--root", default=".", help="the directory that holds runs/ and results/")
    parser.add_argument("--figures", default=None, help="the output directory, default <root>/figures/paper")
    args = parser.parse_args()

    out_dir = args.figures or os.path.join(args.root, "figures", "paper")
    os.makedirs(out_dir, exist_ok=True)

    seeds = {}
    for label, run_dir in SEED_DIRS:
        points = load_arm_points(args.root, run_dir)
        if points:
            seeds[label] = points

    fig_h1_by_seed(seeds, out_dir)
    fig_gain_vs_staleness(seeds, out_dir)
    fig_gain_over_no_search(args.root, out_dir)
    fig_repeated_cell(args.root, out_dir)
    fig_critic_span(args.root, out_dir)


if __name__ == "__main__":
    main()
