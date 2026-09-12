"""Build the standalone episode viewer.

Usage: python viewer/build.py results/episodes.json --out figures/viewer.html
The build reads the recorded episodes, inlines them into viewer/index.html, and writes one
self-contained HTML file. The page needs no server and no other local file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

PLACEHOLDER = "/*__EPISODE_DATA__*/"
REQUIRED_KEYS = ("arena_half", "payload_half", "robot_radius", "dt", "goal_pos_tol", "episodes")


def fail(message: str) -> None:
    """Print the reason and stop. The caller reads the reason and fixes the input."""
    print(f"build.py: {message}", file=sys.stderr)
    raise SystemExit(1)


def read_episodes(path: str) -> dict:
    """Read the recorded episodes and check the shape that the viewer needs."""
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        fail(f"cannot read {path}. Run scripts/record_episodes.py first.")
    except json.JSONDecodeError as err:
        fail(f"{path} holds invalid JSON at line {err.lineno}. Record the episodes again.")
    if not isinstance(data, dict):
        fail(f"{path} must hold a JSON object. Record the episodes again.")
    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        fail(f"{path} has no {', '.join(missing)}. Record the episodes again.")
    if not isinstance(data["episodes"], list) or not data["episodes"]:
        fail(f"{path} holds no episode. Record the episodes again.")
    return data


def inline_script(data: dict) -> str:
    """Return the assignment that the page reads.

    The escape of `<` keeps a string value from closing the script element.
    """
    payload = json.dumps(data, separators=(",", ":")).replace("<", "\\u003c")
    return f"window.EPISODES = {payload};"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inline the recorded episodes into the viewer.")
    parser.add_argument("episodes", nargs="?", default="results/episodes.json")
    parser.add_argument("--out", default="figures/viewer.html")
    args = parser.parse_args()

    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    try:
        with open(template_path, encoding="utf-8") as handle:
            template = handle.read()
    except FileNotFoundError:
        fail(f"cannot read {template_path}. The viewer template is missing.")
    if template.count(PLACEHOLDER) != 1:
        fail(f"{template_path} must hold the line {PLACEHOLDER} exactly once.")

    data = read_episodes(args.episodes)
    page = template.replace(PLACEHOLDER, inline_script(data))

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(page)
    size_mb = len(page.encode("utf-8")) / 1e6
    print(f"wrote {args.out} with {len(data['episodes'])} episodes, {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
