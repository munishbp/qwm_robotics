#!/usr/bin/env bash
# Rerun the fair H1 arms (lags 0, 1, 4) on the three mjlab seeds so every arm carries per episode
# outcomes for the paired bootstrap. The earlier files are kept with a suffix.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"; P="$ROOT/.venv/bin/python"; S="$ROOT/scripts"
for D in runs/mjlab runs/mjlab_seed1 runs/mjlab_seed2; do
  echo "== $(date +%H:%M:%S) fair H1 arms with per episode outcomes, $D"
  [ -f "$D/results/day1_controls_noperenv.json" ] || cp "$D/results/day1_controls.json" "$D/results/day1_controls_noperenv.json"
  (cd "$D" && SWARM_SIM=mjlab $P $S/day1_controls.py --ckpt checkpoints/belief_best.pt --only F1 --out results/day1_controls_f1.json > results/logs/day1_f1_rerun.log 2>&1)
done
echo "== $(date +%H:%M:%S) cooling 15 minutes"; sleep 900
echo "== $(date +%H:%M:%S) day4 done"
