#!/usr/bin/env bash
# Full replication pipeline. One GPU process at a time. Logs go to results/logs.
# Usage: bash scripts/run_all.sh [STEPS]   (default 12000 batched env steps of 256 envs)
set -euo pipefail
cd "$(dirname "$0")/.."
STEPS="${1:-12000}"
P=.venv/bin/python
mkdir -p results/logs
run() { echo "== $(date +%H:%M:%S) $*"; "$@"; }
[ -f results/controls.json ] || run $P scripts/run_controls.py   > results/logs/controls.log 2>&1
[ -f data/offline.pt ] || run $P scripts/collect_offline.py --envs 256 --steps 800 > results/logs/collect.log 2>&1
run $P scripts/train.py --obs belief --steps "$STEPS" --out belief > results/logs/train_belief.log 2>&1
run $P scripts/train.py --obs full   --steps "$STEPS" --out full   > results/logs/train_full.log 2>&1
run $P scripts/world_model_error.py --ckpt checkpoints/belief_best.pt > results/logs/wm_error.log 2>&1
for d in -1 0 2; do
  run $P scripts/evaluate.py --ckpt checkpoints/belief_best.pt --depth $d --lag 1 --out results/h1_depth$d.json > results/logs/h1_depth$d.log 2>&1
done
run $P scripts/sweep.py --ckpt checkpoints/belief_best.pt --which h2 h3 h4 robust transfer > results/logs/sweep.log 2>&1
run $P scripts/record_episodes.py --ckpt checkpoints/belief_best.pt   > results/logs/record.log 2>&1
run $P scripts/plot.py                                           > results/logs/plot.log 2>&1
run $P viewer/build.py                                           > results/logs/viewer.log 2>&1
echo "== $(date +%H:%M:%S) done"
