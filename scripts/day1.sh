#!/usr/bin/env bash
# The first day of docs/next_steps.md: controls on both simulators, then the demonstration
# quality control on 2D (a buffer at about 65 percent success, a retrain, the critic span probe).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"; P="$ROOT/.venv/bin/python"
mkdir -p results/logs runs/mjlab/results/logs
echo "== $(date +%H:%M:%S) controls 2d"
[ -f results/day1_controls.json ] || SWARM_SIM=2d $P scripts/day1_controls.py --ckpt checkpoints/belief_best.pt --out results/day1_controls.json > results/logs/day1_2d.log 2>&1
echo "== $(date +%H:%M:%S) controls mjlab"
[ -f runs/mjlab/results/day1_controls.json ] || (cd runs/mjlab && SWARM_SIM=mjlab $P $ROOT/scripts/day1_controls.py --ckpt checkpoints/belief_best.pt --out results/day1_controls.json > results/logs/day1_mjlab.log 2>&1)
echo "== $(date +%H:%M:%S) demonstration quality control: collect at noise ${NOISE:-0.5}"
mkdir -p runs/demo65 && [ -f runs/demo65/data/offline.pt ] || (cd runs/demo65 && SWARM_SIM=2d $P $ROOT/scripts/collect_offline.py --envs 256 --steps 800 --noise ${NOISE:-0.5} > collect.log 2>&1 && grep transitions collect.log)
echo "== $(date +%H:%M:%S) demonstration quality control: train"
[ -f runs/demo65/checkpoints/belief_best.pt ] || (cd runs/demo65 && SWARM_SIM=2d $P $ROOT/scripts/train.py --obs belief --steps 12000 --out belief > train.log 2>&1)
echo "== $(date +%H:%M:%S) demonstration quality control: probe"
$P scripts/critic_span.py --ckpt runs/demo65/checkpoints/belief_best.pt --out runs/demo65/results/critic_action_span.json > runs/demo65/probe.log 2>&1
cat runs/demo65/probe.log
echo "== $(date +%H:%M:%S) day1 done"
