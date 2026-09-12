#!/usr/bin/env bash
# The remaining main track items: two more mjlab training seeds with the fair H1 arms on each,
# then the mjlab sweep rerun with per batch seeding. The GPU idles 15 minutes after every long
# stage so the machine can cool.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"; P="$ROOT/.venv/bin/python"; S="$ROOT/scripts"
cool() { echo "== $(date +%H:%M:%S) cooling 15 minutes"; sleep 900; }
for seed in 1 2; do
  D="runs/mjlab_seed$seed"; mkdir -p "$D/results/logs" "$D/data" "$D/checkpoints"
  [ -e "$D/data/offline.pt" ] || ln -s "$ROOT/runs/mjlab/data/offline.pt" "$D/data/offline.pt"
  if [ ! -f "$D/checkpoints/belief_best.pt" ]; then
    echo "== $(date +%H:%M:%S) train mjlab seed $seed"
    (cd "$D" && SWARM_SIM=mjlab $P $S/train.py --obs belief --steps 24000 --seed $seed --out belief > results/logs/train_belief.log 2>&1)
    cool
  fi
  if [ ! -f "$D/results/day1_controls.json" ]; then
    echo "== $(date +%H:%M:%S) fair H1 arms, seed $seed"
    (cd "$D" && SWARM_SIM=mjlab $P $S/day1_controls.py --ckpt checkpoints/belief_best.pt --only F1 --out results/day1_controls.json > results/logs/day1_f1.log 2>&1)
  fi
done
echo "== $(date +%H:%M:%S) sweep rerun on mjlab seed 0 with per batch seeding"
cd runs/mjlab
for w in h2 h3 h4 robust transfer; do
  [ -f results/${w}_preseed.json ] || mv results/$w.json results/${w}_preseed.json
  [ -f results/$w.json ] || SWARM_SIM=mjlab $P $S/sweep.py --ckpt checkpoints/belief_best.pt --which $w > results/logs/sweep_${w}_rerun.log 2>&1
done
cd "$ROOT"
cool
echo "== $(date +%H:%M:%S) day2 done"
