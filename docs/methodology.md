# Experiment methodology

This document is the replication recipe. It states what runs, in what order, with what settings,
on what hardware, and how each hypothesis is decided from the outputs. The reasons behind each
design choice are in `design.md`. The equations are in `math.md`. The numbers are in `results.md`.

## 1. Scope

The proposal targets mjlab (MuJoCo Warp). This study runs the same method on a batched 2D rigid
body transport simulator written in PyTorch (`swarm/env.py`). The proposal's risk section asks
for a trivial version of the task before the full one, and this is it. Every hypothesis concerns
the search mechanism under decentralization and staleness, not contact physics, so the 2D task
keeps the question intact. Moving the same code to mjlab is future work and needs only a new env
class with the API in `design.md` section 3.6.

## 2. Hardware and software

| Item | Value |
|---|---|
| GPU | one NVIDIA GeForce RTX 5090, 32 GB, driver 610.43.02 |
| CPU and RAM | shared workstation, 46 GB RAM, other jobs running at the same time |
| Python | 3.13 |
| PyTorch | 2.9.0+cu128 |
| Lock files | `uv.lock`, `pylock.toml` |
| Memory caps | 25 percent of the GPU per process, 3 GB host RSS watchdog (`swarm/compute.py`) |

Every script is deterministic given its seed except for CUDA kernel nondeterminism in scatter
and attention kernels. Rerunning with the same seed reproduces the numbers to within evaluation
noise, not bit for bit.

## 3. The task

See `design.md` section 3 for the constants. In short: a team of 3 pushers, 2 grippers, and 1
scout moves a rectangular payload to a goal pose inside a 10 by 10 m arena. The payload moves
only when the net force exceeds a friction threshold that latched grippers lower. Pushers push,
grippers latch and pull weakly, the scout only senses. Reward is 1 on success and 0 otherwise.
The episode ends on success or after 150 steps of 0.1 s.

Each robot observes its own position, its type, the goal, and the payload pose if the payload
center is within its sensing range (2 m for pushers and grippers, 15 m for the scout). Robots
exchange messages within 4 m unless the payload blocks the line between them.

## 4. Pipeline

`scripts/run_all.sh` runs every step below in order. Each step is one process. The full run
takes about 90 minutes on the hardware above.

| Step | Command | Output | Purpose |
|---|---|---|---|
| 1 | `scripts/run_controls.py` | `results/controls.json` | Task solvable by the scripted team, unsolvable by one robot, throughput |
| 2 | `scripts/collect_offline.py --envs 256 --steps 800` | `data/offline.pt` | 204,800 scripted transitions with action noise 0.2 |
| 3 | `scripts/train.py --obs belief --steps 12000` | `checkpoints/belief.pt`, `results/train_belief.jsonl` | The decentralized belief agent. This is the snapshot every sweep uses |
| 4 | `scripts/train.py --obs full --steps 12000` | `checkpoints/full.pt`, `results/train_full.jsonl` | Centralized full state baseline |
| 5 | `scripts/world_model_error.py` | `results/wm_error.json` | Open loop latent and decoded error against horizon |
| 6 | `scripts/evaluate.py --depth -1, 0, 2 --lag 1` | `results/h1_depth*.json` | H1 |
| 7 | `scripts/sweep.py --which h2 h3 h4 robust transfer` | `results/h2.json` and so on | H2, H3, H4, robustness, transfer |
| 8 | `scripts/record_episodes.py` | `results/episodes.json` | Episode replays for the viewer |
| 9 | `scripts/plot.py`, `viewer/build.py` | `figures/` | Figures and the interactive viewer |

## 5. Training settings

| Setting | Value |
|---|---|
| Parallel envs | 256 |
| Batched env steps | 12,000, which is 3.07 million transitions |
| Updates per batched env step | 4 |
| Batch | 256 rows, half offline and half online, times 6 robots |
| Critic ensemble | 10 heads, target is the minimum of 2 random heads |
| Discount, Polyak rate, learning rate | 0.99, 0.005, 3e-4 |
| Target entropy | -3 |
| Training staleness | lag 1 |
| Latent size, hidden size | 64, 256 |
| Frame stack | 3 observations plus the last action |
| Seed | 0 |

The world model and the decoder train from the first update on the same batches, with a stop
gradient on the latents. There is no separate pretraining phase because the encoders are random
at the start, and training on the offline half of every batch is the same data the proposal
pretrains on.

Evaluation during training runs 128 envs with the mean action every 500 steps and records the
first episode of every env.

## 6. Evaluation protocol

Every reported number is the mean over 3 batches of 256 envs, each batch from a different env
seed (1000, 1001, 1002), with the standard error over batches. A batch runs 150 steps from reset
and scores the first episode of every env, so every env contributes exactly one episode. The
mean action is used unless search is on.

Search settings unless a sweep varies them: 8 sampled candidates plus the mean action, beam 4,
tree search discount 0.5, independent mode. The no search baseline is the mean action, reported
as depth -1. Depth 0 is the critic argmax over the root candidates with no world model call.

## 7. Decision rules

| Hypothesis | Measure | Confirmed when | Refuted when |
|---|---|---|---|
| H1 | success at lag 1: depth -1 against depth 2 | depth 2 exceeds depth -1 by more than 2 standard errors | otherwise |
| H2 | best depth per lag over depths 0 to 6, at lags 0, 1, 2, 4 | the best depth is non increasing in lag and drops by at least one level from lag 0 to lag 4 | the best depth stays flat or rises |
| H3 | best beta per lag at depth 2 | the best beta is non increasing in lag | otherwise |
| H4 | success at matched compute: independent, leader, round robin | leader exceeds independent by more than 2 standard errors at some lag | otherwise |

A hypothesis whose measure is within noise at every cell is reported as not resolved, with the
noise level stated. That is a result about the task, not a failure of the protocol.

## 8. What is not controlled

- One training seed. The sweeps hold the snapshot fixed, so cell to cell comparisons are paired
  and share the same training noise. Claims across training runs (belief against full) rest on one
  seed each and are reported as such.
- The scripted controller is not optimal, so the offline data is a floor, not a ceiling.
- The 2D simulator has no robot to robot collisions and quasi static payload motion.

## 9. Files that a reader needs

- `swarm/env.py` for the physics and observation, `swarm/scripted.py` for the demo source.
- `swarm/belief.py` for the message table, forward correction, and fusion.
- `swarm/search.py` for the tree search.
- `scripts/sweep.py` for the exact grids.
- `tests/` for the checks that ran before training.
