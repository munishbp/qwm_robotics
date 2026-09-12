### Controls (256 envs)

| Control | Success (%) | Length on success | Position error (m) | Angle error (rad) |
|---|---|---|---|---|
| scripted_full_team | 83.6 | 78.9 | 0.330 | 0.140 |
| single_pusher | 0.0 | - | 2.211 | 0.399 |
| single_gripper | 0.0 | - | 2.227 | 0.404 |

Throughput: 9,064 env steps per second at 256 envs.

### Training

| Run | Transitions | Final eval success | Best eval success | Wall minutes |
|---|---|---|---|---|
| belief | 6,144,000 | 27.7 | 48.4 | 66 |
| belief_12k | 3,072,000 | 20.7 | 28.9 | 33 |
| full | 6,144,000 | 25.0 | 25.0 | 28 |

### World model open loop error

| Horizon | Latent MSE | Copy latent MSE | Decoded position error (m) | Copy position error (m) | Decoded angle error (rad) |
|---|---|---|---|---|---|
| 1 | 0.0065 | 0.0102 | 0.209 | 0.218 | 0.088 |
| 2 | 0.0156 | 0.0277 | 0.254 | 0.279 | 0.131 |
| 3 | 0.0264 | 0.0480 | 0.303 | 0.345 | 0.177 |
| 4 | 0.0387 | 0.0722 | 0.356 | 0.419 | 0.227 |
| 6 | 0.0677 | 0.1178 | 0.479 | 0.569 | 0.327 |
| 8 | 0.1028 | 0.1673 | 0.582 | 0.691 | 0.421 |

### H1: search against no search at lag 1 (256 envs, 3 batches)

| Setting | Success (%) | Length on success | ms per step |
|---|---|---|---|
| no search (mean action) | 45.1 ± 1.9 | 77.9 | 30.8 |
| depth 0 (critic argmax) | 61.1 ± 1.4 | 92.8 | 33.0 |
| depth 2 search | 69.5 ± 1.8 | 93.4 | 66.6 |

H1 verdict: depth 2 minus no search = +24.5 points, 2 SE = 5.3 points, **confirmed**.

### H2: success (%) by staleness (rows) and depth (columns), beta 0.5

| Lag | no search | D=0 | D=1 | D=2 | D=4 | D=6 | best depth |
|---|---|---|---|---|---|---|---|
| 0 | 37.2 ± 2.7 | 60.7 ± 2.2 | 69.0 ± 2.5 | 70.8 ± 1.3 | 69.3 ± 1.3 | 72.7 ± 4.3 | 6 |
| 1 | 44.0 ± 1.6 | 62.2 ± 1.1 | 68.0 ± 1.6 | 70.3 ± 2.3 | 68.2 ± 4.4 | 71.4 ± 2.5 | 6 |
| 2 | 45.6 ± 0.7 | 56.5 ± 2.3 | 62.5 ± 1.4 | 62.8 ± 3.0 | 70.6 ± 2.1 | 71.6 ± 2.9 | 6 |
| 4 | 38.8 ± 3.4 | 53.1 ± 0.5 | 58.3 ± 2.5 | 59.6 ± 1.6 | 63.0 ± 1.6 | 60.9 ± 4.3 | 4 |

Best depth per lag: {0: 6, 1: 6, 2: 6, 4: 4}. H2 verdict: **confirmed** (non increasing: True, drop from lag 0 to 4: True).

Cost of one step of imagination (depth 1 minus depth 0, points): lag 0: +8.3, lag 1: +5.7, lag 2: +6.0, lag 4: +5.2.

Best search cell minus no search (points): lag 0: +35.4, lag 1: +27.3, lag 2: +26.0, lag 4: +24.2.

Cost: ms per step by depth at lag 1: D=-1: 28, D=0: 32, D=1: 34, D=2: 46, D=4: 75, D=6: 104.

### H3: success (%) by staleness (rows) and tree search discount beta (columns), depth 2

| Lag | beta=0.0 | beta=0.1 | beta=0.3 | beta=0.5 | beta=0.7 | beta=0.9 | beta=1.0 | best beta |
|---|---|---|---|---|---|---|---|---|
| 0 | 62.2 ± 2.0 | 65.1 ± 1.6 | 69.3 ± 3.4 | 65.9 ± 3.6 | 66.4 ± 2.1 | 72.9 ± 0.3 | 67.2 ± 2.3 | 0.9 |
| 1 | 58.3 ± 1.4 | 62.8 ± 0.9 | 68.5 ± 2.1 | 67.2 ± 0.9 | 68.5 ± 2.8 | 69.0 ± 1.1 | 68.5 ± 1.8 | 0.9 |
| 2 | 60.2 ± 2.7 | 62.8 ± 2.1 | 64.8 ± 2.1 | 67.7 ± 2.8 | 65.9 ± 4.1 | 69.0 ± 0.7 | 66.9 ± 2.1 | 0.9 |
| 4 | 54.7 ± 2.1 | 59.1 ± 1.4 | 58.3 ± 0.3 | 62.0 ± 1.4 | 63.5 ± 0.3 | 58.9 ± 2.3 | 58.6 ± 2.0 | 0.7 |

Best beta per lag: {0: 0.9, 1: 0.9, 2: 0.9, 4: 0.7}. H3 verdict: **confirmed** (non increasing: True).

### H4: leader election at matched compute, depth 2

| Lag | Mode | Success (%) | ms per step | Searches per env | Robots following | Leader disagreement |
|---|---|---|---|---|---|---|
| 1 | independent | 69.0 ± 1.7 | 47 | 6.00 | 1.00 | 0.00 |
| 1 | leader | 41.1 ± 0.3 | 46 | 0.96 | 0.69 | 0.34 |
| 1 | round_robin | 43.8 ± 3.9 | 46 | 1.00 | 1.00 | 0.00 |
| 4 | independent | 60.9 ± 1.6 | 47 | 6.00 | 1.00 | 0.00 |
| 4 | leader | 32.8 ± 1.2 | 45 | 0.95 | 0.65 | 0.40 |
| 4 | round_robin | 32.6 ± 0.9 | 46 | 1.00 | 1.00 | 0.00 |

H4 verdict: **refuted (leader is worse)**.

### Robustness: message dropout at lag 1

| dropout | no search (%) | depth 2 search (%) |
|---|---|---|
| 0.0 | 45.3 ± 1.2 | 67.2 ± 0.9 |
| 0.1 | 50.3 ± 0.7 | 69.5 ± 3.9 |
| 0.25 | 48.7 ± 4.2 | 69.3 ± 4.3 |
| 0.5 | 44.5 ± 2.0 | 67.7 ± 1.7 |

### Transfer: team composition at lag 1 (trained on the default team)

| team | no search (%) | depth 2 search (%) |
|---|---|---|
| default | 45.8 ± 1.4 | 66.4 ± 2.0 |
| p2g1s1 | 2.6 ± 0.9 | 2.3 ± 0.8 |
| p4g4s1 | 56.5 ± 2.9 | 90.4 ± 0.7 |
| p6g5s1 | 57.6 ± 1.4 | 93.2 ± 0.5 |
| p8g7s1 | 63.8 ± 1.4 | 96.6 ± 0.7 |

### Ablations on the snapshot (128 envs, 3 batches)

| Setting | Success (%) | Length on success |
|---|---|---|
| no search, lag 0 | 39.3 ± 0.3 | 77.2 |
| no search, lag 1 | 43.5 ± 2.6 | 74.9 |
| no search, lag 2 | 42.4 ± 2.6 | 73.3 |
| no search, lag 4 | 39.8 ± 1.8 | 67.0 |
| no search, lag 6 | 41.7 ± 2.1 | 73.3 |
| no search, lag 8 | 37.8 ± 0.3 | 69.5 |
| messages masked, lag 1 | 27.3 ± 3.2 | 76.1 |
| messages masked, lag 2 | 26.8 ± 1.0 | 74.3 |
| messages masked, lag 4 | 27.1 ± 1.8 | 72.7 |
| sampled policy, lag 1 | 65.4 ± 2.2 | 85.7 |
| sampled policy, lag 2 | 67.4 ± 0.7 | 87.7 |

