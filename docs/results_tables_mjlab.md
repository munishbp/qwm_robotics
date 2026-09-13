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
| 0 | 37.8 ± 2.7 | 61.7 ± 3.4 | 71.9 ± 1.6 | 67.7 ± 1.7 | 69.3 ± 2.1 | 74.0 ± 0.7 | 6 |
| 1 | 43.8 ± 3.1 | 60.9 ± 1.2 | 67.2 ± 2.8 | 71.6 ± 1.4 | 70.8 ± 1.6 | 71.1 ± 1.2 | 2 |
| 2 | 40.6 ± 1.8 | 61.7 ± 1.2 | 65.1 ± 2.1 | 66.4 ± 1.2 | 67.2 ± 2.5 | 66.7 ± 1.8 | 4 |
| 4 | 44.3 ± 2.5 | 56.5 ± 0.7 | 61.7 ± 2.5 | 59.9 ± 1.0 | 59.6 ± 2.1 | 60.2 ± 1.2 | 1 |

Best depth per lag: {0: 6, 1: 2, 2: 4, 4: 1}. H2 verdict: **not resolved** (non increasing: False, drop from lag 0 to 4: True).

Cost of one step of imagination (depth 1 minus depth 0, points): lag 0: +10.2, lag 1: +6.2, lag 2: +3.4, lag 4: +5.2.

Best search cell minus no search (points): lag 0: +36.2, lag 1: +27.9, lag 2: +26.6, lag 4: +17.4.

Cost: ms per step by depth at lag 1: D=-1: 29, D=0: 32, D=1: 35, D=2: 46, D=4: 75, D=6: 104.

### H3: success (%) by staleness (rows) and tree search discount beta (columns), depth 2

| Lag | beta=0.0 | beta=0.1 | beta=0.3 | beta=0.5 | beta=0.7 | beta=0.9 | beta=1.0 | best beta |
|---|---|---|---|---|---|---|---|---|
| 0 | 60.2 ± 1.6 | 66.1 ± 1.8 | 67.7 ± 2.2 | 71.6 ± 0.9 | 68.2 ± 1.1 | 69.5 ± 2.4 | 73.2 ± 0.3 | 1.0 |
| 1 | 62.2 ± 1.1 | 63.8 ± 2.3 | 66.1 ± 1.1 | 71.1 ± 1.2 | 73.2 ± 1.8 | 72.7 ± 4.6 | 68.0 ± 2.7 | 0.7 |
| 2 | 60.7 ± 3.8 | 62.0 ± 3.2 | 65.1 ± 0.3 | 66.4 ± 0.5 | 66.7 ± 3.3 | 68.8 ± 4.1 | 68.0 ± 1.2 | 0.9 |
| 4 | 54.2 ± 2.1 | 60.7 ± 1.8 | 62.5 ± 1.6 | 58.9 ± 1.9 | 58.9 ± 0.7 | 57.3 ± 1.7 | 59.1 ± 2.3 | 0.3 |

Best beta per lag: {0: 1.0, 1: 0.7, 2: 0.9, 4: 0.3}. H3 verdict: **not resolved** (non increasing: False).

### H4: leader election at matched compute, depth 2

| Lag | Mode | Success (%) | ms per step | Searches per env | Robots following | Leader disagreement |
|---|---|---|---|---|---|---|
| 1 | independent | 69.3 ± 1.8 | 47 | 6.00 | 1.00 | 0.00 |
| 1 | leader | 39.8 ± 3.6 | 46 | 0.96 | 0.69 | 0.34 |
| 1 | round_robin | 43.8 ± 0.5 | 46 | 1.00 | 1.00 | 0.00 |
| 4 | independent | 58.9 ± 2.9 | 46 | 6.00 | 1.00 | 0.00 |
| 4 | leader | 33.6 ± 1.6 | 45 | 0.95 | 0.66 | 0.39 |
| 4 | round_robin | 36.7 ± 3.7 | 46 | 1.00 | 1.00 | 0.00 |

H4 verdict: **refuted (leader is worse)**.

### Robustness: message dropout at lag 1

| dropout | no search (%) | depth 2 search (%) |
|---|---|---|
| 0.0 | 48.2 ± 3.0 | 71.9 ± 0.8 |
| 0.1 | 50.0 ± 2.7 | 69.3 ± 2.6 |
| 0.25 | 44.5 ± 1.6 | 71.1 ± 0.5 |
| 0.5 | 45.1 ± 0.7 | 70.6 ± 1.8 |

### Transfer: team composition at lag 1 (trained on the default team)

| team | no search (%) | depth 2 search (%) |
|---|---|---|
| default | 46.1 ± 1.6 | 69.5 ± 0.9 |
| p2g1s1 | 2.3 ± 0.5 | 3.1 ± 0.5 |
| p4g4s1 | 56.5 ± 2.8 | 93.0 ± 1.2 |
| p6g5s1 | 57.6 ± 3.6 | 94.5 ± 1.2 |
| p8g7s1 | 65.6 ± 1.6 | 93.8 ± 1.8 |

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

### First day controls (mjlab)

| Group | Cell | Success (%) | Envs x batches |
|---|---|---|---|
| F5a | no search, no forward correction, lag 0 | 38.8 ± 3.3 | 128 x 3 |
| F5a | no search, forward correction, lag 0 | 39.8 ± 2.0 | 128 x 3 |
| F5a | no search, no forward correction, lag 1 | 40.1 ± 2.1 | 128 x 3 |
| F5a | no search, forward correction, lag 1 | 44.8 ± 0.7 | 128 x 3 |
| F5a | no search, no forward correction, lag 2 | 39.6 ± 1.4 | 128 x 3 |
| F5a | no search, forward correction, lag 2 | 45.1 ± 0.7 | 128 x 3 |
| F5a | no search, no forward correction, lag 4 | 37.0 ± 1.1 | 128 x 3 |
| F5a | no search, forward correction, lag 4 | 45.1 ± 3.1 | 128 x 3 |
| F3 | trained world model, depth 2 | 67.4 ± 2.5 | 128 x 3 |
| F3 | random world model, depth 2 | 57.0 ± 1.4 | 128 x 3 |
| F3 | trained world model, depth 6 | 68.8 ± 0.5 | 128 x 3 |
| F3 | random world model, depth 6 | 56.0 ± 2.9 | 128 x 3 |
| F2 | decoded scorer, depth 0 | 45.6 ± 2.1 | 128 x 3 |
| F2 | decoded scorer, depth 2 | 39.3 ± 1.1 | 128 x 3 |
| F2 | random scorer, depth 0 | 68.2 ± 1.4 | 128 x 3 |
| F2 | critic scorer, depth 0 | 58.3 ± 1.8 | 128 x 3 |
| F2 | sampled policy | 71.4 ± 2.8 | 128 x 3 |
| F7 | independent, lag 1 | 68.8 ± 0.5 | 128 x 3 |
| F7 | leader fresh election no fallback, lag 1 | 28.6 ± 2.3 | 128 x 3 |
| F7 | independent, lag 4 | 56.8 ± 2.0 | 128 x 3 |
| F7 | leader fresh election no fallback, lag 4 | 21.1 ± 1.2 | 128 x 3 |
| F1 | H1 sampled policy, lag 0 | 60.5 ± 1.8 | 256 x 5 |
| F1 | H1 random candidate, lag 0 | 58.8 ± 0.7 | 256 x 5 |
| F1 | H1 depth 0, lag 0 | 64.3 ± 1.4 | 256 x 5 |
| F1 | H1 depth 6 beta 0.9, lag 0 | 73.3 ± 0.5 | 256 x 5 |
| F1 | H1 sampled policy, lag 1 | 65.9 ± 1.2 | 256 x 5 |
| F1 | H1 random candidate, lag 1 | 63.4 ± 0.6 | 256 x 5 |
| F1 | H1 depth 0, lag 1 | 62.6 ± 0.6 | 256 x 5 |
| F1 | H1 depth 6 beta 0.9, lag 1 | 73.0 ± 1.8 | 256 x 5 |
| F1 | H1 sampled policy, lag 4 | 65.8 ± 0.6 | 256 x 5 |
| F1 | H1 random candidate, lag 4 | 64.9 ± 1.3 | 256 x 5 |
| F1 | H1 depth 0, lag 4 | 54.5 ± 1.7 | 256 x 5 |
| F1 | H1 depth 6 beta 0.9, lag 4 | 62.9 ± 1.5 | 256 x 5 |

