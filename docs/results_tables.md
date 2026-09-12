### Controls (256 envs)

| Control | Success (%) | Length on success | Position error (m) | Angle error (rad) |
|---|---|---|---|---|
| scripted_full_team | 99.6 | 91.4 | 0.274 | 0.081 |
| single_pusher | 0.0 | - | 2.227 | 0.404 |
| single_gripper | 0.0 | - | 2.227 | 0.404 |

Throughput: 41,627 env steps per second at 256 envs.

### Training

| Run | Transitions | Final eval success | Best eval success | Wall minutes |
|---|---|---|---|---|
| belief | 3,072,000 | 46.9 | 51.2 | 27 |
| full | 3,072,000 | 28.9 | 28.9 | 9 |
| full36k | 9,216,000 | 72.3 | 77.3 | 55 |

### World model open loop error

| Horizon | Latent MSE | Copy latent MSE | Decoded position error (m) | Copy position error (m) | Decoded angle error (rad) |
|---|---|---|---|---|---|
| 1 | 0.0013 | 0.0022 | 0.204 | 0.217 | 0.071 |
| 2 | 0.0035 | 0.0066 | 0.234 | 0.257 | 0.086 |
| 3 | 0.0063 | 0.0108 | 0.260 | 0.289 | 0.114 |
| 4 | 0.0086 | 0.0152 | 0.290 | 0.322 | 0.144 |
| 6 | 0.0148 | 0.0268 | 0.372 | 0.414 | 0.203 |
| 8 | 0.0220 | 0.0376 | 0.441 | 0.496 | 0.252 |

### H1: search against no search at lag 1 (256 envs, 3 batches)

| Setting | Success (%) | Length on success | ms per step |
|---|---|---|---|
| no search (mean action) | 52.3 ± 0.5 | 98.3 | 13.5 |
| depth 0 (critic argmax) | 42.3 ± 1.4 | 101.0 | 27.5 |
| depth 2 search | 43.2 ± 2.3 | 104.5 | 118.9 |

H1 verdict: depth 2 minus no search = -9.1 points, 2 SE = 4.6 points, **refuted (search is worse)**.

### H2: success (%) by staleness (rows) and depth (columns), beta 0.5

| Lag | no search | D=0 | D=1 | D=2 | D=4 | D=6 | best depth |
|---|---|---|---|---|---|---|---|
| 0 | 38.8 ± 2.3 | 31.5 ± 1.4 | 33.6 ± 1.4 | 33.6 ± 2.4 | 37.2 ± 1.7 | 34.6 ± 1.7 | 4 |
| 1 | 50.3 ± 2.6 | 42.7 ± 3.8 | 42.7 ± 1.7 | 45.6 ± 2.5 | 45.6 ± 1.4 | 45.1 ± 0.7 | 2 |
| 2 | 63.8 ± 1.8 | 52.3 ± 2.4 | 48.2 ± 0.5 | 50.8 ± 2.5 | 49.7 ± 1.9 | 47.9 ± 2.5 | 0 |
| 4 | 61.7 ± 2.8 | 51.6 ± 1.2 | 38.3 ± 1.6 | 40.1 ± 3.9 | 43.0 ± 0.8 | 42.4 ± 1.7 | 0 |

Best depth per lag: {0: 4, 1: 2, 2: 0, 4: 0}. H2 verdict: **confirmed** (non increasing: True, drop from lag 0 to 4: True).

Cost of one step of imagination (depth 1 minus depth 0, points): lag 0: +2.1, lag 1: +0.0, lag 2: -4.2, lag 4: -13.3.

Best search cell minus no search (points): lag 0: -1.6, lag 1: -4.7, lag 2: -11.5, lag 4: -10.2.

Cost: ms per step by depth at lag 1: D=-1: 25, D=0: 26, D=1: 32, D=2: 68, D=4: 143, D=6: 218.

### H3: success (%) by staleness (rows) and tree search discount beta (columns), depth 2

| Lag | beta=0.0 | beta=0.1 | beta=0.3 | beta=0.5 | beta=0.7 | beta=0.9 | beta=1.0 | best beta |
|---|---|---|---|---|---|---|---|---|
| 0 | 31.0 ± 2.3 | 30.7 ± 2.1 | 30.5 ± 1.8 | 34.4 ± 0.9 | 34.4 ± 2.8 | 34.9 ± 2.5 | 36.5 ± 0.7 | 1.0 |
| 1 | 42.2 ± 2.5 | 45.3 ± 2.4 | 42.7 ± 0.7 | 44.3 ± 2.1 | 47.1 ± 0.7 | 45.3 ± 2.3 | 43.5 ± 0.3 | 0.7 |
| 2 | 51.0 ± 1.1 | 49.7 ± 1.1 | 51.6 ± 0.0 | 50.0 ± 0.5 | 50.3 ± 0.9 | 50.0 ± 1.4 | 51.3 ± 2.1 | 0.3 |
| 4 | 48.7 ± 2.6 | 45.6 ± 1.1 | 41.7 ± 3.1 | 41.1 ± 3.9 | 44.0 ± 3.1 | 41.7 ± 3.5 | 41.1 ± 2.8 | 0.0 |

Best beta per lag: {0: 1.0, 1: 0.7, 2: 0.3, 4: 0.0}. H3 verdict: **confirmed** (non increasing: True).

### H4: leader election at matched compute, depth 2

| Lag | Mode | Success (%) | ms per step | Searches per env | Robots following | Leader disagreement |
|---|---|---|---|---|---|---|
| 1 | independent | 44.0 ± 2.1 | 68 | 6.00 | 1.00 | 0.00 |
| 1 | leader | 50.0 ± 3.4 | 68 | 0.98 | 0.56 | 0.50 |
| 1 | round_robin | 27.3 ± 2.1 | 69 | 1.00 | 1.00 | 0.00 |
| 4 | independent | 41.7 ± 1.0 | 67 | 6.00 | 1.00 | 0.00 |
| 4 | leader | 50.3 ± 3.6 | 68 | 0.96 | 0.52 | 0.55 |
| 4 | round_robin | 18.2 ± 0.3 | 70 | 1.00 | 1.00 | 0.00 |

H4 verdict: **confirmed**.

### Robustness: message dropout at lag 1

| dropout | no search (%) | depth 2 search (%) |
|---|---|---|
| 0.0 | 50.3 ± 2.6 | 43.8 ± 4.3 |
| 0.1 | 55.7 ± 1.8 | 44.8 ± 1.6 |
| 0.25 | 59.4 ± 3.6 | 46.6 ± 1.3 |
| 0.5 | 63.5 ± 1.1 | 46.6 ± 0.7 |

### Transfer: team composition at lag 1 (trained on the default team)

| team | no search (%) | depth 2 search (%) |
|---|---|---|
| default | 50.3 ± 2.6 | 42.4 ± 1.4 |
| p2g1s1 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| p4g4s1 | 89.1 ± 1.6 | 85.9 ± 2.8 |
| p6g5s1 | 92.7 ± 1.4 | 85.4 ± 1.9 |
| p8g7s1 | 95.6 ± 1.8 | 89.6 ± 1.3 |

### Ablations on the snapshot (128 envs, 3 batches)

| Setting | Success (%) | Length on success |
|---|---|---|
| no search, lag 0 | 38.8 ± 2.3 | 96.2 |
| no search, lag 1 | 50.3 ± 2.6 | 98.6 |
| no search, lag 2 | 63.8 ± 1.8 | 99.0 |
| no search, lag 4 | 61.7 ± 2.8 | 103.3 |
| no search, lag 6 | 58.1 ± 0.5 | 105.4 |
| no search, lag 8 | 55.2 ± 1.8 | 104.5 |
| messages masked, lag 1 | 13.5 ± 2.6 | 100.9 |
| messages masked, lag 2 | 13.5 ± 2.6 | 100.9 |
| messages masked, lag 4 | 13.5 ± 2.6 | 100.9 |
| sampled policy, lag 1 | 62.2 ± 1.4 | 106.8 |
| sampled policy, lag 2 | 66.9 ± 0.5 | 106.5 |

### First day controls (2d)

| Group | Cell | Success (%) | Envs x batches |
|---|---|---|---|
| F5a | no search, no forward correction, lag 0 | 37.5 ± 3.1 | 128 x 3 |
| F5a | no search, forward correction, lag 0 | 38.8 ± 2.3 | 128 x 3 |
| F5a | no search, no forward correction, lag 1 | 48.4 ± 2.3 | 128 x 3 |
| F5a | no search, forward correction, lag 1 | 50.3 ± 2.6 | 128 x 3 |
| F5a | no search, no forward correction, lag 2 | 45.8 ± 1.4 | 128 x 3 |
| F5a | no search, forward correction, lag 2 | 63.8 ± 1.8 | 128 x 3 |
| F5a | no search, no forward correction, lag 4 | 41.9 ± 2.5 | 128 x 3 |
| F5a | no search, forward correction, lag 4 | 61.7 ± 2.8 | 128 x 3 |
| F3 | trained world model, depth 2 | 43.5 ± 2.0 | 128 x 3 |
| F3 | random world model, depth 2 | 39.1 ± 2.8 | 128 x 3 |
| F3 | trained world model, depth 6 | 45.3 ± 2.3 | 128 x 3 |
| F3 | random world model, depth 6 | 41.7 ± 1.4 | 128 x 3 |
| F2 | decoded scorer, depth 0 | 50.5 ± 2.6 | 128 x 3 |
| F2 | decoded scorer, depth 2 | 41.4 ± 1.6 | 128 x 3 |
| F2 | random scorer, depth 0 | 66.7 ± 2.5 | 128 x 3 |
| F2 | critic scorer, depth 0 | 44.3 ± 4.3 | 128 x 3 |
| F2 | sampled policy | 61.7 ± 1.2 | 128 x 3 |
| F7 | independent, lag 1 | 43.5 ± 2.0 | 128 x 3 |
| F7 | leader fresh election no fallback, lag 1 | 24.7 ± 1.9 | 128 x 3 |
| F7 | independent, lag 4 | 43.0 ± 2.5 | 128 x 3 |
| F7 | leader fresh election no fallback, lag 4 | 19.3 ± 1.8 | 128 x 3 |

