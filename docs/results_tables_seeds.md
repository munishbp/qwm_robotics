### Fair H1 on mjlab across training seeds (256 envs, 5 batches per cell)

Per seed, the best evaluation success during training (2 batches of 128 envs):

- mjlab: 48.4
- mjlab_seed1: 30.9
- mjlab_seed2: 30.1


#### Lag 0

| Arm | mjlab | mjlab_seed1 | mjlab_seed2 | Mean ± SD across seeds |
|---|---|---|---|---|
| Sampled policy | 60.5 | 52.6 | 47.5 | 53.5 ± 6.5 |
| Random candidate | 58.8 | 49.2 | 44.1 | 50.7 ± 7.5 |
| Depth 0 | 64.3 | 79.3 | 67.8 | 70.5 ± 7.8 |
| Depth 6, beta 0.9 | 73.3 | 77.9 | 67.3 | 72.8 ± 5.3 |

Depth 6 minus sampled policy per seed: +12.8, +25.3, +19.8; mean +19.3 ± 6.3. Sign agrees in 3 of 3 seeds.

#### Lag 1

| Arm | mjlab | mjlab_seed1 | mjlab_seed2 | Mean ± SD across seeds |
|---|---|---|---|---|
| Sampled policy | 65.9 | 57.0 | 52.7 | 58.5 ± 6.7 |
| Random candidate | 63.4 | 55.8 | 50.0 | 56.4 ± 6.7 |
| Depth 0 | 62.6 | 80.2 | 70.4 | 71.1 ± 8.8 |
| Depth 6, beta 0.9 | 73.0 | 78.8 | 68.2 | 73.4 ± 5.3 |

Depth 6 minus sampled policy per seed: +7.2, +21.9, +15.5; mean +14.9 ± 7.4. Sign agrees in 3 of 3 seeds.

#### Lag 4

| Arm | mjlab | mjlab_seed1 | mjlab_seed2 | Mean ± SD across seeds |
|---|---|---|---|---|
| Sampled policy | 65.8 | 60.2 | 54.7 | 60.2 ± 5.5 |
| Random candidate | 64.9 | 56.6 | 54.6 | 58.7 ± 5.5 |
| Depth 0 | 54.5 | 73.0 | 69.5 | 65.7 ± 9.8 |
| Depth 6, beta 0.9 | 62.9 | 72.7 | 62.7 | 66.1 ± 5.7 |

Depth 6 minus sampled policy per seed: -2.9, +12.5, +8.0; mean +5.9 ± 7.9. Sign agrees in 2 of 3 seeds.
