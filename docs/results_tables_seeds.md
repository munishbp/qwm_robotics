### Fair H1 on mjlab across training seeds (256 envs, 5 batches per cell)

Per seed, the best evaluation success during training (2 batches of 128 envs):

- mjlab: 48.4
- mjlab_seed1: 30.9
- mjlab_seed2: 30.1


#### Lag 0

| Arm | mjlab | mjlab_seed1 | mjlab_seed2 | Mean ± SD across seeds |
|---|---|---|---|---|
| Sampled policy | 59.6 | 50.2 | 50.0 | 53.3 ± 5.5 |
| Random candidate | 58.6 | 48.3 | 44.8 | 50.6 ± 7.2 |
| Depth 0 | 64.8 | 79.6 | 67.6 | 70.7 ± 7.9 |
| Depth 6, beta 0.9 | 73.1 | 80.2 | 65.8 | 73.0 ± 7.2 |

Depth 6 minus sampled policy: mean +19.8 ± 8.9 across 3 seeds, sign positive in 3 of 3.
Paired bootstrap 95 percent intervals per seed: mjlab: +13.5 [+10.1, +17.0]; mjlab_seed1: +30.0 [+26.7, +33.3]; mjlab_seed2: +15.8 [+12.0, +19.4].

#### Lag 1

| Arm | mjlab | mjlab_seed1 | mjlab_seed2 | Mean ± SD across seeds |
|---|---|---|---|---|
| Sampled policy | 66.0 | 57.4 | 54.8 | 59.4 ± 5.9 |
| Random candidate | 63.6 | 53.3 | 50.5 | 55.8 ± 6.9 |
| Depth 0 | 61.7 | 78.4 | 71.2 | 70.4 ± 8.3 |
| Depth 6, beta 0.9 | 70.5 | 79.8 | 69.9 | 73.4 ± 5.6 |

Depth 6 minus sampled policy: mean +14.0 ± 9.0 across 3 seeds, sign positive in 3 of 3.
Paired bootstrap 95 percent intervals per seed: mjlab: +4.5 [+1.1, +8.0]; mjlab_seed1: +22.4 [+19.0, +25.9]; mjlab_seed2: +15.2 [+11.8, +18.7].

#### Lag 2

| Arm | mjlab | mjlab_seed1 | mjlab_seed2 | Mean ± SD across seeds |
|---|---|---|---|---|
| Sampled policy | 69.1 | 59.5 | 58.0 | 62.2 ± 6.0 |
| Depth 6, beta 0.9 | 70.1 | 77.7 | 66.6 | 71.5 ± 5.7 |

Depth 6 minus sampled policy: mean +9.3 ± 8.6 across 3 seeds, sign positive in 3 of 3.
Paired bootstrap 95 percent intervals per seed: mjlab: +1.0 [-2.6, +4.3]; mjlab_seed1: +18.2 [+14.6, +21.6]; mjlab_seed2: +8.7 [+4.8, +12.3].

#### Lag 4

| Arm | mjlab | mjlab_seed1 | mjlab_seed2 | Mean ± SD across seeds |
|---|---|---|---|---|
| Sampled policy | 64.6 | 60.0 | 55.9 | 60.2 ± 4.4 |
| Random candidate | 68.0 | 57.3 | 54.7 | 60.0 ± 7.0 |
| Depth 0 | 55.4 | 75.0 | 68.0 | 66.1 ± 9.9 |
| Depth 6, beta 0.9 | 64.1 | 71.2 | 61.7 | 65.7 ± 5.0 |

Depth 6 minus sampled policy: mean +5.5 ± 5.9 across 3 seeds, sign positive in 2 of 3.
Paired bootstrap 95 percent intervals per seed: mjlab: -0.5 [-4.2, +3.3]; mjlab_seed1: +11.2 [+7.3, +14.8]; mjlab_seed2: +5.9 [+2.3, +9.4].

#### Lag 8

| Arm | mjlab | mjlab_seed1 | mjlab_seed2 | Mean ± SD across seeds |
|---|---|---|---|---|
| Sampled policy | 61.0 | 58.8 | 58.4 | 59.4 ± 1.4 |
| Depth 6, beta 0.9 | 50.5 | 67.2 | 59.5 | 59.1 ± 8.4 |

Depth 6 minus sampled policy: mean -0.3 ± 9.5 across 3 seeds, sign positive in 2 of 3.
Paired bootstrap 95 percent intervals per seed: mjlab: -10.5 [-14.1, -6.6]; mjlab_seed1: +8.4 [+4.7, +11.8]; mjlab_seed2: +1.2 [-2.6, +4.8].
