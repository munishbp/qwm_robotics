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

Depth 6 minus sampled policy: mean +19.3 ± 6.3 across 3 seeds, sign positive in 3 of 3.
Paired bootstrap 95 percent intervals per seed: mjlab: +12.8 [no per episode data]; mjlab_seed1: +25.3 [no per episode data]; mjlab_seed2: +19.8 [no per episode data].

#### Lag 1

| Arm | mjlab | mjlab_seed1 | mjlab_seed2 | Mean ± SD across seeds |
|---|---|---|---|---|
| Sampled policy | 65.9 | 57.0 | 52.7 | 58.5 ± 6.7 |
| Random candidate | 63.4 | 55.8 | 50.0 | 56.4 ± 6.7 |
| Depth 0 | 62.6 | 80.2 | 70.4 | 71.1 ± 8.8 |
| Depth 6, beta 0.9 | 73.0 | 78.8 | 68.2 | 73.4 ± 5.3 |

Depth 6 minus sampled policy: mean +14.9 ± 7.4 across 3 seeds, sign positive in 3 of 3.
Paired bootstrap 95 percent intervals per seed: mjlab: +7.2 [no per episode data]; mjlab_seed1: +21.9 [no per episode data]; mjlab_seed2: +15.5 [no per episode data].

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
| Sampled policy | 65.8 | 60.2 | 54.7 | 60.2 ± 5.5 |
| Random candidate | 64.9 | 56.6 | 54.6 | 58.7 ± 5.5 |
| Depth 0 | 54.5 | 73.0 | 69.5 | 65.7 ± 9.8 |
| Depth 6, beta 0.9 | 62.9 | 72.7 | 62.7 | 66.1 ± 5.7 |

Depth 6 minus sampled policy: mean +5.9 ± 7.9 across 3 seeds, sign positive in 2 of 3.
Paired bootstrap 95 percent intervals per seed: mjlab: -2.9 [no per episode data]; mjlab_seed1: +12.5 [no per episode data]; mjlab_seed2: +8.0 [no per episode data].

#### Lag 8

| Arm | mjlab | mjlab_seed1 | mjlab_seed2 | Mean ± SD across seeds |
|---|---|---|---|---|
| Sampled policy | 61.0 | 58.8 | 58.4 | 59.4 ± 1.4 |
| Depth 6, beta 0.9 | 50.5 | 67.2 | 59.5 | 59.1 ± 8.4 |

Depth 6 minus sampled policy: mean -0.3 ± 9.5 across 3 seeds, sign positive in 2 of 3.
Paired bootstrap 95 percent intervals per seed: mjlab: -10.5 [-14.1, -6.6]; mjlab_seed1: +8.4 [+4.7, +11.8]; mjlab_seed2: +1.2 [-2.6, +4.8].
