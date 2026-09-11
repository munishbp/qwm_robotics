# Decentralized World Model Search for Heterogeneous Cooperative Transport

A team of heterogeneous robots moves an awkward payload to a goal pose under sparse reward and
partial observability. Each robot keeps its own belief, trains a Q function on real transitions,
and runs a short tree search over a learned world model at decision time. The question is whether
test time search still helps when the robot must imagine what its teammates do from information
that is already a frame old.

The full proposal, hypotheses, method, baselines, and schedule are in
[proposal_shared_latent_swarm_transport.md](proposal_shared_latent_swarm_transport.md).

A primer on every concept the proposal depends on is in [concepts.md](concepts.md).

## Stack

- mjlab (MuJoCo Warp physics)
- PyTorch
- RLPD, an off policy Q learning base
- Single RTX 5090

## Branches

One branch per phase of the schedule. Each phase merges into `main` before the next phase starts.
Rebase a phase branch on `main` when its week begins.

| Branch | Week | Owns |
|---|---|---|
| `env` | 1 | mjlab task, negative control, scripted controller, offline buffer |
| `rlpd` | 2 | Off policy loop with a concatenated observation |
| `beliefs` | 3 | Type specific encoders, attention fusion, lag, uncertainty heads |
| `world-model` | 4 | Residual dynamics model pretraining and error curves |
| `search` | 5 | Multi agent tree search, the H1 result |
| `experiments` | 6 to 7 | H2 grid, H3 sweep, H4 leader variants, robustness, transfer |
| `writeup` | 8 | Figures and the final report |
