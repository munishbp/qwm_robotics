# Decentralized World Model Search for Heterogeneous Cooperative Transport

**Project type:** Research, with a strong literature reimplementation component
**Timeline:** 8 weeks, starting the first week of October, due the last week of November
**Hardware:** single RTX 5090
**Stack:** mjlab (MuJoCo Warp physics), PyTorch, off policy Q learning

---

## 1. What this project is

QWM (Dong et al., 2026) makes a specific claim: the right way to use a learned world model in online RL is not to train the policy inside it, but to keep policy and critic trained on real transitions and use the model only at decision time, to run a short tree search over candidate actions. Training inside a learned model compounds its bias. Searching with it at test time does not, because nothing the model imagines ever becomes a training target.

They show this works well on single arm manipulation in Robomimic and LIBERO, beating both model free baselines and model based methods like TD-MPC2 and EfficientZero V2 under sparse rewards.

Every part of that setup is single agent, fully observed, and centralized. My project asks whether the mechanism survives when it is none of those things.

**Research question.** Does test time world model search still improve online RL when the agent is one member of a heterogeneous team, cannot observe the whole state, and can only imagine what its teammates will do from information that is already a frame old?

**Why this is not just a reimplementation.** In the single agent case, the imagined future is fully determined by the actions you are searching over. In a team it is not. When one robot imagines the payload moving, that prediction is only correct if it also correctly imagines what its three teammates are about to do. Those teammates are acting on their own beliefs, which the searching robot only knows as of the previous timestep. So the search tree has a second error source that QWM never faces, and that error source grows with depth in a different way than model error does.

---

## 2. Hypotheses

Stating these upfront because they are falsifiable and the experiments are designed to test them specifically.

**H1.** Test time search improves a decentralized multi agent Q learning baseline, reproducing QWM's single agent result in a harder setting.

**H2 (headline).** The best search depth shrinks as teammate information gets staler. QWM found moderate depth best and attributed the falloff at high depth to world model error. My claim is that in a team, imagined teammate error dominates model error, so the useful depth collapses faster and collapses further the older the teammate information is.

**H3.** The tree search discount that controls how much deep imagined value counts should also shrink with staleness, for the same reason. QWM landed on small values already. I expect mine to be smaller still and to trend down as lag increases.

**H4.** Letting the lowest uncertainty robot run the search and broadcast its choice beats every robot searching independently, at an equal total compute budget.

H2 is the result the project lives or dies on, and it is a two dimensional sweep of depth against staleness that requires no retraining at all.

---

## 3. Task

A team of robots moves an awkward payload to a goal pose. Sparse reward, meaning zero everywhere except on task completion, matching QWM's setting so I do not need to learn a reward model.

The team is heterogeneous. Pushers have traction but cannot grip. Grippers can latch but are weak. One scout senses well but cannot move the payload. No single robot finishes alone.

Partial observability is built in, not hoped for. Each robot senses a limited angular sector of the payload, sensing range is capped, and the payload body occludes views across it.

**Negative control, week 1.** Train one robot with full state sensing and confirm it fails. If it succeeds, the task is too easy and I redesign before writing model code. This is the cheapest possible insurance against a dead project.

**Offline data.** QWM pretrains the world model on demonstration transitions and RLPD samples half its batches from an offline buffer. I have no human demos, so a scripted centralized force allocation controller with known payload inertia generates them. That controller was already going to be a baseline, so it does double duty.

---

## 4. Method

### 4.1 Per robot belief with lagged neighbors

There is no central fusion node. Every robot runs the same model and builds its own belief.

For robot i at step t:

1. Robot i encodes its own current observation. This is its primary view and enters at full weight, never discounted.
2. It receives latents from neighbors inside its communication radius, stamped at t-1.
3. Each stale neighbor latent is rolled forward one step through the shared dynamics model before fusion, so everything being combined is nominally at time t.
4. An attention layer over the agent set fuses fresh local and corrected remote latents. Permutation invariant, so team size is not baked in.
5. Each robot also emits a scalar uncertainty from ensemble disagreement. The lowest uncertainty robot is elected leader for that step and its latent is weighted highest. Election happens independently inside each robot from whatever messages it received, so robots can briefly disagree about the leader. That is realistic and I will measure it.

Robot types get their own encoder and decoder heads. What they share is one dynamics core. That sharing is the load bearing claim: if a pusher and a gripper drive the same model, the model has learned payload physics rather than per robot quirks.

### 4.2 World model

Following QWM's state based setup, a deterministic residual model that predicts the state delta, implemented as a small MLP, pretrained offline on scripted demonstration transitions with an MSE objective, then frozen. Nothing exotic. It stays frozen so that any change in performance across the sweeps is attributable to search settings and not to a drifting model.

Input is the fused belief plus the joint action vector across the team. Output is the next payload and agent state.

### 4.3 Multi agent tree search

This is the piece that is actually new.

At each step, robot i:

- samples N candidate actions for itself from its own policy
- **imagines teammate actions** by running the shared policy on its own estimate of each teammate's belief, which is the stale t-1 latent rolled forward
- queries the world model on the resulting joint action to get predicted next states
- recurses to depth D, pruning to J surviving paths using accumulated discounted Q values, exactly as QWM does
- scores nodes with a combination of the direct critic estimate and the model rollout estimate, weighted by a tree search discount that damps deeper imagined value
- picks the action maximizing the aggregated root score

Search runs during both online data collection and evaluation, since QWM's ablation found the two stages complementary and the combination strongest.

**Compute.** QWM already lists search overhead as a limitation, and I am multiplying it by team size. Three things save it. The world model is a small MLP, not a video diffusion model. QWM found performance largely insensitive to how many paths survive pruning, so a very narrow beam is fine. And the whole search batches across agents and environments on the GPU. I will still measure wall clock per step and report it, because a method that only works with unlimited decision time is worth knowing about.

### 4.4 Base algorithm

RLPD, meaning SAC style updates with a critic ensemble, a high update to data ratio, and symmetric sampling from offline and online buffers. QWM implements on top of both EXPO and RLPD and shows gains on both, and RLPD is much simpler to stand up.

**Why not PPO.** QWM's appendix compares searching with a learned state value function against searching on top of Q learning, and the V variant loses substantially everywhere. PPO gives me V. Adopting the method means adopting Q learning. This is the real cost of building on this paper and I am accepting it deliberately.

**Why not GRPO.** GRPO's selling point is dropping the value network, which matters when the critic is a multi billion parameter model. Here the critic is the thing the entire method depends on. Beyond that, group relative advantage assigns one trajectory level scalar to every timestep, which cannot say whether the gripper or the scout caused a success, and the spread within a group would mostly reflect message dropout and randomization rather than policy quality. It would be a reasonable choice for a discrete subgoal layer above this one, which is a future work sentence and not part of this project.

---

## 5. Baselines

| Baseline | What it isolates |
|---|---|
| RLPD with no search, decentralized beliefs | Whether search helps at all (H1) |
| Search with V instead of Q | Replicates QWM's appendix finding in a team |
| Independent per robot world models, no messages | Whether sharing is doing the work |
| Zero lag oracle communication | The cost of staleness |
| Scripted centralized force allocation | The classical control comparison, also the demo source |

The zero lag oracle is the most informative one, because the gap between it and the real system is exactly the quantity H2 is about.

---

## 6. Experiments

**Core**
- Environment steps to a success threshold, search versus no search
- Success rate and final payload pose error
- Wall clock per decision step, with and without search

**Headline, H2**
- Grid of search depth against teammate staleness. Depth values spanning shallow to deep, staleness at 0, 1, 2, and 4 frames. Every cell is a test time evaluation on the same trained model.

**H3**
- Tree search discount sweep, run at each staleness level

**H4**
- Leader only search versus per robot search versus round robin leader, at matched total compute

**Generalization, all test time**
- Train with 8 robots, evaluate at 4, 12, 16, 32
- Train on 4 pushers plus 4 grippers, evaluate on unseen type ratios

**Robustness**
- Message dropout at 10, 25, 50 percent
- Belief divergence between robots over time

The point worth repeating to the grader: everything after the core section is a test time evaluation. One training run funds the entire results section.

---

## 7. Schedule

**Week 1, Oct 1 to 7. Environment and negative control.**
Build the heterogeneous transport task in mjlab. Benchmark throughput. Run the single robot full sensing control and confirm failure. Write the scripted force allocation controller and collect the offline demonstration buffer.
Deliverable: working env, throughput numbers, offline dataset, confirmed task difficulty.

**Week 2, Oct 8 to 14. Off policy loop.**
Stand up RLPD against the environment with a simple concatenated observation, no beliefs, no search. This is the week the algorithm switch gets paid for.
Deliverable: a learning curve that goes up.

**Week 3, Oct 15 to 21. Beliefs.**
Type specific encoders, attention fusion, one frame lag, forward correction, uncertainty heads. Still no search.
Deliverable: decentralized belief RLPD baseline, plus belief prediction error.

**Week 4, Oct 22 to 28. World model.**
Pretrain the residual dynamics model on the offline buffer. Validate prediction error against horizon before wiring it into anything.
Deliverable: model error curves, which also tell me a sane starting depth.

**Week 5, Oct 29 to Nov 4. Tree search.**
Implement search with imagined teammate actions, pruning, and value aggregation. Get H1.
Deliverable: search versus no search, the core result.

**Week 6, Nov 5 to 11. Headline sweeps.**
Depth by staleness grid, discount sweep. No retraining, so this week is compute and plotting.
Deliverable: the H2 figure.

**Week 7, Nov 12 to 18. Remaining ablations.**
Search with V, leader election variants, dropout, scale and composition transfer.

**Week 8, Nov 19 to 25. Writeup and figures. Submit in the last week of November.**

**Cut order if I slip.** Week 7 goes first, and within it the V comparison survives longest because it is the direct replication. Weeks 5 and 6 are protected. If week 2 runs long I cut the scale transfer study rather than compressing the sweeps.

---

## 8. Risks

**The algorithm switch eats week 2 and part of week 3.** This is the biggest schedule risk and it is entirely front loaded, which is the good place for risk to sit. Mitigation is to use an existing PyTorch RLPD implementation rather than writing SAC from scratch, and to get it running against a trivial version of the task first.

**Imagined teammate error swamps everything.** If teammate prediction is so bad that search never helps at any depth, H1 fails and H2 is untestable. This is a real possibility and I would rather find it in week 5 than week 7. The zero lag oracle baseline diagnoses it directly: if search helps with oracle communication and not without, that is itself a clean and reportable finding about the limits of QWM's mechanism under decentralization.

**Search cost.** Team size multiplies QWM's known overhead. Narrow pruning, shallow depth, a small MLP model, and GPU batching across agents and environments. Measured and reported either way.

**Sparse reward with no demos.** The scripted controller has to be good enough to produce some successful trajectories or the offline buffer is worthless. Week 1 deliverable, so I find out immediately.

**Contact scaling in the simulator.** MuJoCo Warp gets its throughput from many parallel environments, not many bodies in one. Off policy Q learning needs fewer environment steps than PPO does, which actually helps here, but team size stays modest at 8 to 16 regardless.

---

## 9. What a good outcome looks like

Minimum: a working decentralized heterogeneous transport policy, plus an honest answer to whether test time world model search helps when teammate actions must be imagined from stale information. A clean negative here is still a contribution, because it maps a boundary of a method that is currently being applied enthusiastically in single agent settings.

Good: the depth by staleness grid shows the predicted collapse, giving a concrete rule for how far to search as a function of how old your teammate information is.

Stretch: leader elected search beats independent search at matched compute, which would say something useful about where to spend decision time budget in a decentralized team.
