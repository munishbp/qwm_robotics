# Concepts behind this project

This file explains every idea the proposal depends on, in the order you need them. Read it top to
bottom once. Each section builds on the one before it. The last section is a glossary for lookup
later.

The proposal is [proposal_shared_latent_swarm_transport.md](proposal_shared_latent_swarm_transport.md).

---

## 1. Reinforcement learning, the vocabulary

An agent lives in an environment. At every step it sees something, does something, and gets a
number back. The goal is to pick actions that make the sum of those numbers large.

- **State** `s`: everything about the world right now. Payload pose, every robot pose, every
  velocity. The simulator knows the state. The robot usually does not.
- **Observation** `o`: what the agent actually sees. In this project each robot sees a limited
  sector of the payload. The observation is a lossy view of the state.
- **Action** `a`: what the agent does. Here it is a continuous vector, for example wheel torques or
  a gripper force.
- **Reward** `r`: the number the environment hands back after a step. This project uses a sparse
  reward: zero on every step except the one where the payload reaches the goal pose.
- **Episode**: one attempt from reset to termination or a time limit.
- **Return**: the discounted sum of future rewards from a step onward, `r_t + γ r_{t+1} + γ² r_{t+2} + ...`.
  The **discount** `γ` is a number just under 1, usually 0.99. It makes far future rewards count
  less and keeps the sum finite.
- **Policy** `π(a | o)`: the function the agent uses to pick actions. It is a neural network that
  takes an observation and outputs a distribution over actions. Training RL means training this
  network.
- **Transition**: one tuple `(o, a, r, o', done)`. Everything an off policy learner needs about one
  step. The **replay buffer** is a big array of these tuples.

**Why sparse reward is hard.** The agent gets zero signal until it succeeds once by chance. With
dense reward you can shape the agent toward the goal. Sparse reward means you need either luck,
demonstrations, or a lot of samples. The proposal avoids a learned reward model because with sparse
reward the reward is trivially known: one at the goal, zero elsewhere.

---

## 2. Value functions, and why Q matters more than V here

A **value function** predicts the return from some point. There are two.

- **State value** `V(s)`: the expected return if you are in state `s` and follow the policy from
  here.
- **Action value** `Q(s, a)`: the expected return if you are in state `s`, do action `a` now, and
  follow the policy after that.

`Q` answers a question `V` cannot: "which action is better right now?" You can compare `Q(s, a1)`
against `Q(s, a2)` directly. To compare actions with only `V` you need a model to tell you which
state each action leads to, and then you compare the values of those states. That is one extra model
query per candidate and one extra source of error.

This is why the proposal refuses PPO. PPO learns `V`. QWM's appendix shows that search on top of `V`
loses badly to search on top of `Q`. The tree search scores candidate actions, so it wants `Q`.

**The Bellman equation** is the recursion that makes `Q` learnable:

```
Q(s, a) = r + γ · Q(s', a')      where a' is what the policy does next
```

The right side is called the **TD target** (temporal difference target). Training a Q network means
pushing `Q(s, a)` toward that target with a squared error loss, over and over, on transitions from
the buffer. The target uses the network's own estimate of the next step, so errors can feed back
into themselves. Every trick in section 4 exists to keep that feedback loop stable.

---

## 3. On policy versus off policy

**On policy** methods (PPO, GRPO) learn only from data that the current policy generated. After
each update the old data is thrown away. Simple and stable, but every environment step is used
once.

**Off policy** methods (DQN, SAC, RLPD) learn from any transition regardless of which policy
produced it. Old data stays in the replay buffer and gets reused. This has two consequences that
matter for this project.

1. Off policy learners are far more **sample efficient**. They need fewer environment steps. This
   matters because MuJoCo Warp with contact rich multi body scenes does not run as many parallel
   environments as a simple locomotion task would.
2. Off policy learners can train on data from a **different policy** entirely. The scripted
   controller's demonstrations can go into the buffer. The policy network never has to have
   produced them.

The price is stability. Reusing old data and bootstrapping Q targets from the network's own
estimates makes off policy learning easier to break. Section 4 is the standard set of fixes.

---

## 4. SAC, the actor critic that RLPD is built on

**Soft Actor Critic** is the standard continuous action off policy algorithm. Its parts:

- An **actor**: the policy network. Takes an observation, outputs a mean and standard deviation of
  a Gaussian over actions, squashed through `tanh` to keep actions in bounds.
- A **critic**: the Q network. Takes an observation and an action, outputs one number.
- A **target critic**: a slow moving copy of the critic. The TD target is computed with this copy,
  not the live one, so the target does not jump around every update. It is updated as an
  exponential moving average of the live critic, usually with rate 0.005.
- **Twin critics**: two independent Q networks. The TD target uses the minimum of the two. A single
  Q network overestimates, because the policy chases whatever `Q` overestimates, and the
  overestimate gets baked into the target. Taking the minimum of two counters that. This project
found the minimum too pessimistic over a 90 step horizon and uses the mean of two with a clamp
instead (design.md section 6.4).
- **Entropy bonus**: SAC adds a term to the objective that rewards the policy for staying random.
  The weight on that term is the **temperature** `α`, and SAC tunes it automatically to hit a target
  entropy. This keeps exploration alive under sparse reward and stops the policy from collapsing
  onto one action too early.

One SAC update, in order:

1. Sample a batch of transitions from the buffer.
2. Compute the TD target with the target critics and the current policy's next action, minus the
   entropy term.
3. Update both critics toward the target with a squared error loss.
4. Update the actor to maximize `min(Q1, Q2)(o, π(o))` plus entropy.
5. Update `α` toward the target entropy.
6. Move the target critics a small step toward the live critics.

CleanRL's `sac_continuous_action.py` is this list in about 300 lines and is the recommended thing
to read.

---

## 5. RLPD, three changes to SAC

**RLPD** stands for Reinforcement Learning with Prior Data (Ball et al., 2023). Its claim is that
you do not need a special offline RL algorithm to use demonstrations. You need plain SAC plus three
changes that let it absorb offline data and train hard on it without collapsing.

1. **Symmetric sampling.** Every batch is half from the offline buffer and half from the online
   buffer. Not a schedule, not a ratio that decays. Fifty fifty for the whole run. The offline half
   keeps successful trajectories in every batch even when the online policy has not succeeded yet.
   With sparse reward this is what makes the reward signal reach the critic at all.
2. **Layer normalization in the critic.** A LayerNorm after each hidden layer of the Q network.
   The reason is subtle. The critic queries `Q(s', a')` for actions the policy proposes, and early
   on those actions are far from anything in the buffer. An unnormalized MLP can output huge values
   for out of distribution inputs, and those values become TD targets, and the critic diverges.
   LayerNorm bounds the growth and the divergence stops.
3. **Large critic ensemble and a high update to data ratio.** Instead of two critics, use ten.
   Each update samples a random pair from the ten and takes the minimum, which gives a tunable
   amount of pessimism. And instead of one gradient update per environment step, do many.
   The **update to data ratio** (UTD) is that count. RLPD uses 20. High UTD is what turns sample
   efficiency into wall clock efficiency, because environment steps are the scarce thing and
   gradient steps on a 5090 are cheap. LayerNorm and the ensemble are what let UTD go that high
   without the critic blowing up.

Everything else is SAC. When the proposal says "RLPD" it means SAC with these three changes and two
buffers.

**Where the offline buffer comes from here.** There are no human demos. A scripted centralized
controller with full state and known payload inertia drives the team to success. Those transitions
fill the offline buffer. The controller is also a baseline, so it does two jobs.

---

## 6. Model free versus model based, and why QWM is neither in the usual sense

A **world model** is a learned function that predicts the next state from the current state and
action. `f(s, a) → s'`.

**Model free** RL never learns one. SAC and RLPD are model free. They learn `Q` and `π` directly
from transitions.

**Model based** RL learns one and uses it. The classic way (Dyna, Dreamer, TD-MPC2) is to generate
imagined transitions from the model and train the policy and critic on them. That gives you far
more training data per real step. The cost is **compounding error**. The model is wrong by a small
amount per step. Roll it forward ten steps and the imagined trajectory is somewhere the real
environment would never go. Train the critic on that, and the critic believes in states that do not
exist. The policy then exploits the model's mistakes rather than the real physics.

**QWM's move** is to keep the model out of training entirely. Policy and critic train on real
transitions only, exactly like RLPD. The model is used only at **decision time**, to look a few
steps ahead over candidate actions before committing to one. Nothing the model imagines is ever a
training target, so model bias cannot accumulate in the weights. If the model is wrong, one action
choice is slightly worse. Next step the search starts fresh from a real observation.

This is the single most important idea in the project. Everything downstream inherits it.

---

## 7. Test time tree search, what QWM actually does at each step

At every environment step, instead of just running the policy once and acting, QWM does this:

1. **Sample candidates.** Draw `N` actions from the policy at the current state. These are the root
   children.
2. **Imagine.** For each candidate, query the world model to get the predicted next state.
3. **Recurse.** At each predicted next state, sample `N` more actions and predict again. Stop at
   **depth** `D`.
4. **Prune.** The tree grows as `N^D`, which explodes. So after each level keep only the `J` best
   paths, ranked by accumulated discounted Q value along the path. `J` is the **beam width**.
   QWM found results insensitive to `J`, so a narrow beam is fine.
5. **Score.** Each node gets a score that mixes two things: the critic's direct estimate `Q(s, a)`
   at that node, and the value found by rolling the model further from that node. Deeper imagined
   value is damped by a **tree search discount**, a second discount separate from `γ`, that says
   how much to trust what the model says three steps out compared to what the critic says now.
   QWM landed on small values for this discount, meaning the critic dominates and the model
   contributes a correction.
6. **Act.** Pick the root child with the best aggregated score. Execute it in the real environment.
   Throw the tree away.

Search runs during both data collection and evaluation. QWM found the two are complementary.
Searching during collection puts better transitions in the buffer, and searching during evaluation
extracts more from the trained critic.

**Depth is the knob H2 is about.** QWM found moderate depth is best, and attributed the drop at high
depth to compounding model error. This project claims a second error source, and that the second
one dominates in a team.

---

## 8. Partial observability and belief

A **POMDP** (partially observable Markov decision process) is an MDP where the agent sees an
observation, not the state. The right thing to act on is then not the observation but the
**belief**: the agent's best estimate of the state given everything it has seen and received so far.

In this project the belief is a vector, a **latent**, produced by a neural network. It is not a
probability distribution written out. It is a learned summary that the policy, critic, and world
model all consume. Nobody tells the network what the latent should mean. It gets shaped by whatever
losses flow into it, and the RL losses are the ones that matter, so the latent ends up encoding what
is useful for choosing actions and predicting value.

Each robot's belief is built from two sources:

- Its own current observation, encoded fresh. Full weight.
- Latents received from teammates inside communication range, stamped at time `t-1`.

The proposal's whole argument rests on that `t-1`. The teammate information is a frame old before
the robot even starts thinking.

---

## 9. Encoders, latents, and the shared core

An **encoder** is the network that maps a raw observation to a latent. Each robot **type** has its
own encoder, because a scout's observation has a different shape and meaning than a pusher's.

A **decoder** is the reverse map, latent to something interpretable. Here a decoder head recovers
payload and robot state from the latent. It is not used for control. It exists so that prediction
error can be reported in meters and radians instead of in latent units that mean nothing to a
reader.

The **dynamics core** is shared across all types. One network predicts the next latent from the
current latent and the joint action, regardless of which robot type is asking. This is the load
bearing claim of section 4.1: if a pusher and a gripper drive the same model and it works, the model
has learned payload physics rather than per robot quirks.

### 9.1 Why the world model is latent based, not state based

QWM's model predicts raw states. That works because QWM's agent observes the state. Here a robot
never has the state. At search time the only thing it has is its fused belief. So the model must
take a belief in and give a belief out. Input is the fused belief plus the team's joint action.
Output is the next fused belief.

### 9.2 Residual prediction

The model does not output the next latent directly. It outputs the **delta**, `z' - z`, and the
caller adds it to `z`. This is the residual form. It helps because consecutive latents are usually
close, so the delta is small and centered near zero, which is an easier regression target. It also
means an untrained model predicts "nothing changes", which is a much safer default than random
output.

### 9.3 Training the model online, and the stop gradient

The model pretrains in week 4 on the offline buffer passed through the week 3 encoders. That is
only its initialization. From week 5 it keeps training on the same real transitions RLPD samples,
with the same UTD as the critic.

The problem this solves: the encoders keep learning during RL, so the latent space keeps moving. A
model frozen at week 4 would be predicting a latent space that no longer exists. Training the model
online keeps it chasing the current encoder.

The **stop gradient** is the detail that makes this clean. The dynamics loss compares the model's
predicted next latent against the encoder's actual next latent. Without a stop gradient, that loss
would push on the encoder too, and the encoder could cheat by making latents trivially predictable,
for example by collapsing to a constant. With `detach()` on the encoder output inside the dynamics
loss, gradient flows into the model only. The encoder is shaped by the RL losses alone. The model
follows the encoder and never steers it. Week 7 lists removing the stop gradient as an optional
ablation, which is what TD-MPC2 does.

This also preserves QWM's claim. The model trains on real latent transitions. Nothing imagined
enters any loss.

### 9.4 The snapshot

At the end of week 5, the encoders, policy, critic, and model are saved. Every sweep in weeks 6 and
7 loads that snapshot and only changes search settings. This is what lets you say "depth caused
this difference" instead of "the model drifted between runs".

---

## 10. Attention fusion and permutation invariance

Each robot has a fresh local latent and a set of corrected teammate latents. They need to become
one belief. The proposal uses an **attention** layer over the set of agents.

Attention here means: each latent computes a weight for each other latent based on how relevant
they are to each other, and the output is a weighted mix. It is the same mechanism as in a
transformer, applied to a set of at most sixteen vectors rather than a sequence of tokens.

The property that matters is **permutation invariance**. The output must not depend on the order in
which teammates are listed. Attention over a set without positional encodings has this property
automatically. It also does not care how many inputs there are, which is why the same network can
train with 8 robots and be evaluated with 4 or 32 without retraining.

The robot's own latent enters at full weight and is never discounted. Teammate latents are weighted
by attention and additionally by uncertainty.

---

## 11. Forward correction of stale latents

A teammate's latent arrived stamped at `t-1`. Before fusing it, roll it forward one step through the
shared dynamics model. The result is the model's guess at what that teammate's latent is at time
`t`. Everything being fused is then nominally at the same time.

This is a cheap use of the world model and it is the same model the search uses. If the model is
good, the correction removes most of the lag. If the model is bad, the correction adds error on top
of staleness. The zero lag oracle baseline in section 5 of the proposal tells you which.

---

## 12. Ensemble uncertainty and leader election

An **ensemble** is several copies of the same network with different random initializations,
trained on the same data. Where the data is dense they agree. Where the data is thin they disagree.
The spread of their outputs is a usable **uncertainty** estimate that costs no extra machinery.

Each robot emits a scalar uncertainty from its ensemble disagreement. The robot with the lowest
uncertainty is elected **leader** for that step and its latent gets the highest fusion weight.
Election is local: each robot computes the leader from the messages it received. Two robots can
disagree about who the leader is. The proposal accepts that as realistic and measures it.

H4 extends this. Instead of every robot running its own search, only the leader searches and
broadcasts. Same total compute, spent by the robot with the best information.

---

## 13. Multi agent RL, the terms that matter here

- **Centralized**: one controller sees everything and commands every robot. The scripted
  controller is centralized. So is the week 2 baseline, which concatenates all observations into
  one vector.
- **Decentralized**: each robot runs its own copy of the policy on its own belief. There is no
  central node at execution time. This is the target system.
- **Heterogeneous**: robots have different capabilities. Pushers, grippers, one scout. No type can
  finish alone. This forces cooperation instead of allowing it.
- **Joint action**: the concatenation of every robot's action at one step. The world model needs
  the joint action, because the payload moves according to all forces on it, not one robot's.
- **Shared policy**: every robot runs the same network weights. The type specific encoder heads
  differ, but the policy and critic on top of the latent are shared. This is what makes it
  possible for robot `i` to imagine what robot `j` will do: `i` runs the shared policy on its
  estimate of `j`'s belief.
- **Non-stationarity**: from robot `i`'s point of view, the environment includes the other robots,
  and they are changing as they learn. The transition function is not fixed. Off policy methods
  suffer from this because old buffer data was generated by teammates who no longer behave that
  way. The proposal does not attack this directly. It accepts it as part of the harder setting.

---

## 14. The new part: imagining teammates inside the search

This is section 4.3 of the proposal, and it is the only piece with no prior implementation to copy.

In QWM the imagined next state depends only on the action being searched over. In a team the
payload's next state depends on the **joint** action. Robot `i` searches over its own candidates,
but to query the world model it needs the other three actions too. It has to imagine them.

At every step, robot `i`:

1. Samples `N` candidate actions for itself from the shared policy on its own belief.
2. Estimates each teammate's belief. At the root, that estimate is the received `t-1` latent
   rolled forward one step, the same forward correction from section 11.
3. Runs the shared policy on each teammate estimate to get an imagined action per teammate.
4. Assembles the joint action: its own candidate plus the imagined teammate actions.
5. Queries the world model on its own belief and the joint action to get its predicted next
   belief.
6. Also rolls each **teammate estimate** forward through the world model with the same joint
   action, so that at the next level it has a fresh estimate of each teammate to run the policy on.
7. Recurses to depth `D`, prunes to `J` paths, scores with the tree search discount, and acts.
   Same as section 7.

Step 6 is what keeps the H2 experiment clean. Because the teammate estimates are carried forward
level by level with the same model, a teammate estimate at depth 3 is exactly as stale as it was at
the root. The **staleness** parameter means one thing at every depth. Without step 6 you would have
to decide what teammates know about the searcher's imagined future, and there is no good answer.

**Why H2 predicts collapse.** In QWM the error in the tree comes from one source: model error,
which compounds with depth. Here there is a second source: imagined teammate error. Robot `i`'s
guess at what robot `j` does is wrong whenever `j`'s real belief differs from `i`'s estimate of it,
and that difference starts at one frame of lag and grows with every imagined step. The proposal
claims this error dominates, so the depth at which search stops helping arrives sooner, and arrives
sooner still as staleness grows. The experiment is a grid of depth against staleness, evaluated on
one frozen snapshot. Every cell is a test time change. No retraining.

---

## 15. Why almost every experiment is test time only

One training run produces the snapshot. After that, depth, beam width, tree search discount,
staleness, leader election mode, message dropout, team size, and type ratio are all things you set
at evaluation. The trained weights do not change.

This is the practical reason the project fits in 8 weeks on one GPU. It is also the scientific
reason the results are clean. When every cell of a sweep uses identical weights, the only thing
that differs between cells is the setting being swept.

The exceptions, which need a separate trained model:

- Search with `V` instead of `Q` (needs a value network).
- Independent per robot world models (needs models trained without message passing).
- The no stop gradient ablation in week 7.

---

## 16. The simulator

**MuJoCo** is a rigid body physics engine. **MuJoCo Warp** is its GPU port, where many copies of
the scene step in parallel in one kernel launch. **mjlab** is a task framework on top of MuJoCo
Warp with the same manager style API as Isaac Lab: you declare a scene, observation terms, reward
terms, and termination terms, and it gives you a vectorized environment that steps thousands of
copies at once and returns batched tensors.

Two facts about it shape the project:

- Throughput comes from many parallel environments, not from many bodies per environment. A
  contact rich scene with 8 robots and a payload runs fewer copies than a single humanoid would.
  Off policy RL needs fewer environment steps, which offsets this.
- mjlab's own training scripts assume PPO through rsl-rl. There is no off policy loop in the box.
  The RLPD loop must be written against mjlab's vectorized environment directly. That is the week 2
  risk the proposal names.

---

## 17. The four hypotheses in plain words

- **H1.** Search helps at all, in the decentralized team. If false, the project reports a boundary
  of QWM's method and stops there.
- **H2.** The best depth gets shallower as teammate information gets older. This is the headline
  and the reason the project exists.
- **H3.** The tree search discount should also shrink with staleness. Same logic as H2, applied to
  the other knob that controls how much imagined value counts.
- **H4.** One well informed robot searching and broadcasting beats every robot searching alone, at
  equal compute.

H1 must hold for H2 and H3 to be testable. H4 is independent and is the stretch result.

---

## 18. Glossary

| Term | One line meaning |
|---|---|
| Actor | The policy network |
| Beam width `J` | How many paths survive pruning at each tree level |
| Belief | A robot's latent estimate of the world state from its own view plus messages |
| Bellman target, TD target | `r + γ Q(s', a')`, what the critic is trained toward |
| Compounding error | Model error that grows with each imagined step |
| Critic | The Q network |
| Critic ensemble | Many Q networks; the target uses a random pair. RLPD takes the minimum for pessimism; this project takes the mean and clamps the target instead (design.md 6.4) |
| Decoder head | Maps a latent back to state units, for reporting error only |
| Depth `D` | How many imagined steps the search looks ahead |
| Discount `γ` | Per step factor on future reward, about 0.99 |
| Dynamics core | The shared world model that all robot types drive |
| Encoder | Maps a raw observation to a latent; one per robot type |
| Entropy, temperature `α` | SAC's exploration bonus and its auto tuned weight |
| Forward correction | Rolling a `t-1` teammate latent one step ahead through the model |
| Joint action | Every robot's action at one step, concatenated |
| Latent | A learned vector summary; the belief is one |
| LayerNorm | Normalization inside the critic that stops divergence at high UTD |
| Leader | The robot with lowest ensemble uncertainty this step |
| Model free / model based | Whether a learned dynamics model exists and is used |
| Off policy | Learns from any transition, keeps a replay buffer |
| Offline buffer | Scripted controller transitions, sampled 50/50 with online data |
| POMDP | An MDP where the agent sees observations, not the state |
| QWM | Q learning with world models: search at decision time, never train on imagination |
| Residual model | Predicts `z' - z` rather than `z'` |
| RLPD | SAC + symmetric sampling + critic LayerNorm + large ensemble at high UTD |
| SAC | Soft actor critic, the base off policy algorithm |
| Snapshot | The saved weights at the end of week 5 that every sweep runs on |
| Sparse reward | One at the goal, zero everywhere else |
| Staleness | How many frames old the teammate information is; the H2 axis |
| Stop gradient | `detach()` so the dynamics loss trains the model but not the encoder |
| Target network | A slow moving copy of the critic used to compute TD targets |
| Tree search discount | Damps deep imagined value relative to the direct critic estimate |
| UTD ratio | Gradient updates per environment step; RLPD uses about 20 |
| World model | `f(z, joint action) → z'`, the learned dynamics |
| Zero lag oracle | Baseline with instant messages; the gap to it is what H2 measures |

---

## 19. What to read next, in order

1. RLPD paper, sections 3 and 4 only: https://arxiv.org/abs/2302.02948
2. CleanRL `sac_continuous_action.py` with the RLPD paper open beside it:
   https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py
3. QWM paper, method section and the depth ablation: https://arxiv.org/abs/2608.17163
4. The official RLPD JAX code for the ensemble and sampling details:
   https://github.com/ikostrikov/rlpd
