# Mathematics of the system

This document states every equation the project uses and the reason each one exists.

It builds on [concepts.md](../concepts.md), which explains the ideas in plain language, and on [docs/design.md](design.md),
which fixes every constant. The design is the source of truth. Where the proposal and the design differ, the design wins, and
section 8 of the design gives the reason.

Every displayed equation carries a number, and later documents cite these numbers. Section 12 lists the two questions the
design still leaves open. This document decides neither of them.

## 1. Notation

| Symbol | Meaning | Value in the design |
|---|---|---|
| $E$ | Parallel environments | 512 train, 256 evaluate |
| $K$ | Robots on the team | 6 |
| $i, j, k$ | Robot indices. $i$ searches, $j$ is a teammate | |
| $\tau_k$ | Type of robot $k$: pusher, gripper, or scout | one hot, length 3 |
| $t$, $T$, $\Delta t$ | Step index, episode length, time step | $T = 150$, $\Delta t = 0.1$ s |
| $s_t$ | Full environment state | equation (1) |
| $o_{k,t}$ | Local observation of robot $k$ | $\mathbb{R}^{16}$ |
| $o^{\text{full}}_{k,t}$ | Centralized observation, oracle baseline only | $\mathbb{R}^{16+4+3K}$ |
| $\mathbf{o}_t$ | Joint observation $(o_{1,t}, \dots, o_{K,t})$ | |
| $a_{k,t}$ | Action of robot $k$, written $(v_x, v_y, u)$ | $[-1,1]^3$ |
| $\mathbf{a}_t$ | Joint action $(a_{1,t}, \dots, a_{K,t})$ | $[-1,1]^{3K}$ |
| $e_{k,t}$ | Fresh latent from robot $k$'s own observation | $\mathbb{R}^{64}$ |
| $\hat{z}_j$ | Robot $i$'s rolled forward estimate of teammate $j$ | $\mathbb{R}^{64}$ |
| $b_{k,t}$ | Fused belief of robot $k$ | $\mathbb{R}^{64}$ |
| $r_t$ | Shared team reward | 1 on success, else 0 |
| $\gamma$ | Reward discount | 0.99 |
| $\alpha$, $\bar{\mathcal{H}}$ | Entropy temperature and target entropy | $\alpha$ learned, $\bar{\mathcal{H}} = -3$ |
| $\theta, \phi, \bar\phi, \psi$ | Actor, critic, target critic, world model parameters | Polyak rate $\rho = 0.005$ |
| $M$ | Critic ensemble size | 10 |
| $\lambda_{\text{bc}}$ | Behavior cloning weight on the offline half | 1.0 |
| $B$, $G$ | Batch rows, updates per batched env step | 256, 4 |
| $L$, $L_{\max}$ | Staleness of a teammate message, and the cap on it | $L_{\text{train}} = 1$, swept 0, 1, 2, 4; $L_{\max} = 8$ |
| $D$, $d$ | Search depth, and the depth index $0 \le d \le D$ | swept $-1$, 0, 1, 2, 4, 6 |
| $N$, $J$ | Candidates per node, beam width | 8, 4 |
| $\beta$ | Tree search discount | 0.5 default, swept 0.0 to 1.0 |
| $\nu_i$ | Scalar uncertainty of robot $i$ | equation (28) |
| $A$, $h_x$, $h_y$, $R_c$ | Arena half size, payload half extents, comm radius | 5.0, 0.8, 0.4, 4.0 m |

## 2. The task as a Dec POMDP

### 2.1 Tuple, state, observation, reward

$$\mathcal{M} = \langle K, \mathcal{S}, \{\mathcal{A}_k\}, P, R, \{\Omega_k\}, O, \gamma, T \rangle, \qquad s_t = \big(p_t,\ g,\ \{x_{k,t}\},\ \{\ell_{k,t}\},\ \{q_k\}_{k \in \text{grip}},\ t\big) \tag{1}$$

This is the form of Oliehoek and Amato (2016), with $p_t$ the payload pose, $g$ the goal pose, $x_{k,t}$ a robot position,
$\ell_{k,t}$ a latch flag, and $q_k$ a latch anchor in the payload frame. The anchor is state because the design fixes the
latch point in the payload frame at the moment the gripper latches.

$$o_{k,t} = O_k(s_t) \in \mathbb{R}^{16}, \qquad c_{ij,t} = \mathbb{1}\big[\, i \ne j,\ \|x_{i,t} - x_{j,t}\| < R_c,\ \neg\,\text{occluded}(i,j,s_t)\,\big] \tag{2}$$

The observation map is deterministic and lossy, and design section 3.5 lists its 16 fields. The payload enters the observation
only inside the sensing range of the type, 2.0 m for a pusher and a gripper and 8.0 m for a scout, and $c_{ij,t}$ decides which
messages robot $i$ receives.

$$R(s_t, \mathbf{a}_t) = \mathbb{1}\big[\|(p_x,p_y) - (g_x,g_y)\| < 0.3 \ \wedge\ |\mathrm{wrap}(p_\theta - g_\theta)| < 0.2\big], \qquad \max_\pi\ \mathbb{E}\Big[\textstyle\sum_t \gamma^t R(s_t, \mathbf{a}_t)\Big] \tag{3}$$

The reward is sparse and shared, so one number arrives on the success step and the episode terminates. Every robot runs the
same weights $\pi$ on its own belief, so the maximization is over one parameter set and not $K$ sets. The
design sets $\gamma = 0.99$.

### 2.2 Payload dynamics

Design section 3.3 fixes a quasi static model with a Coulomb style threshold. Write $n_\ell$ for the number of latched
grippers, $\Pi(x)$ for the closest payload boundary point to $x$, and $\hat{n}(x)$ for the inward unit normal there.

$$F_k = \begin{cases} u_k F_p \hat{n}(x_k) & \tau_k = \text{pusher},\ \|x_k - \Pi(x_k)\| \le d_c,\ u_k > 0 \\ F_g\,\mathrm{clip}(v_{x,k}, v_{y,k}) & \tau_k = \text{gripper, latched} \\ 0 & \text{otherwise}\end{cases} \qquad F_p = 1.0,\ F_g = 0.3,\ d_c = 0.25 \tag{4}$$

A pusher cannot choose a direction, because it pushes along the inward normal. A latched gripper chooses any direction but is
weak, with $\|F_k\| \le F_g\sqrt{2} \approx 0.424$ N.

$$F_s(n_\ell) = \max\big(F_0 - \Delta F\, n_\ell,\ F_{\min}\big), \qquad \tau_s(n_\ell) = 0.5\,F_s(n_\ell)\,h_x, \qquad F_0 = 3.5,\ \Delta F = 1.0,\ F_{\min} = 0.5 \tag{5}$$

Every latched gripper lowers the translation threshold by 1.0 N, so a gripper is a threshold reducer first and a force source
second. The rotation threshold follows the translation threshold, so one latch unlocks both by the same factor.

$$F_{\text{net}} = \sum_k F_k, \quad \tau_{\text{net}} = \sum_k (r_k \times F_k), \quad v = \frac{\max(\|F_{\text{net}}\| - F_s, 0)}{c_t}\frac{F_{\text{net}}}{\|F_{\text{net}}\|}, \quad \omega = \frac{\max(|\tau_{\text{net}}| - \tau_s, 0)}{c_r}\mathrm{sign}(\tau_{\text{net}}), \quad p_{t+1} = p_t + (v_x, v_y, \omega)\Delta t \tag{6}$$

Here $r_k = \Pi(x_k) - (p_x, p_y)$ and $c_t = c_r = 2.0$, and the payload responds to the sum of forces and the sum of moments,
which is why the world model takes the joint action. The $\max(\cdot, 0)$ is the whole design: below the threshold the payload
does not move and the reward stays zero, so the task is hard because the team must cross a threshold together.

### 2.3 The three friction facts

Each follows from equations (4) to (6) by arithmetic on the design's constants.

**Fact 1. Pushers alone fail.** Three pushers latch nothing, so $n_\ell = 0$ and $F_s = 3.5$, and each contributes at most
$1.0$ N along a unit normal.

$$\|F_{\text{net}}\| \le 3 \times 1.0 = 3.0 \ <\ 3.5 = F_s(0) \quad \implies \quad v = 0 \tag{7}$$

The bound is tight, because three pushers reach 3.0 N exactly and still fall 0.5 N short. A fourth pusher would break the task
design, which is why the default team has three.

**Fact 2. Grippers alone fail.** Two latched grippers give $F_s = 3.5 - 2.0 = 1.5$, and each contributes at most $0.424$ N.

$$\|F_{\text{net}}\| \le 2 \times 0.424 = 0.849 \ <\ 1.5 = F_s(2) \quad \implies \quad v = 0 \tag{8}$$

The grippers lower the threshold to 1.5 N and then cannot reach it. The design writes this case as $0.6 < 1.5$ using unit
command magnitude, and equation (8) uses the maximum magnitude, so the conclusion survives the worst case.

**Fact 3. The mixed team succeeds.** Two latched grippers plus two pushers, with $F_s(2) = 1.5$.

$$\|F_{\text{net}}\| = 2 \times 1.0 + 2 \times 0.3 = 2.6 \ >\ 1.5, \qquad v = \frac{2.6 - 1.5}{2.0} = 0.55\ \text{m/s} \tag{9}$$

The pushers alone already clear the reduced threshold, since $2.0 > 1.5$. Each gripper contributes 1.0 N of threshold
reduction and only 0.3 N of force, so the reduction is worth more than three times the force, and that ratio is the
quantitative form of "no type finishes alone".

**One robot of any type fails.** A pusher gives $1.0 < 3.5$, a gripper gives $n_\ell = 1$ and $0.424 < 2.5$, and a scout
applies no force. This is the week 1 negative control, and it holds by construction, so it confirms the task design only.

**Feasibility of the full team.** Three pushers and two latched grippers give $F_s(2) = 1.5$ and $\tau_s(2) = 0.6$ N m.

$$v = \frac{3.6 - 1.5}{2.0} = 1.05\ \text{m/s}, \qquad |\tau_{\text{net}}| \le 3(1.0 \times 0.5 h_x) + 2\big(F_g\sqrt{h_x^2 + h_y^2}\big) = 1.737, \qquad \omega = \frac{1.737 - 0.6}{2.0} = 0.568\ \text{rad/s} \tag{10}$$

Translation crosses the 2.5 m to 4.0 m gap in 24 to 39 of the 150 steps, so it has a wide margin. Rotation is tighter, and the
worst case really occurs: design section 3.1 samples the goal orientation up to 90 degrees from the start orientation, while
two pushers alone give $\omega = 0.1$ rad/s and need about 157 steps for that angle, which exceeds the 150 step episode. The
team must commit most of its pushers to torque when the angle error is large, and that is why design section 4 makes the
scripted controller spread the pushers along the face near the goal.

## 3. Soft actor critic

RLPD is SAC with three changes. This section states SAC and section 4 states the changes. The critic is per robot on its own
belief $b$ and its own action $a$, as design section 8 fixes, so teammates are part of the environment for the critic.

$$J(\pi) = \mathbb{E}_\pi\Big[\textstyle\sum_t \gamma^t\big(r_t + \alpha\,\mathcal{H}(\pi(\cdot \mid b_t))\big)\Big], \qquad \mathcal{H}(\pi(\cdot \mid b)) = -\mathbb{E}_{a \sim \pi}\big[\log \pi(a \mid b)\big] \tag{11}$$

The entropy term pays the policy to stay random. Under a sparse reward the policy gets no gradient from the reward until it
succeeds once, so without this term it collapses early and never finds the goal.

$$\mathcal{T}^\pi Q(b,a) = r + \gamma\,\mathbb{E}_{b', a' \sim \pi}\big[Q(b',a') - \alpha \log \pi(a' \mid b')\big], \qquad y = \mathrm{clip}\Big(r + \gamma\,(1 - \text{terminated})\,\tilde{Q}_{\bar\phi}(b',a'),\ 0,\ 1\Big) \tag{12}$$

The left side is the soft Bellman operator of SAC. The right side is the target this project uses, and it leaves the
entropy term out. This is RLPD's `backup_entropy = False` setting, and it exists for a sparse terminal reward: the reward is
at most 1 once per episode, while the entropy stream pays about $\alpha \mathcal{H} / (1 - \gamma)$ per episode, so a
critic that backs up the entropy learns that ending the episode is a loss and the actor learns to stall. The first training
run of this project showed exactly that: $Q$ near 20 and lower on the transitions closest to success. With the entropy out
of the backup the critic estimates the discounted success probability, and the entropy still acts on the actor through
equation (15). Only `terminated` zeroes the bootstrap, because a truncated episode still has a future, and equation (19)
defines $\tilde{Q}_{\bar\phi}$.

The target is also clamped to $[0,1]$. The reward of equation (3) is one terminal unit, so every true value lies in that
range, which makes the clip exact knowledge of the task and not a heuristic. Without it the minimum of two noisy heads in
equation (19) biases each target low by about $0.56$ times the head spread, because the mean of the minimum of two
independent draws sits $\sigma/\sqrt{\pi}$ below their own mean, and the bootstrap compounds that constant offset to
$0.56\,\sigma/(1-\gamma)$; at the spread of 0.005 the training run showed, this predicts $-0.28$, against the $Q$ near $-0.3$
the run reached. The clip does not move the fixed point when the heads agree, because the minimum then equals the value and
every true value already lies inside the range.

$$\mathcal{L}_Q(\phi) = \frac{1}{M}\sum_{m=1}^{M}\mathbb{E}_{\mathcal{D}}\Big[\big(Q_{\phi,m}(b,a) - y\big)^2\Big] \tag{13}$$

Every head trains on the same target and differs only by its initialization. Their disagreement therefore measures how much
the data constrains the value at that input, which is what equation (28) needs.

$$a_\theta(b,\xi) = \tanh(u),\ \ u = \mu_\theta(b) + \sigma_\theta(b)\odot\xi,\ \ \xi \sim \mathcal{N}(0,I); \qquad \log \pi(a \mid b) = \log \mathcal{N}(u; \mu_\theta, \sigma_\theta) - \sum_{n=1}^{3}\log\big(1 - \tanh^2 u_n\big) \tag{14}$$

The reparameterization moves the randomness into $\xi$ so gradient passes through $\mu$ and $\sigma$, and the $\tanh$ keeps
the action in $[-1,1]^3$. The correction term is $\log|\det \partial a/\partial u|$ for a diagonal Jacobian with entries
$1 - \tanh^2 u_n$; without it the entropy belongs to the pre squash Gaussian, the temperature loss chases the wrong number,
and $\alpha$ drifts. Use $\log(1 - \tanh^2 u) = 2(\log 2 - u - \mathrm{softplus}(-2u))$, because $1 - \tanh^2 u$ underflows
for $|u| > 10$.

$$\mathcal{L}_\pi(\theta) = \underbrace{\mathbb{E}_{b,\xi}\left[\alpha \log \pi_\theta\big(a_\theta(b,\xi) \mid b\big) - \frac{1}{M}\sum_{m=1}^{M} Q_{\phi,m}\big(b, a_\theta(b,\xi)\big)\right]}_{\text{SAC}} \ +\ \underbrace{\lambda_{\text{bc}}\,\mathbb{E}_{\mathcal{D}_{\text{off}}}\Big[\big\|\mu^{\tanh}_\theta(b) - a_{\text{data}}\big\|_2^2\Big]}_{\text{behavior cloning}}, \qquad \mu^{\tanh}_\theta(b) = \tanh\big(\mu_\theta(b)\big), \qquad \lambda_{\text{bc}} = 1.0 \tag{15}$$

Design section 6.4 fixes the first term to the mean over all $M$ heads and not a minimum, because the actor should climb the
ensemble's best estimate while the pessimism stays in the target of equation (12), where a bootstrap can compound.

The second term is behavior cloning on the offline half of every batch, against the scripted action $a_{\text{data}}$ recorded
in that row. It exists because under the sparse reward the critic stays flat in the action: after 8,000 updates the value at
the data action, at the policy mean, at zero, and at a random action all sat at 0.54, so the first term carried no gradient in
$a$ and the actor stayed near zero velocity. The term hands the actor the scripted behavior while the critic keeps learning
the value that the search of section 9 needs.

This is a deliberate deviation from RLPD, in the style of TD3+BC. It acts on the actor alone, and it never enters the critic
target of equation (12), the critic loss of equation (13), or the world model loss of equation (22), so the pessimism of
equation (19) and the QWM claim of section 5 are both untouched.

$$\mathcal{L}_\alpha = \mathbb{E}_{b, a \sim \pi_\theta}\big[-\alpha\big(\log \pi_\theta(a \mid b) + \bar{\mathcal{H}}\big)\big], \qquad \bar{\mathcal{H}} = -3 \tag{16}$$

The gradient is $-(\log \pi + \bar{\mathcal{H}})$, so when the policy entropy falls below $-\bar{\mathcal{H}}$ the temperature
rises and pays the policy to spread out again. The design sets $\bar{\mathcal{H}} = -3$, the standard choice of minus the
action dimension, and starts $\alpha$ at 0.1.

## 4. RLPD

RLPD is Ball et al. (2023). It makes three changes to SAC and adds a second buffer. Equation (15) adds a fourth change that
RLPD itself does not have.

$$\mathcal{D} = \tfrac12 \mathcal{D}_{\text{off}} + \tfrac12 \mathcal{D}_{\text{on}}, \qquad \mathbb{E}_{\mathcal{D}}[f] = \tfrac12 \mathbb{E}_{\mathcal{D}_{\text{off}}}[f] + \tfrac12 \mathbb{E}_{\mathcal{D}_{\text{on}}}[f] \tag{17}$$

Symmetric sampling is a fixed mixture over two buffers, and every expectation in section 3 is taken under it. Each update
draws $B/2 = 128$ rows from each buffer for the whole run, which is what puts a nonzero reward in every batch while the online
policy still succeeds almost never.

$$\mathrm{LN}(h) = g \odot \hat{h} + \beta_{\text{LN}}, \quad \hat{h}_n = \frac{h_n - \bar{h}}{\sqrt{\mathrm{Var}(h) + \epsilon}} \quad \implies \quad \|\hat{h}\|_2 = \sqrt{H}, \quad \|\mathrm{LN}(h)\|_2 \le \|g\|_\infty\sqrt{H} + \|\beta_{\text{LN}}\|_2 \tag{18}$$

The normalized vector has a fixed norm, because $\frac1H\sum_n \hat{h}_n^2 = 1$ forces $\|\hat{h}\|_2 = \sqrt{H}$ exactly, so
the layer output is bounded for every input. This is why LayerNorm stops value extrapolation: an unnormalized ReLU MLP is
positively homogeneous, so a far out of distribution input produces a far out of distribution value, that value becomes a
target through equation (12), and the critic diverges. The bound is exact, and the claim that it stops divergence at a high
update ratio is the empirical result of the RLPD paper.

$$\tilde{Q}_{\bar\phi}(b',a') = \min_{m \in \{m_1, m_2\}} Q_{\bar\phi,m}(b',a'), \qquad m_1, m_2 \sim \mathrm{Uniform}\{1,\dots,M\},\ \ m_1 \ne m_2 \tag{19}$$

Design section 6.4 fixes $M = 10$ heads with batched weights, draws two head indices uniformly without replacement, and takes
their minimum, exactly as in the RLPD paper. The minimum of two draws is a biased low estimate, and that bias is the pessimism
that counters the overestimation equation (12) would otherwise compound, because the actor chases whatever the critic
overestimates and the overestimate returns as a target. Ten heads make the amount of pessimism tunable, since a pair drawn
from a large ensemble is milder than the minimum over all of it, and equation (15) keeps the actor on the mean over all heads
so the policy climbs the best estimate and not the pessimistic one. The clip in equation (12) bounds the pessimism, so the
downward bias corrects one target and does not accumulate across updates.

$$\text{rows sampled per collected row} = \frac{G \cdot B}{E} = \frac{4 \times 256}{256} = 4 \tag{20}$$

RLPD quotes an update to data ratio of 20 for a single environment, which is not comparable here because one batched step
collects 256 transitions at once. Equation (20) is the comparable quantity, and the world model receives the same update count
as the critic. The design fixes Adam with learning rate $3 \times 10^{-4}$ for every module, $B = 256$ rows per update, and
the Polyak rate $\rho = 0.005$ on the target critic of equation (12).

## 5. Latent world model

$$f_\psi(z, a_{\text{self}}, c) = z + g_\psi(z, a_{\text{self}}, c), \qquad c_i = \frac{1}{|\mathcal{N}_i|}\sum_{j \in \mathcal{N}_i} h_\psi(\tau_j, a_j) \tag{21}$$

The model outputs a delta and the caller adds it, so the target is small and centred near zero and an untrained model predicts
"nothing changes". The mean over teammates makes the action context invariant to teammate order and defined for any team size,
and the type enters $h_\psi$ because the same command numbers produce different forces for a pusher and a gripper.

$$\mathcal{L}_{\text{wm}}(\psi) = \mathbb{E}_{\mathcal{D}}\Big[\big\|f_\psi\big(\mathrm{sg}(e_t), a_t, \mathrm{sg}(c_t)\big) - \mathrm{sg}(e_{t+1})\big\|_2^2\Big], \qquad \mathcal{L}_{\text{dec}} = \mathbb{E}_{\mathcal{D}}\Big[\big\|\mathrm{dec}(\mathrm{sg}(z)) - y_{\text{state}}\big\|_2^2\Big] \tag{22}$$

The stop gradient $\mathrm{sg}$ is `detach()`, and it sits on both sides, so gradient reaches $\psi$ only. Without it the loss
would push on the encoder, and the encoder has a trivial way to lower it: collapse every latent to a constant, which is
perfectly predictable and useless for control. The decoder target $y_{\text{state}} \in \mathbb{R}^6$ is the payload pose
relative to the robot plus the latch and visible flags, and it carries a stop gradient too, so it reads the latent and never
shapes it.

**Why the stop gradient keeps QWM's claim intact.** The right hand side of the first loss is $e_{t+1}$, the encoding of a
**real** observation from the buffer, so no imagined quantity appears in any loss. Section 9 uses the model at decision time
only and discards the tree after the action, so model bias never enters the weights of the policy or the critic. If the model
is wrong, one action choice is slightly worse and the next step starts from a real observation again.

$$\hat{z}^{(m+1)} = f_\psi\big(\hat{z}^{(m)}, a_{t+m}, c_{t+m}\big),\ \ \hat{z}^{(0)} = e_t; \qquad \mathcal{E}_k = \mathbb{E}\big\|\hat{z}^{(k)} - e_{t+k}\big\|_2^2, \qquad \mathcal{S}_k = A\cdot\mathbb{E}\big\|\mathrm{dec}_{0:2}(\hat{z}^{(k)}) - \mathrm{dec}_{0:2}(e_{t+k})\big\|_2 \tag{23}$$

This is the $k$ step open loop error that week 4 plots, and it uses the **recorded** joint actions from the buffer, which
isolates model error from policy error. The factor $A = 5.0$ m undoes the arena normalization, so $\mathcal{S}_k$ reports
metres for the reader while $\mathcal{E}_k$ reports the loss; both are computed on held out transitions.

## 6. Messages, staleness, and forward correction

$$m_j = \big(e_{j,t_j},\ a_{j,t_j},\ \nu_{j,t_j},\ \tau_j\big), \qquad L_j = \min\big(t - t_j,\ L_{\max}\big), \qquad L_{\max} = 8 \tag{24}$$

A message carries the sender's latent, the action the sender took, the sender's uncertainty, and the sender's type. The design
initializes every table entry at reset with the step 0 encoding, so no entry is ever missing and no code path handles a null
message. A message sent at $t - L$ arrives when the link of equation (2) is up at $t$ and it survives the dropout draw, a
robot that hears nothing keeps its older entry, and the cap $L_{\max} = 8$ models a channel with a bounded delay.

$$\hat{z}_j^{(m+1)} = \begin{cases} f_\psi\big(e_{j,t_j},\ a_{j,t_j},\ c^{\text{tab}}_{i}\big) & m = 0 \ \text{(recorded action)} \\[4pt] f_\psi\Big(\hat{z}_j^{(m)},\ \mu_\theta\big(\mathrm{fuse}(\hat{z}_j^{(m)})\big),\ c^{\text{tab}}_{i}\Big) & m \ge 1 \ \text{(imagined action)}\end{cases}, \quad \hat{z}_j = \hat{z}_j^{(L_j)}, \quad c^{\text{tab}}_{i} = \frac{1}{|\mathcal{N}_i| - 1}\sum_{j' \in \mathcal{N}_i \setminus \{j\}} h_\psi\big(\tau_{j'},\ a_{j', t_{j'}}\big) \tag{25}$$

Step one knows what the sender did, because the sender put it in the message. After that robot $i$ guesses with the shared
policy mean on the **self only belief** of the estimate, which is the fusion of equations (26) and (27) with no teammate entries. The
context $c^{\text{tab}}_i$ pools the recorded actions already in robot $i$'s own table and stays fixed for the whole roll,
because the receiver has no newer information and never sees the sender's table. The design states that this is an
approximation and that the search of section 9 makes the same one, so training and execution share one code path and one
source of error.

Equation (25) is the project's central quantity. Robot $i$'s estimate of robot $j$ receives no new information after step
$t_j$, so every step substitutes a guess for an observation, and section 10 turns this into the statement of H2.

## 7. Attention fusion

$$\varphi_i = [e_i;\ 0;\ \tau_i], \quad \varphi_j = [\hat{z}_j;\ L_j / L_{\max};\ \tau_j]; \qquad q = W_q e_i,\ k_s = W_k \varphi_s,\ v_s = W_v \varphi_s, \qquad w_s = \frac{\exp(\langle q, k_s\rangle/\sqrt{d_k})}{\sum_{s'}\exp(\langle q, k_{s'}\rangle/\sqrt{d_k})}, \qquad \mathrm{att}_i = \sum_{s \in \{i\} \cup \mathcal{N}_i} w_s v_s \tag{26}$$

Four extra values ride along with each latent: the age fraction $L_j/L_{\max}$ and the type one hot, so the layer can learn to
discount an old estimate instead of the designer fixing a weight by hand. The uncertainty $\nu_j$ is **not** a fusion feature
and serves only the election of equation (28), because the offline buffer has no critic, so a stored uncertainty would mark
the data source inside every batch and the fusion layer could read it. This is scaled dot product attention from Vaswani et
al. (2017) over a set of at most 16 vectors, the query comes from the own encoding only, and the design uses 4 heads.

$$b_i = \mathrm{att}_i + \mathrm{MLP}(\mathrm{att}_i) \tag{27}$$

The residual MLP adds a nonlinearity without breaking the identity path. At initialization its output is near zero, so the
belief starts as the attention output, and the own encoding dominates because $q$ and $k_i$ come from the same vector.

**Claim.** For any bijection $\sigma: \mathcal{N}_i \to \mathcal{N}_i$, reordering the teammate list leaves $b_i$ unchanged.

**Proof.** The query $q$ depends on $e_i$ alone, so it is unchanged. Each key and each value is a function of $\varphi_s$
alone, with no positional term added, so the pair $(k_s, v_s)$ travels with its element under $\sigma$. The softmax
denominator is a sum over the set $\{i\} \cup \mathcal{N}_i$ and addition is commutative, so it is unchanged, and each
numerator depends only on $q$ and $k_s$, so each weight $w_s$ travels with its element. The output $\sum_s w_s v_s$ is again a
sum over the same set, so $\mathrm{att}_i$ is unchanged, and equation (27) makes $b_i$ a function of $\mathrm{att}_i$ alone.
For the multi head version the argument applies per head, and the concatenation and output projection act after the sum.
$\blacksquare$

The load bearing condition is that no positional encoding appears in $\varphi_s$; with one, the values would depend on slot
index and the proof would fail. The consequence is that nothing in equations (26) and (27) depends on $|\mathcal{N}_i|$, so
the design trains at $K = 6$ and evaluates at other team sizes and type ratios with no retraining. Equation (21) is invariant
by the same argument, with a mean in place of the softmax weighted sum.

## 8. Critic ensemble uncertainty and leader election

$$\nu_i = \sqrt{\frac{1}{M}\sum_{m=1}^{M}\big(Q_{\phi,m}(b_i, \mu_\theta(b_i)) - \bar{Q}\big)^2}; \qquad U_i[j] = \begin{cases}\nu_i & j = i \ \ \text{(fresh)} \\ \nu_{j, t_j} & j \ne i \ \ (L_j \text{ steps stale})\end{cases}; \qquad \ell_i = \arg\min_{j \in \{i\} \cup \mathcal{N}_i} U_i[j] \tag{28}$$

The heads share a target and differ by initialization, so they agree where the buffer constrains the value and disagree where
it does not, which makes the spread an uncertainty estimate that costs no extra network. The evaluation point is the mean
action and not a sample, so $\nu_i$ measures uncertainty about the belief and not the randomness of the policy.

Every robot elects from its own uncertainty row $U_i$, which holds its own entry fresh and every teammate's entry as of the
message of equation (24). There is no consensus step, so the rows disagree, and `leader` mode resolves each robot's action
from its own row alone.

| Condition on robot $i$ | What robot $i$ does |
|---|---|
| $\ell_i = i$ | runs one joint search over $NK$ sampled candidates per node and broadcasts the joint action |
| $\ell_i = l \ne i$ and $\ell_l = l$ | executes slot $i$ of leader $l$'s broadcast |
| $\ell_i = l \ne i$ and $\ell_l \ne l$ | falls back to its own mean action $\mu_\theta(b_i)$ |

Two failure cases follow from the staleness and the design accepts both. No robot in an env elects itself, in which case
nobody searches and the whole team runs the depth $-1$ policy for that step. Or a robot elects itself and broadcasts while no
other robot elected it, in which case the search cost is paid and nobody follows. The evaluation reports the share of robots
following a leader and the number of searches per env, so both cases are visible, along with the leader disagreement rate
$\frac{1}{K}\sum_i \mathbb{1}[\ell_i \ne \ell^{\text{fresh}}]$, where $\ell^{\text{fresh}}$ is the election under every
robot's current uncertainty. A consensus protocol would need a round trip that the staleness model does not have.

## 9. Test time search

### 9.1 The procedure

This is design section 7, written as an algorithm. It runs batched over every environment and every searching robot at once.

```
procedure SEARCH(robot i, env e, depth D, width N, beam J, discount beta)
  z[i] <- e_i                                        # fresh own encoding
  for j in N_i: z[j] <- ROLL_FORWARD(m_j, L_j)       # equation (25)
  b    <- FUSE(z[i], {z[j]})                         # equations (26), (27)

  paths <- empty                                     # root: score before any roll
  for a_own in { a1..aN ~ pi(. | b) } U { mu(b) }:   # N + 1 root candidates
      for j in N_i: a[j] <- mu( FUSE(z[j], others) ) # imagined teammate actions
      paths.append( path(a_root=a_own, z=z, b=b, a_joint=(a_own,{a[j]}),
                         score=Q_bar(b, a_own)) )    # Q_0, no model call

  for d = 1 .. D:
      for p in paths:                                # roll EVERY surviving path
          for k in {i} U N_i:
              p.z[k] <- f(p.z[k], p.a_joint[k], c(p.a_joint))
          p.b     <- FUSE(p.z[i], {p.z[j]})          # age features unchanged
          p.score <- p.score + beta^d * Q_bar(p.b, mu(p.b))
      paths <- TOP_J(paths, key=score)               # prune AFTER scoring
      if d = D: break
      children <- empty                              # expand every survivor
      for p in paths:
          for a_own in { a1..aN ~ pi(. | p.b) } U { mu(p.b) }:
              for j in N_i: a[j] <- mu( FUSE(p.z[j], others) )
              children.append( copy of p with a_joint=(a_own,{a[j]}) )
      paths <- children

  for a in A_root:
      S(a) <- max{ p.score : p.a_root = a } / sum_{d=0..D} beta^d
  return argmax_a S(a)
```

The order is roll, score, prune, expand. Every candidate is rolled before it is scored, so no path is discarded on a score
the world model never checked, and the beam keeps $J$ paths only after each level is scored. Expansion adds the mean joint
action plus $N$ sampled own actions, so every survivor has $N+1$ children at every level and not $N$.

The teammate imagination step is the piece with no prior implementation. In QWM the imagined next state depends only on the
action under search, but here the payload obeys the joint action by equation (6), so robot $i$ must imagine the other $K-1$
actions before it can query the model at all.

Rolling every teammate estimate forward with the same joint action keeps the H2 experiment clean. A teammate estimate at depth
3 has received exactly as much new information as it had at the root, which is none, and its age feature in equation (26)
stays at the root value, so its staleness is identical at every level and the axis $L$ means one thing across the whole grid. Without this step the design would have to say what a teammate
knows about the searcher's imagined future, and there is no correct answer to that question.

### 9.2 The score and the beam

$$S(a) = \frac{Q_0 + \sum_{d=1}^{D}\beta^d Q_d}{\sum_{d=0}^{D}\beta^d}, \quad Q_0 = \bar{Q}(b, a), \quad Q_d = \bar{Q}\big(b^{(d)}, \mu_\theta(b^{(d)})\big); \qquad \text{score}^{(d')}(a) = Q_0 + \sum_{d=1}^{d'}\beta^d Q_d \tag{29}$$

$Q_0$ reads the critic at the real root belief with the candidate action, and each $Q_d$ reads it at the rolled belief with
the policy mean there, so the numerator mixes what the critic says now with what the critic says at imagined states while the
denominator keeps $S$ on the scale of a $Q$ value. The beam ranks partial paths by $\text{score}^{(d')}$, which is valid
because the denominator is the same constant for every path at a given level, and pruning to $J$ after scoring is what stops
the tree growing as $(N+1)^{D}$. Here $\beta$ is a second discount that damps trust in the model rather than reward, with
default 0.5 and a sweep over 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, and 1.0.

$$D = 0 \quad \text{or} \quad \beta = 0 \qquad \implies \qquad S(a) = \frac{\beta^0 Q_0}{\beta^0} = \bar{Q}(b,a) \qquad \implies \qquad \text{zero calls to } f_\psi \tag{30}$$

With $D = 0$ both sums hold one term, and with $\beta = 0$ and the convention $0^0 = 1$ every term with $d \ge 1$ vanishes, so
the robot acts by $\arg\max_{a \in A_{\text{root}}}\bar{Q}(b,a)$ over the $N+1$ policy samples and never queries the model.
This is **not** the plain policy, because the critic still re-ranks the samples. The design names three distinct settings at
the shallow end and the sweeps report all three.

| Setting | What the robot does | Calls to $f_\psi$ |
|---|---|---|
| depth $-1$ | acts with the mean action $\mu_\theta(b)$; this is the no search baseline | 0 |
| $D = 0$, or any $D$ with $\beta = 0$ | critic argmax over the $N+1$ root candidates | 0 |
| $D \ge 1$ with $\beta > 0$ | the full tree search of section 9.1 | equation (31) |

The gap from depth $-1$ to $D = 0$ measures what the critic adds by re-ranking policy samples, and the gap from $D = 0$ to
$D \ge 1$ measures what the world model adds. Keeping the two gaps apart is what makes the H2 and H3 sweeps honest, because
the $D = 0$ column and the $\beta = 0$ column are then the same critic argmax, computed by the same code on the same
snapshot.

### 9.3 Compute per decision step

$$p(D) = \begin{cases}0 & D = 0 \\ (N+1)\big[1 + (D-1)J\big] & D \ge 1\end{cases}; \qquad \text{per robot per env: } K\,p(D) \text{ model calls}, \quad (N+1) + p(D) \text{ critic ensemble reads} \tag{31}$$

$p(D)$ counts the paths that are rolled: $N+1$ at depth 1, then $J(N+1)$ at every deeper level, because the beam holds $J$
survivors and each expands into $N+1$ children. Each rolled path advances all $K$ slots, which gives $(N+1)K$ model calls at
depth 1 and $J(N+1)K$ at each deeper level, matching design section 7. The root scoring adds $N+1$ critic reads with no model
call, and the root forward correction of equation (25) adds $L(K-1)$ model calls per robot.

$$\text{model calls}_{\text{indep}} = K^2 p(D), \quad \text{model calls}_{\text{leader}} = K\,p_{\text{leader}}(D), \quad p_{\text{leader}}(D) = (NK+1)\big[1 + (D-1)J\big] \quad \implies \quad \frac{\text{leader}}{\text{indep}} = \frac{NK+1}{K(N+1)} = \frac{NK+1}{NK+K} \tag{32}$$

In `independent` mode all $K$ robots search, so the cost is quadratic in team size, which is the overhead QWM names as a
limitation multiplied by $K$. Because every candidate is rolled, one joint search with $NK$ samples per node costs the same as
$K$ independent searches up to the $+1$ and $+K$ terms, and the ratio is independent of depth. With $K=6, N=8, J=4$ it is
$49/54 = 0.907$ at every depth, and at $D = 2$ the counts are $36 \times 45 = 1620$ model calls per environment per step
against $6 \times 245 = 1470$; report measured wall clock too, because equation (32) counts calls and not kernel launches.

## 10. The two error sources in the tree

This section is the mathematical statement of H2 and H3. One result is a bound under a stated assumption, and the rest is a
heuristic that is labelled as one.

### 10.1 Source one: model error

$$\varepsilon_1 = \sqrt{\mathcal{E}_1} = \mathbb{E}\big\|f_\psi(e_t, a_t, c_t) - e_{t+1}\big\|_2 \tag{33}$$

This is the one step latent prediction error of equation (23). It is the only error source QWM faces, and the one QWM blames
for the falloff at high depth.

**Assumption A (Lipschitz model).** There exists $\lambda \ge 0$ with
$\|f_\psi(z,a,c) - f_\psi(z',a,c)\| \le \lambda\|z - z'\|$ for all $z, z'$ and all $a, c$.

$$\delta_{d+1} \le \lambda\delta_d + \varepsilon_1 \qquad \implies \qquad \delta_d \le \varepsilon_1\sum_{m=0}^{d-1}\lambda^m = \varepsilon_1 \cdot \begin{cases} d & \lambda = 1 \\[2pt] \dfrac{\lambda^d - 1}{\lambda - 1} & \lambda \ne 1\end{cases} \tag{34}$$

Write $\delta_d = \|\hat{z}^{(d)} - e_{t+d}\|$; one step of equation (23) adds a fresh $\varepsilon_1$ and propagates the
existing gap through $f_\psi$, and unrolling gives the bound. Equation (34) is a real bound under Assumption A: model error
grows linearly with depth when the model is non expansive and geometrically when it is not. Measure $\lambda$ rather than
assume it, though the residual form of equation (21) makes $\lambda$ close to 1 by construction.

### 10.2 Source two: teammate imagination error

$$\Delta_j^{(d)} = \big\|\hat{z}_j^{(d)} - b_{j, t+d}\big\|, \qquad \Delta_j^{(0)}(L) \approx \underbrace{\varepsilon_1\sum_{m=0}^{L-1}\lambda^m}_{\text{model error in the correction}} + \underbrace{\kappa_{\text{obs}} L}_{\text{observations } j \text{ made and } i \text{ never saw}} \tag{35}$$

This is the gap between robot $i$'s estimate of teammate $j$ and teammate $j$'s real belief, and unlike $\delta_d$ it is not
zero at the root. The first term is equation (34) applied to the roll forward of equation (25), the second is the information
gap that equation (25) fills with guesses, $\kappa_{\text{obs}}$ is a heuristic stand in for how much one fresh observation
moves a belief, and the argument needs only that both terms grow with $L$.

$$\big\|\mu_\theta(\mathrm{fuse}(\hat{z}_j^{(d)})) - a_{j,t+d}\big\| \le \kappa_\pi \Delta_j^{(d)}, \qquad \big\|f_\psi(z,a,c) - f_\psi(z,a,\tilde{c})\big\| \le \frac{\kappa_c\,\mathrm{Lip}(h_\psi)}{|\mathcal{N}_i|}\sum_j \kappa_\pi \Delta_j^{(d)} \tag{36}$$

A gap in beliefs becomes a gap in actions through the shared policy, with $\kappa_\pi = \mathrm{Lip}(\mu_\theta \circ
\mathrm{fuse})$. That gap in actions becomes a gap in the predicted payload motion through the action context of equation
(21), which enters the model.

### 10.3 How the two sources grow with depth and staleness

$$\delta_{d+1} \le \lambda\delta_d + \varepsilon_1 + \kappa\bar{\Delta}^{(d)}, \quad \kappa = \kappa_c \mathrm{Lip}(h_\psi)\kappa_\pi; \qquad \bar{\Delta}^{(d)} \ge \bar{\Delta}^{(0)}(L) \ \implies\ \delta_d \le \big(\varepsilon_1 + \kappa\bar{\Delta}^{(0)}(L)\big)\sum_{m=0}^{d-1}\lambda^m \tag{37}$$

Each step of the tree now injects two errors and not one, with $\bar{\Delta}^{(d)}$ the mean teammate gap at depth $d$. Robot
$i$'s estimate of a teammate receives no new information at any depth and its age feature stays at the root value, by design,
so the teammate gap does not shrink as the tree deepens and the conservative reading holds. Read the right side against equation (34): the two have the same shape in
$d$ and differ only in the per step injection, because QWM injects $\varepsilon_1$ and this project injects
$\varepsilon_1 + \kappa\bar{\Delta}^{(0)}(L)$, a coefficient that grows with $L$ by equation (35).

$$D^*(L) \approx \frac{\eta}{\varepsilon_1 + \kappa\,\bar{\Delta}^{(0)}(L)} \tag{38}$$

Search stops helping at the depth where imagined value in equation (29) becomes less accurate than the root critic estimate;
call it $D^*$, let $\eta$ be the error budget at that crossover, and solve equation (37) in the linear case $\lambda = 1$.
$D^*(L)$ is decreasing in $L$, which is H2, and it is smaller than QWM's $D^* \approx \eta/\varepsilon_1$ at every $L \ge 0$,
which is the claim that useful depth collapses faster in a team.

**Which parts are which.** Equation (34) is a bound, valid under Assumption A. Equations (35) to (38) are a **heuristic**, for
three reasons: the constants $\kappa_{\text{obs}}$, $\kappa_\pi$, and $\kappa_c$ are not measured, the fusion layer composed
with a tanh policy has no verified Lipschitz constant, and the step from a latent error $\delta_d$ to a loss in success rate
is not modelled at all. The claim $\kappa\bar{\Delta}^{(0)}(L) > \varepsilon_1$, meaning teammate error dominates model error,
is an empirical claim and is exactly what the H2 grid tests; a flat $D^*(L)$ refutes the heuristic and the project reports it.

### 10.4 The same argument for the tree search discount, which is H3

$$\big|Q_d - Q_d^{\text{true}}\big| \le \kappa_Q \delta_d \quad \implies \quad \mathrm{bias}(S) \le \kappa_Q \frac{\sum_{d=0}^{D}\beta^d \delta_d}{\sum_{d=0}^{D}\beta^d}, \qquad \text{therefore } \beta^*(L) \text{ is non increasing in } L \tag{39}$$

The score of equation (29) is a weighted average of $D+1$ critic readings, each carrying a bias that grows with the latent
error at that depth, with $\kappa_Q = \mathrm{Lip}(\bar{Q})$. Here $\delta_0 = 0$ because the root belief comes from a real
observation, and every deeper $\delta_d$ is positive and increasing by equation (37), so the bound is increasing in $\beta$.
Raising $\beta$ shifts weight onto the more biased readings, lowering it discards the lookahead the search exists to provide,
and equation (37) says $\delta_d$ grows faster when $L$ grows, which moves the balance point down. That is H3, and it rests on
the same heuristic as H2 and inherits the same caveats.

## 11. The hypotheses as measurable statements

$$\hat{p} = \frac{1}{R}\sum_{r=1}^{R}\hat{p}_r, \quad \mathrm{se}(\hat{p}) = \frac{1}{\sqrt{R}}\sqrt{\frac{1}{R-1}\sum_r (\hat{p}_r - \hat{p})^2}, \quad R \ge 3; \qquad \hat{d} = \hat{p}_A - \hat{p}_B, \quad \mathrm{se}(\hat{d}) = \sqrt{\mathrm{se}(\hat{p}_A)^2 + \mathrm{se}(\hat{p}_B)^2} \tag{40}$$

The design's protocol fixes evaluation at 256 environments and at least 3 batches, with $\hat{p}_r$ the success rate over the
256 environments of batch $r$. The standard error is taken over batch means and not over the environments inside one batch,
because those share the loaded snapshot and the process state. A difference counts when $|\hat{d}| > 2\,\mathrm{se}(\hat{d})$,
and all runs use seed 0, so every cell of every sweep loads the identical week 5 snapshot.

### 11.1 H1. Search helps at all

| Item | Statement |
|---|---|
| Quantity | $\hat{p}(D, L=1)$, the mean success rate of equation (40) |
| Comparison | $\hat{d} = \hat{p}(D=2) - \hat{p}(D=0)$, both at $L = 1$, same snapshot, with depth $-1$ reported beside both |
| Script | `scripts/evaluate.py --depth 0` and `--depth 2`, output `results/h1.json` |
| Confirms | $\hat{d} > 2\,\mathrm{se}(\hat{d})$ |
| Refutes | $\hat{d} \le 2\,\mathrm{se}(\hat{d})$, including any negative $\hat{d}$ |

By equation (30) the $D = 0$ arm calls the world model zero times, so this comparison isolates the world model and nothing
else. It does not isolate search against the plain policy, because $D = 0$ already re-ranks the policy samples with the
critic, which is why depth $-1$ goes in the same table. If H1 fails, H2 and H3 are untestable and the project reports a
boundary of QWM's mechanism under decentralization. The zero lag oracle baseline separates the two causes, because search that
helps at $L=0$ and not at $L=1$ points at staleness and not at a broken model.

### 11.2 H2. The best depth shrinks as teammate information gets staler

| Item | Statement |
|---|---|
| Quantity | $D^*(L) = \arg\max_{D \ge 0} \hat{p}(D, L)$ over $D \in \{0,1,2,4,6\}$, with depth $-1$ as the reference row |
| Comparison | $D^*(L)$ across $L \in \{0,1,2,4\}$, at the default $\beta = 0.5$ |
| Script | `scripts/sweep.py --grid depth x lag`, output `results/h2.json` |
| Confirms | $D^*(L)$ non increasing in $L$, $D^*(L_{\max}) < D^*(0)$, and at the smallest $L$ the peak beats $D=0$ by more than $2\,\mathrm{se}$ |
| Refutes | $D^*(L)$ flat across $L$, or increasing with $L$ |

This is equation (38) read off a grid, and every cell is a test time evaluation on one snapshot, so no cell can differ because
of training. The requirement that the peak beat $D=0$ exists so that $D^*(L)$ marks a real peak and not the argmax of noise,
and the depth $-1$ row anchors the whole grid against the plain policy.

### 11.3 H3. The tree search discount shrinks as teammate information gets staler

| Item | Statement |
|---|---|
| Quantity | $\beta^*(L) = \arg\max_\beta \hat{p}(\beta, L)$ at fixed $D$ |
| Comparison | $\beta^*(L)$ across $L \in \{0,1,2,4\}$, with $\beta \in \{0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0\}$ |
| Script | `scripts/sweep.py --grid beta x lag`, output `results/h3.json` |
| Confirms | $\beta^*(L)$ non increasing in $L$, and $\beta^*(L_{\max}) < \beta^*(0)$ |
| Refutes | $\beta^*(L)$ flat or increasing with $L$ |

This is equation (39) read off a grid. Run one check first: the $\beta = 0$ column must reproduce the $D = 0$ column of H2 to
within $2\,\mathrm{se}$, by equation (30). If it does not, the search code has a defect and neither grid is
interpretable. The sweep includes $\beta = 0$ as its consistency cell, so the check is a cell of the grid and not a separate
run, and by equation (30) that cell makes zero world model calls.

### 11.4 H4. Leader elected search beats independent search at matched compute

| Item | Statement |
|---|---|
| Quantity | $\hat{p}$ under `independent`, `leader`, and `round_robin` |
| Compute match | equation (32) gives a ratio of $0.907$ at every depth, plus measured wall clock per decision step |
| Also report | leader disagreement rate, searches per env, share of robots following a leader |
| Comparison | $\hat{d}_1 = \hat{p}(\text{leader}) - \hat{p}(\text{independent})$, $\hat{d}_2 = \hat{p}(\text{leader}) - \hat{p}(\text{round\_robin})$ |
| Script | `scripts/sweep.py --leader`, output `results/h4.json` |
| Confirms | $\hat{d}_1 > 2\,\mathrm{se}(\hat{d}_1)$ **and** $\hat{d}_2 > 2\,\mathrm{se}(\hat{d}_2)$ |
| Refutes | $\hat{d}_1 \le 2\,\mathrm{se}(\hat{d}_1)$ |

Both differences must hold, because they answer different questions: $\hat{d}_1$ asks whether concentrating the compute budget
in one searcher beats spreading it, and $\hat{d}_2$ asks whether the uncertainty election of equation (28) beats picking the
leader by $t \bmod K$. A result with $\hat{d}_1 > 0$ and $\hat{d}_2 \approx 0$ says the joint candidate search does the work
and the election contributes nothing, which is a weaker claim than H4.

The three extra metrics exist because election runs on stale rows and can fail in two ways, by the table in section 8. When no
robot in an env elects itself, the searches per env drop to zero and that env runs the depth $-1$ policy for the step. When a
robot elects itself and no other robot elected it, the share following a leader drops while the search cost is still paid.
Read $\hat{d}_1$ against these two numbers, because a `leader` arm that loses while few robots follow a leader is a failure of
the election and not of the joint search.

## 12. Open questions

The design fixes every constant this document uses. The H1 protocol runs depth $-1$, $0$, and $2$
at lag 1, so both no search arms are reported. The decoder target divides the relative payload
position by $A$, so equation (23) reports metres. No item remains open.

## 13. References

- Ball, P. J., Smith, L., Kostrikov, I., and Levine, S. (2023). *Efficient Online Reinforcement Learning with Offline Data*. arXiv:2302.02948. Source for equations (17) to (20).
- Haarnoja, T., Zhou, A., Abbeel, P., and Levine, S. (2018). *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*. arXiv:1801.01290. Source for equations (11) to (16).
- Dong, Y. et al. (2026). *Q-learning with World Models (QWM)*. arXiv:2608.17163. Source for the search of section 9 and the depth ablation that H2 extends.
- Vaswani, A. et al. (2017). *Attention Is All You Need*. arXiv:1706.03762. Source for equations (26) and (27).
- Fujimoto, S. and Gu, S. S. (2021). *A Minimalist Approach to Offline Reinforcement Learning*. arXiv:2106.06860. Source for the behavior cloning term in equation (15).
- Oliehoek, F. A. and Amato, C. (2016). *A Concise Introduction to Decentralized POMDPs*. Springer. Source for equation (1).
