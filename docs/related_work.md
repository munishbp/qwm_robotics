# Related work

The proposal and the primer cite five papers. This section places each claim of the study
against the closest prior work, so the reader can see what is new and what is a restatement.

## Test time search on a learned model with a Q critic

QWM (Dong et al. 2026) keeps policy and critic trained on real transitions and uses a learned
world model only at decision time. The search of this study is that mechanism with imagined
teammates. Selecting an action by ranking sampled candidates with a Q function, which is the
depth 0 arm here, is QT-Opt (Kalashnikov et al. 2018), which ran a cross entropy search over Q at
every step. Hamrick et al. (2021, "On the role of planning in model based deep reinforcement
learning") found that planning at decision time helps mainly through better action selection when
the value function is informative, and little otherwise. Conclusion 1 of `results.md` is a case
of that finding in a decentralized team. MuZero (Schrittwieser et al. 2020) and TD-MPC2 (Hansen
et al. 2024) plan with learned models but train on imagined or model rolled data, which is the
compounding QWM avoids.

## Off policy learning from demonstrations under sparse reward

RLPD (Ball et al. 2023) is the base learner: SAC with symmetric offline and online sampling,
LayerNorm, a critic ensemble, and a high update ratio. REDQ (Chen et al. 2021) introduced the
random subset target the ensemble uses. The cloning term added here is the TD3+BC regularizer
(Fujimoto and Gu 2021). The out of distribution action problem that made the policy mean bootstrap
collapse is the standard offline RL failure that CQL (Kumar et al. 2020) and IQL (Kostrikov et
al. 2022) address; this study used a SARSA target on the recorded next action instead. Training
the representation with a supervised decoder from privileged state is the asymmetric actor critic
idea (Pinto et al. 2018). The detached SAC actor term follows SAC+AE (Yarats et al. 2021).

## Decentralized execution, communication, and delays

The task is a Dec-POMDP (Oliehoek and Amato 2016). Learned communication between agents that
attend over received messages is CommNet (Sukhbaatar et al. 2016) and TarMAC (Das et al. 2019);
the attention fusion here is of that family with an explicit age feature. MAPPO (Yu et al. 2022)
and QMIX (Rashid et al. 2018) are the standard centralized training, decentralized execution
baselines; this study trains per robot critics on per robot beliefs with a shared team reward
and does not use a centralized critic.

## Imagining teammates

Rolling a teammate's stale latent forward and imagining its action from the shared policy is a
special case of recursive modeling of other agents: I-POMDPs (Gmytrasiewicz and Doshi 2005),
Interactive POMCP planning, and Self Other Modeling (Raileanu et al. 2018), which uses the
agent's own policy to infer a teammate's action. What this study adds is the measurement of how
that imagination degrades with the age of the teammate information inside a decision time
search (H2 and H3), which those works do not measure.

## Cooperative transport

Multi robot object transport by pushing and caging is a classical robotics problem; the scripted
controller here is a force allocation heuristic of that kind and is not a contribution. The
heterogeneous constraint (pushers cannot latch, grippers cannot push hard) is the study's own
construction so that no single type can finish alone.

## What is new, stated plainly

1. The QWM search with imagined teammates, and the measurement that its value falls with
   teammate staleness while the baseline does not (mjlab, `results.md` 13.4).
2. The observation that the same search hurts on a task whose critic cannot rank the policy's own
   candidates, with the critic's action span measured on both tasks.
3. The refutation of leader elected broadcast search against independent search on both tasks.
4. A recipe, with its failure history, for getting an RLPD style learner to train on a sparse
   reward, partially observed, cooperative task within a small budget.

## References

- Ball, Smith, Kostrikov, Levine. Efficient online reinforcement learning with offline data. ICML 2023. arXiv:2302.02948.
- Chen, Wang, Zhou, Ross. Randomized ensembled double Q learning. ICLR 2021.
- Das, Gervet, Romoff, Batra, Parikh, Rabbat, Pineau. TarMAC: targeted multi agent communication. ICML 2019.
- Dong et al. Q world models (QWM). 2026. arXiv:2608.17163.
- Fujimoto, Gu. A minimalist approach to offline reinforcement learning. NeurIPS 2021.
- Gmytrasiewicz, Doshi. A framework for sequential planning in multi agent settings. JAIR 2005.
- Haarnoja, Zhou, Abbeel, Levine. Soft actor critic. ICML 2018.
- Hamrick et al. On the role of planning in model based deep reinforcement learning. ICLR 2021.
- Hansen, Su, Wang. TD-MPC2. ICLR 2024.
- Kalashnikov et al. QT-Opt: scalable deep reinforcement learning for vision based robotic manipulation. CoRL 2018.
- Kostrikov, Nair, Levine. Offline reinforcement learning with implicit Q learning. ICLR 2022.
- Kumar, Zhou, Tucker, Levine. Conservative Q learning for offline reinforcement learning. NeurIPS 2020.
- Oliehoek, Amato. A concise introduction to decentralized POMDPs. Springer 2016.
- Pinto, Andrychowicz, Welinder, Zaremba, Abbeel. Asymmetric actor critic for image based robot learning. RSS 2018.
- Raileanu, Denton, Szlam, Fergus. Modeling others using oneself in multi agent reinforcement learning. ICML 2018.
- Rashid et al. QMIX. ICML 2018.
- Schrittwieser et al. Mastering Atari, Go, chess and shogi by planning with a learned model. Nature 2020.
- Sukhbaatar, Szlam, Fergus. Learning multiagent communication with backpropagation. NeurIPS 2016.
- Yarats, Zhang, Kostrikov, Amos, Pineau, Fergus. Improving sample efficiency in model free reinforcement learning from images. AAAI 2021.
- Yu et al. The surprising effectiveness of PPO in cooperative multi agent games. NeurIPS 2022.
