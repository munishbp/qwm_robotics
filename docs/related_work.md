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

## Delayed and stale information in multi agent control

Delayed communication is a classic setting in networked control and in multi agent
reinforcement learning with communication delays; this study does not survey that literature and
cites none of it, which is a gap a paper would have to close. What the study measures is specific
to the decision time search: the age of a teammate's latent at the root of the tree.

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

## Multi step targets, pessimism, and deciding when to search

Multi step returns are standard (Sutton and Barto 2018) and are one of the components of Rainbow
(Hessel et al. 2018). The n step target here is that idea restricted to recorded rows, so it stays
a SARSA target. It is not a contribution. The finding is that it decides whether test time search
helps at all on this task. Scoring by the ensemble mean minus a multiple of the ensemble spread is
the lower confidence bound that offline methods use inside training (An et al. 2021). Here it is
used only at test time, inside the argmax of the search. Deciding per step whether to spend
computation follows adaptive computation time (Graves 2016) and metacontrol of imagination
(Hamrick et al. 2017). The gate here needs no training: it reads the critic ensemble that RLPD
already provides. The prompt for the gate was a text decision model that pairs a fast typed answer
with an act or escalate head (Laya model card, 2026). No part of that model is used.

## What is new, stated plainly

1. The QWM search with imagined teammates, and the measurement that its value falls with
   teammate staleness while the baseline does not (mjlab, `results.md` 13.4).
2. The observation that the same search does not help on the quasi static task in three seeds
   and a momentum variant, with three candidate explanations (demonstration quality, the critic's
   action span, momentum) measured and eliminated. A five step critic target then repairs the
   search on three 2D seeds, which places the cause in the one step target (`results.md` 22).
3. The refutation of leader elected broadcast search against independent search on both tasks.
4. A recipe, with its failure history, for getting an RLPD style learner to train on a sparse
   reward, partially observed, cooperative task within a small budget.

5. Two test time settings for a critic guided search: a pessimistic score that removes most of
   the ranking noise at no cost, and a gate on the critic's own span that keeps the success of
   the search with 11 to 35 percent of the searches, each with a shuffled control.

## References

- An, Moon, Kim, Song. Uncertainty based offline reinforcement learning with diversified Q ensemble. NeurIPS 2021.
- Ball, Smith, Kostrikov, Levine. Efficient online reinforcement learning with offline data. ICML 2023. arXiv:2302.02948.
- Chen, Wang, Zhou, Ross. Randomized ensembled double Q learning. ICLR 2021.
- Das, Gervet, Romoff, Batra, Parikh, Rabbat, Pineau. TarMAC: targeted multi agent communication. ICML 2019.
- Dong et al. Q world models (QWM). 2026. arXiv:2608.17163.
- Fujimoto, Gu. A minimalist approach to offline reinforcement learning. NeurIPS 2021.
- Gmytrasiewicz, Doshi. A framework for sequential planning in multi agent settings. JAIR 2005.
- Graves. Adaptive computation time for recurrent neural networks. 2016. arXiv:1603.08983.
- Haarnoja, Zhou, Abbeel, Levine. Soft actor critic. ICML 2018.
- Hamrick, Ballard, Pascanu, Vinyals, Heess, Battaglia. Metacontrol for adaptive imagination based optimization. ICLR 2017.
- Hamrick et al. On the role of planning in model based deep reinforcement learning. ICLR 2021.
- Hansen, Su, Wang. TD-MPC2. ICLR 2024.
- Hessel et al. Rainbow: combining improvements in deep reinforcement learning. AAAI 2018.
- Kalashnikov et al. QT-Opt: scalable deep reinforcement learning for vision based robotic manipulation. CoRL 2018.
- Kostrikov, Nair, Levine. Offline reinforcement learning with implicit Q learning. ICLR 2022.
- Kumar, Zhou, Tucker, Levine. Conservative Q learning for offline reinforcement learning. NeurIPS 2020.
- Laya model card. Convai Innovations, 2026. https://huggingface.co/convaiinnovations/laya
- Oliehoek, Amato. A concise introduction to decentralized POMDPs. Springer 2016.
- Pinto, Andrychowicz, Welinder, Zaremba, Abbeel. Asymmetric actor critic for image based robot learning. RSS 2018.
- Raileanu, Denton, Szlam, Fergus. Modeling others using oneself in multi agent reinforcement learning. ICML 2018.
- Rashid et al. QMIX. ICML 2018.
- Schrittwieser et al. Mastering Atari, Go, chess and shogi by planning with a learned model. Nature 2020.
- Sukhbaatar, Szlam, Fergus. Learning multiagent communication with backpropagation. NeurIPS 2016.
- Sutton, Barto. Reinforcement learning: an introduction, second edition. MIT Press 2018.
- Yarats, Zhang, Kostrikov, Amos, Pineau, Fergus. Improving sample efficiency in model free reinforcement learning from images. AAAI 2021.
- Yu et al. The surprising effectiveness of PPO in cooperative multi agent games. NeurIPS 2022.
