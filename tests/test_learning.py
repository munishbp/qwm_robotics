"""Unit tests for the buffer, beliefs, and search on synthetic data. No env needed."""

import torch

from swarm.belief import BeliefConfig, MessageTable, beliefs, fuse_table
from swarm.buffer import AGE_MAX, Buffer
from swarm.compute import limit_memory
from swarm.nets import FEAT_DIM, LATENT, LOCAL_DIM, Nets
from swarm.search import SearchConfig, search

limit_memory()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TYPES = torch.tensor([0, 0, 0, 1, 1, 2])
K = 6


def fill(buf: Buffer, table: MessageTable, steps: int, lag: int = 1, done_p: float = 0.05):
    E = buf.E
    ep_start = torch.zeros(E, dtype=torch.long, device=DEV)
    cfg = BeliefConfig(lag=lag)
    for t in range(steps):
        comm = torch.rand(E, K, K, device=DEV) > 0.3
        stamp = table.update(t, ep_start, comm, cfg)
        buf.begin(torch.full((E, K, LOCAL_DIM), float(t), device=DEV), ep_start, stamp)
        done = torch.rand(E, device=DEV) < done_p
        keep = table.stamp.clone()
        nxt = table.update(t + 1, ep_start, comm, cfg)
        table.stamp = keep
        ep_start = torch.where(done, torch.full_like(ep_start, t + 1), ep_start)
        buf.finish(action=torch.full((E, K, 3), float(t), device=DEV), unc=torch.zeros(E, K, device=DEV),
                   reward=done.float(), terminated=done, truncated=torch.zeros_like(done),
                   next_local=torch.full((E, K, LOCAL_DIM), float(t + 1), device=DEV),
                   next_stamp=nxt, target=torch.zeros(E, K, 6, device=DEV))


def test_stack_reads_the_right_rows():
    buf = Buffer(4, 64, TYPES, DEV)
    fill(buf, MessageTable(4, K, DEV), 40, done_p=0.0)
    env = torch.tensor([0, 1], device=DEV)
    row = torch.tensor([20, 5], device=DEV)
    s = buf.stack(env, row)
    assert s.shape == (2, K, 3 * LOCAL_DIM + 3)
    # Frames are t, t-1, t-2 and the previous action is a(t-1).
    assert torch.allclose(s[0, 0, :LOCAL_DIM], torch.full((LOCAL_DIM,), 20.0, device=DEV))
    assert torch.allclose(s[0, 0, LOCAL_DIM:2 * LOCAL_DIM], torch.full((LOCAL_DIM,), 19.0, device=DEV))
    assert torch.allclose(s[0, 0, -3:], torch.full((3,), 19.0, device=DEV))
    n = buf.next_stack(env, row)
    assert torch.allclose(n[0, 0, :LOCAL_DIM], torch.full((LOCAL_DIM,), 21.0, device=DEV))
    assert torch.allclose(n[0, 0, -3:], torch.full((3,), 20.0, device=DEV))


def test_episode_start_repeats_first_frame_and_zero_action():
    buf = Buffer(2, 64, TYPES, DEV)
    fill(buf, MessageTable(2, K, DEV), 30, done_p=0.0)
    # Force an episode start at row 10 for env 0 and check the stack at row 10.
    buf.ep_start[0, 10] = 10
    s = buf.stack(torch.tensor([0], device=DEV), torch.tensor([10], device=DEV))
    assert torch.allclose(s[0, 0, LOCAL_DIM:2 * LOCAL_DIM], torch.full((LOCAL_DIM,), 10.0, device=DEV))
    assert torch.allclose(s[0, 0, -3:], torch.zeros(3, device=DEV))


def test_message_table_lag_and_age_cap():
    table = MessageTable(1, K, DEV)
    ep_start = torch.zeros(1, dtype=torch.long, device=DEV)
    comm = torch.ones(1, K, K, dtype=torch.bool, device=DEV)
    for t in range(20):
        stamp = table.update(t, ep_start, comm, BeliefConfig(lag=2))
    off = ~torch.eye(K, dtype=torch.bool, device=DEV)
    assert (stamp[0][off] == 17).all()
    assert (stamp[0].diagonal() == 19).all()
    blocked = torch.zeros_like(comm)
    for t in range(20, 40):
        stamp = table.update(t, ep_start, blocked, BeliefConfig(lag=2))
    assert (stamp[0][off] == 39 - AGE_MAX).all()


def test_beliefs_gradient_reaches_encoder_only_through_own_encoding():
    nets = Nets().to(DEV)
    buf = Buffer(8, 64, TYPES, DEV)
    fill(buf, MessageTable(8, K, DEV), 40)
    env, row = buf.sample_rows(16)
    out = beliefs(nets, buf, env, row, use_next=False)
    assert out["b"].shape == (16, K, LATENT)
    out["b"].sum().backward()
    assert nets.enc.encoders[0][0].weight.grad.abs().sum() > 0
    assert nets.wm.g[0].weight.grad is None


def test_fuse_table_is_permutation_invariant_over_teammates():
    nets = Nets().to(DEV)
    table = torch.randn(5, K, LATENT, device=DEV)
    feat = torch.rand(5, K, FEAT_DIM, device=DEV)
    b = fuse_table(nets, table, feat)
    perm = torch.tensor([0, 2, 1, 5, 4, 3], device=DEV)  # keeps slot 0 in place
    b2 = fuse_table(nets, table[:, perm], feat[:, perm])
    assert torch.allclose(b[:, 0], b2[:, 0], atol=1e-5)


def test_search_shapes_and_bounds():
    nets = Nets().to(DEV)
    table = torch.randn(4, K, K, LATENT, device=DEV)
    feat = torch.rand(4, K, K, FEAT_DIM, device=DEV)
    unc = torch.rand(4, K, device=DEV)
    for mode in ["independent", "leader", "round_robin"]:
        for depth in [0, 1, 3]:
            unc_table = torch.rand(4, K, K, device=DEV)
            a, stats = search(nets, table, feat, TYPES.to(DEV), unc_table, SearchConfig(depth=depth, mode=mode), 2)
            assert a.shape == (4, K, 3)
            assert (a.abs() <= 1).all()


def test_search_depth_zero_picks_best_root_by_critic():
    torch.manual_seed(0)
    nets = Nets().to(DEV)
    table = torch.randn(3, K, K, LATENT, device=DEV)
    feat = torch.rand(3, K, K, FEAT_DIM, device=DEV)
    unc_table = torch.rand(3, K, K, device=DEV)
    a, _ = search(nets, table, feat, TYPES.to(DEV), unc_table, SearchConfig(depth=0, candidates=0), 0)
    # With no sampled candidates the only root candidate is the mean action.
    b = fuse_table(nets, table[:, 0], feat[:, 0])
    assert torch.allclose(a[:, 0], nets.actor.mean(b)[:, 0], atol=1e-5)


def test_search_scores_the_best_root_by_the_critic_at_depth_zero():
    torch.manual_seed(0)
    nets = Nets().to(DEV)
    from swarm.search import search_rows
    z = torch.randn(3, K, LATENT, device=DEV)
    feat = torch.rand(3, K, FEAT_DIM, device=DEV)
    own = torch.tensor([0, 2, 5], device=DEV)
    joint, score, _ = search_rows(nets, z, feat, own, TYPES.to(DEV), SearchConfig(depth=0, candidates=16), False)
    b = fuse_table(nets, z, feat)
    rows = torch.arange(3, device=DEV)
    q = nets.critic(b[rows, own], joint[rows, own]).mean(0)
    assert torch.allclose(q, score, atol=1e-4)
    # The chosen action must score at least as well as the mean action.
    q_mean = nets.critic(b[rows, own], nets.actor.mean(b[rows, own])).mean(0)
    assert (q >= q_mean - 1e-5).all()


def test_gate_above_every_r_returns_the_fallback_and_searches_nothing():
    torch.manual_seed(0)
    nets = Nets().to(DEV)
    table = torch.randn(3, K, K, LATENT, device=DEV)
    feat = torch.rand(3, K, K, FEAT_DIM, device=DEV)
    unc_table = torch.rand(3, K, K, device=DEV)
    cfg = SearchConfig(depth=2, gate=1e9, gate_fallback="mean")
    a, stats = search(nets, table, feat, TYPES.to(DEV), unc_table, cfg, 0)
    assert stats["searched_fraction"] == 0.0
    b = fuse_table(nets, table.reshape(3 * K, K, LATENT), feat.reshape(3 * K, K, FEAT_DIM)).reshape(3, K, K, LATENT)
    own_b = b.diagonal(dim1=1, dim2=2).transpose(1, 2)
    assert torch.allclose(a, nets.actor.mean(own_b), atol=1e-5)


def test_gate_below_every_r_matches_the_ungated_search_at_depth_zero():
    nets = Nets().to(DEV)
    table = torch.randn(3, K, K, LATENT, device=DEV)
    feat = torch.rand(3, K, K, FEAT_DIM, device=DEV)
    unc_table = torch.rand(3, K, K, device=DEV)
    torch.manual_seed(1)
    a0, _ = search(nets, table, feat, TYPES.to(DEV), unc_table, SearchConfig(depth=0), 0)
    torch.manual_seed(1)
    a1, stats = search(nets, table, feat, TYPES.to(DEV), unc_table, SearchConfig(depth=0, gate=1e-9), 0)
    assert stats["searched_fraction"] == 1.0
    assert torch.allclose(a0, a1)


def test_gate_splits_rows_and_keeps_shapes_at_depth():
    torch.manual_seed(0)
    nets = Nets().to(DEV)
    from swarm.search import search_rows
    z = torch.randn(64, K, LATENT, device=DEV)
    feat = torch.rand(64, K, FEAT_DIM, device=DEV)
    own = torch.arange(64, device=DEV) % K
    # The shuffle keeps the searched fraction, because it permutes r and does not change it.
    went = {}
    for shuffle in (False, True):
        torch.manual_seed(2)
        cfg = SearchConfig(depth=2, gate=_median_r(nets, z, feat, own), gate_shuffle=shuffle)
        joint, score, went[shuffle] = search_rows(nets, z, feat, own, TYPES.to(DEV), cfg, False)
        assert joint.shape == (64, K, 3) and score.shape == (64,)
        assert 0 < went[shuffle].sum() < 64
    assert went[False].sum() == went[True].sum()


def _median_r(nets, z, feat, own) -> float:
    torch.manual_seed(2)
    from swarm.search import _candidates
    b = fuse_table(nets, z, feat)
    rows = torch.arange(z.shape[0], device=DEV)
    joint = _candidates(nets, b, own, 8, False)
    b_own = b[rows, own]
    q = nets.critic(b_own.unsqueeze(1).expand(-1, joint.shape[1], -1), joint[rows, :, own]).mean(0)
    spread = nets.critic(b_own, joint[rows, 0, own]).std(0)
    return ((q.max(1).values - q.min(1).values) / spread).median().item()


def test_lcb_score_is_the_ensemble_mean_minus_the_scaled_spread():
    torch.manual_seed(0)
    nets = Nets().to(DEV)
    from swarm.search import _q
    b, a = torch.randn(7, LATENT, device=DEV), torch.rand(7, 3, device=DEV)
    q = nets.critic(b, a)
    assert torch.allclose(_q(nets, b, a, SearchConfig(lcb=1.5)), q.mean(0) - 1.5 * q.std(0), atol=1e-6)
    assert torch.allclose(_q(nets, b, a, SearchConfig()), q.mean(0))


def test_gate_rejects_a_scorer_that_is_not_the_critic():
    import pytest
    nets = Nets().to(DEV)
    table = torch.randn(2, K, K, LATENT, device=DEV)
    feat = torch.rand(2, K, K, FEAT_DIM, device=DEV)
    with pytest.raises(ValueError, match="scorer q"):
        search(nets, table, feat, TYPES.to(DEV), torch.rand(2, K, K, device=DEV), SearchConfig(scorer="random", gate=1.0), 0)


def test_leader_mode_follows_self_elected_leaders_only():
    torch.manual_seed(0)
    nets = Nets().to(DEV)
    table = torch.randn(2, K, K, LATENT, device=DEV)
    feat = torch.rand(2, K, K, FEAT_DIM, device=DEV)
    # Env 0: everyone elects robot 1. Env 1: everyone elects robot 3 except robot 3, who elects 0.
    unc_table = torch.ones(2, K, K, device=DEV)
    unc_table[0, :, 1] = 0.0
    unc_table[1, :, 3] = 0.0
    unc_table[1, 3, 3] = 0.5
    unc_table[1, 3, 0] = 0.0
    a, stats = search(nets, table, feat, TYPES.to(DEV), unc_table, SearchConfig(depth=1, mode="leader"), 0)
    assert a.shape == (2, K, 3)
    assert stats["searches_per_env"] == 0.5
    assert stats["robots_following_a_leader"] == 0.5
    assert stats["leader_disagreement"] > 0


def test_n_step_window_stops_at_the_episode_end_and_at_the_newest_row():
    from swarm.rlpd import Agent, RLPDConfig
    torch.manual_seed(0)
    buf, steps, n, gamma = Buffer(4, 64, TYPES, DEV), 40, 5, 0.9
    fill(buf, MessageTable(4, K, DEV), steps, done_p=0.15)
    buf.reward.copy_(torch.rand_like(buf.reward))
    agent = Agent(TYPES, RLPDConfig(n_step=n, gamma=gamma), DEV)
    env = torch.arange(4, device=DEV).repeat_interleave(steps - 12)
    row = torch.arange(12, steps, device=DEV).repeat(4)
    buf.sample_rows = lambda _: (env, row)
    d = agent._batch(buf, env.numel())
    for i in range(env.numel()):
        e, t = int(env[i]), int(row[i])
        ret, disc, last = 0.0, 1.0, t
        for m in range(n):
            if t + m > steps - 1 or int(buf.ep_start[e, t + m]) != int(buf.ep_start[e, t]):
                break
            ret, disc, last = ret + disc * float(buf.reward[e, t + m]), disc * gamma, t + m
        assert abs(float(d["r"][i]) - ret) < 1e-5 and abs(float(d["disc"][i]) - disc) < 1e-6
        assert bool(d["term"][i]) == bool(buf.terminated[e, last])
    # One step keeps the old target: the reward of the row and one discount.
    d1 = Agent(TYPES, RLPDConfig(gamma=gamma), DEV)._batch(buf, env.numel())
    assert torch.equal(d1["r"], buf.reward[env, row]) and (d1["disc"] == gamma).all()
