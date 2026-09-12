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
    joint, score = search_rows(nets, z, feat, own, TYPES.to(DEV), SearchConfig(depth=0, candidates=16), False)
    b = fuse_table(nets, z, feat)
    rows = torch.arange(3, device=DEV)
    q = nets.critic(b[rows, own], joint[rows, own]).mean(0)
    assert torch.allclose(q, score, atol=1e-4)
    # The chosen action must score at least as well as the mean action.
    q_mean = nets.critic(b[rows, own], nets.actor.mean(b[rows, own])).mean(0)
    assert (q >= q_mean - 1e-5).all()


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
