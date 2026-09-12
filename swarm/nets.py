"""Neural network modules. Every latent lives in one space of size LATENT.

The modules follow docs/design.md section 6. Shapes use a leading batch of any rank, written as
`[..., n]`. K is the team size, M the critic ensemble size.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT = 64
STACK = 3
ACT_DIM = 3
NUM_TYPES = 3
LOCAL_DIM = 16
ENC_IN = STACK * LOCAL_DIM + ACT_DIM
FEAT_DIM = 2 + NUM_TYPES  # age, uncertainty, type one hot
HIDDEN = 256
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def mlp(in_dim: int, out_dim: int, hidden: int = HIDDEN, layers: int = 2) -> nn.Sequential:
    mods: list[nn.Module] = []
    d = in_dim
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    mods.append(nn.Linear(d, out_dim))
    return nn.Sequential(*mods)


class TypeEncoders(nn.Module):
    """One MLP encoder per robot type. The output space is shared."""

    def __init__(self, in_dim: int = ENC_IN) -> None:
        super().__init__()
        self.encoders = nn.ModuleList([mlp(in_dim, LATENT) for _ in range(NUM_TYPES)])

    def forward(self, obs_stack: torch.Tensor, types: torch.Tensor) -> torch.Tensor:
        # Run every encoder and select by type. K is small, so the wasted compute is cheap and the
        # code stays free of per type scatter logic.
        outs = torch.stack([enc(obs_stack) for enc in self.encoders], dim=-2)  # [..., T, LATENT]
        idx = types.long().unsqueeze(-1).unsqueeze(-1).expand(*types.shape, 1, LATENT)
        return outs.gather(-2, idx).squeeze(-2)


class Fusion(nn.Module):
    """Attention over the own encoding and the teammate estimates.

    The own encoding is the query. Keys and values are the own encoding and every estimate, each
    concatenated with its feature vector (age, uncertainty, type). The result is permutation
    invariant over the estimates because attention weights depend on content only.
    """

    def __init__(self, heads: int = 4) -> None:
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(LATENT, LATENT)
        self.kv = nn.Linear(LATENT + FEAT_DIM, 2 * LATENT)
        self.out = nn.Linear(LATENT, LATENT)
        self.norm1 = nn.LayerNorm(LATENT)
        self.norm2 = nn.LayerNorm(LATENT)
        self.ff = mlp(LATENT, LATENT, hidden=HIDDEN, layers=1)

    def forward(
        self,
        own: torch.Tensor,
        own_feat: torch.Tensor,
        est: torch.Tensor,
        est_feat: torch.Tensor,
        est_mask: torch.Tensor,
    ) -> torch.Tensor:
        """own [..., L], own_feat [..., F], est [..., n, L], est_feat [..., n, F], est_mask [..., n] bool.

        A masked estimate is ignored. Passing n = 0 gives the self only belief.
        """
        tokens = torch.cat([own.unsqueeze(-2), est], dim=-2)
        feats = torch.cat([own_feat.unsqueeze(-2), est_feat], dim=-2)
        own_mask = torch.ones(*est_mask.shape[:-1], 1, dtype=torch.bool, device=own.device)
        mask = torch.cat([own_mask, est_mask], dim=-1)
        k, v = self.kv(torch.cat([tokens, feats], dim=-1)).chunk(2, dim=-1)
        q = self.q(own).unsqueeze(-2)
        h = self.heads
        dh = LATENT // h
        # Split heads: [..., n+1, h, dh] -> [..., h, n+1, dh].
        k = k.unflatten(-1, (h, dh)).transpose(-3, -2)
        v = v.unflatten(-1, (h, dh)).transpose(-3, -2)
        q = q.unflatten(-1, (h, dh)).transpose(-3, -2)
        logits = (q @ k.transpose(-1, -2)) / math.sqrt(dh)  # [..., h, 1, n+1]
        logits = logits.masked_fill(~mask.unsqueeze(-2).unsqueeze(-2), float("-inf"))
        attn = logits.softmax(-1) @ v  # [..., h, 1, dh]
        attn = attn.transpose(-3, -2).flatten(-2)  # [..., 1, L]
        x = self.norm1(own + self.out(attn.squeeze(-2)))
        return self.norm2(x + self.ff(x))


class Actor(nn.Module):
    """Tanh Gaussian policy over the belief."""

    def __init__(self) -> None:
        super().__init__()
        self.net = mlp(LATENT, 2 * ACT_DIM)

    def dist_params(self, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, log_std = self.net(b).chunk(2, dim=-1)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mu, log_std

    def mean(self, b: torch.Tensor) -> torch.Tensor:
        mu, _ = self.dist_params(b)
        return torch.tanh(mu)

    def sample(self, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reparameterized sample and its log probability under the tanh squash."""
        mu, log_std = self.dist_params(b)
        std = log_std.exp()
        u = mu + std * torch.randn_like(mu)
        a = torch.tanh(u)
        logp = (-0.5 * ((u - mu) / std) ** 2 - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
        # Change of variables for the tanh squash, in the numerically stable form.
        logp = logp - (2 * (math.log(2) - u - F.softplus(-2 * u))).sum(-1)
        return a, logp

    def sample_n(self, b: torch.Tensor, n: int) -> torch.Tensor:
        """n samples per belief: [..., n, ACT_DIM]."""
        mu, log_std = self.dist_params(b)
        u = mu.unsqueeze(-2) + log_std.exp().unsqueeze(-2) * torch.randn(
            *mu.shape[:-1], n, ACT_DIM, device=mu.device
        )
        return torch.tanh(u)


class EnsembleCritic(nn.Module):
    """M critic heads with batched weights, LayerNorm after each hidden layer.

    One forward pass evaluates every head. The LayerNorm follows RLPD: it keeps the critic from
    extrapolating to large values on actions outside the data.
    """

    def __init__(self, num_heads: int = 10) -> None:
        super().__init__()
        self.m = num_heads
        dims = [LATENT + ACT_DIM, HIDDEN, HIDDEN, 1]
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.ln_w = nn.ParameterList()
        self.ln_b = nn.ParameterList()
        for i in range(len(dims) - 1):
            w = torch.empty(num_heads, dims[i], dims[i + 1])
            bound = 1 / math.sqrt(dims[i])
            nn.init.uniform_(w, -bound, bound)
            self.weights.append(nn.Parameter(w))
            self.biases.append(nn.Parameter(torch.zeros(num_heads, 1, dims[i + 1])))
            if i < len(dims) - 2:
                self.ln_w.append(nn.Parameter(torch.ones(num_heads, 1, dims[i + 1])))
                self.ln_b.append(nn.Parameter(torch.zeros(num_heads, 1, dims[i + 1])))

    def forward(self, b: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Returns [M, ...] values for belief [..., L] and action [..., A]."""
        x = torch.cat([b, a], dim=-1)
        lead = x.shape[:-1]
        x = x.reshape(1, -1, x.shape[-1]).expand(self.m, -1, -1)
        n = len(self.weights)
        for i in range(n):
            x = torch.baddbmm(self.biases[i], x, self.weights[i])
            if i < n - 1:
                x = F.layer_norm(x, (x.shape[-1],)) * self.ln_w[i] + self.ln_b[i]
                x = F.relu(x)
        return x.squeeze(-1).reshape(self.m, *lead)


class WorldModel(nn.Module):
    """Residual latent dynamics with a permutation invariant teammate action context."""

    def __init__(self) -> None:
        super().__init__()
        self.g = mlp(LATENT + ACT_DIM + LATENT, LATENT)
        self.h = mlp(NUM_TYPES + ACT_DIM, LATENT, hidden=128, layers=1)

    def context(self, types: torch.Tensor, actions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean of h over the masked teammates. types [..., n], actions [..., n, A], mask [..., n]."""
        one_hot = F.one_hot(types.long(), NUM_TYPES).to(actions.dtype)
        hv = self.h(torch.cat([one_hot, actions], dim=-1)) * mask.unsqueeze(-1).to(actions.dtype)
        count = mask.sum(-1, keepdim=True).clamp(min=1).to(actions.dtype)
        return hv.sum(-2) / count

    def forward(self, z: torch.Tensor, a_self: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        return z + self.g(torch.cat([z, a_self, ctx], dim=-1))


class Decoder(nn.Module):
    """Maps a latent to state units for error reporting. It never shapes the encoder."""

    def __init__(self, out_dim: int = 6) -> None:
        super().__init__()
        self.net = mlp(LATENT, out_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Nets(nn.Module):
    """All learned modules in one container so checkpoints and optimizers stay simple."""

    def __init__(self, num_critics: int = 10, enc_in: int = ENC_IN) -> None:
        super().__init__()
        self.enc = TypeEncoders(enc_in)
        self.fuse = Fusion()
        self.actor = Actor()
        self.critic = EnsembleCritic(num_critics)
        self.critic_target = EnsembleCritic(num_critics)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad_(False)
        self.wm = WorldModel()
        self.dec = Decoder()
        self.log_alpha = nn.Parameter(torch.zeros(()))

    @torch.no_grad()
    def polyak(self, tau: float) -> None:
        for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
            pt.lerp_(p, tau)
