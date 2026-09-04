"""Small single-target-frame predictors for the R7 quick generation probe."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SingleTargetGenerator(nn.Module):
    """Predict one target R7 frame from one clean anchor frame.

    The R7 tokenizer and RGB decoder remain frozen in the quick probe.  The
    target is one frame of ``[B, N, D]`` tokens; no temporal folding or rollout
    is hidden inside this module.
    """

    def __init__(self, latent_dim=192, num_tokens=324, hidden_dim=768,
                 depth=4, max_target_index=8):
        super().__init__()
        if min(latent_dim, num_tokens, hidden_dim, depth, max_target_index) < 1:
            raise ValueError("generator dimensions and target range must be positive")
        self.latent_dim = int(latent_dim)
        self.num_tokens = int(num_tokens)
        self.hidden_dim = int(hidden_dim)
        self.max_target_index = int(max_target_index)
        self.anchor_proj = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
        )
        self.target_embedding = nn.Embedding(max_target_index + 1, hidden_dim)
        self.positional = nn.Parameter(
            torch.randn(1, num_tokens, hidden_dim) * 0.02)
        heads = next(
            (value for value in (12, 8, 6, 4, 3, 2, 1)
             if hidden_dim % value == 0), 1)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=heads,
            dim_feedforward=hidden_dim * 4, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True)
        self.backbone = nn.TransformerEncoder(layer, num_layers=depth)
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, anchor, target_index):
        if anchor.ndim != 3 or tuple(anchor.shape[1:]) != (
                self.num_tokens, self.latent_dim):
            raise ValueError(
                f"expected anchor [B,{self.num_tokens},{self.latent_dim}], "
                f"got {tuple(anchor.shape)}")
        target_index = torch.as_tensor(
            target_index, device=anchor.device, dtype=torch.long).reshape(-1)
        if target_index.numel() == 1:
            target_index = target_index.expand(anchor.shape[0])
        if target_index.shape[0] != anchor.shape[0]:
            raise ValueError("target_index batch does not match anchor")
        if (target_index < 0).any() or (target_index > self.max_target_index).any():
            raise ValueError("target_index is outside the configured range")
        x = self.anchor_proj(anchor.float())
        x = x + self.positional.to(device=x.device, dtype=x.dtype)
        x = x + self.target_embedding(target_index)[:, None]
        residual = self.output(self.backbone(x)).to(dtype=anchor.dtype)
        return anchor + residual


class SingleTargetFlowGenerator(nn.Module):
    """A conditional x0 predictor over one target frame for an FM smoke."""

    def __init__(self, latent_dim=192, num_tokens=324, hidden_dim=768,
                 depth=4, max_target_index=8):
        super().__init__()
        if min(latent_dim, num_tokens, hidden_dim, depth, max_target_index) < 1:
            raise ValueError("generator dimensions and target range must be positive")
        self.latent_dim = int(latent_dim)
        self.num_tokens = int(num_tokens)
        self.max_target_index = int(max_target_index)
        self.anchor_proj = nn.Sequential(
            nn.LayerNorm(latent_dim), nn.Linear(latent_dim, hidden_dim), nn.SiLU())
        self.target_embedding = nn.Embedding(max_target_index + 1, hidden_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(128, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.positional = nn.Parameter(
            torch.randn(1, num_tokens, hidden_dim) * 0.02)
        heads = next(
            (value for value in (12, 8, 6, 4, 3, 2, 1)
             if hidden_dim % value == 0), 1)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=heads,
            dim_feedforward=hidden_dim * 4, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True)
        self.backbone = nn.TransformerEncoder(layer, num_layers=depth)
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim),
                                    nn.Linear(hidden_dim, latent_dim))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    @staticmethod
    def sinusoidal_time(t, dim=128):
        half = dim // 2
        frequency = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32)
            * (-math.log(10000.0) / max(half - 1, 1)))
        phase = t.float().reshape(-1, 1) * frequency.reshape(1, -1)
        return torch.cat((phase.sin(), phase.cos()), dim=-1)

    def forward(self, noisy, anchor, target_index, t):
        if (noisy.ndim != 3 or noisy.shape != anchor.shape
                or tuple(noisy.shape[1:]) != (
                    self.num_tokens, self.latent_dim)):
            raise ValueError(
                f"noisy and anchor must share "
                f"[B,{self.num_tokens},{self.latent_dim}] shape")
        target_index = torch.as_tensor(
            target_index, device=noisy.device, dtype=torch.long).reshape(-1)
        t = torch.as_tensor(t, device=noisy.device,
                            dtype=torch.float32).reshape(-1)
        if target_index.numel() == 1:
            target_index = target_index.expand(noisy.shape[0])
        if (target_index < 0).any() or (
                target_index > self.max_target_index).any():
            raise ValueError("target_index is outside the configured range")
        if t.numel() == 1:
            t = t.expand(noisy.shape[0])
        if target_index.shape[0] != noisy.shape[0] or t.shape[0] != noisy.shape[0]:
            raise ValueError("conditioning batch dimensions do not match")
        condition = self.anchor_proj(anchor.float())
        condition = condition + self.target_embedding(target_index)[:, None]
        condition = condition + self.time_proj(
            self.sinusoidal_time(t))[:, None]
        x = self.input_proj(noisy.float()) + condition
        x = x + self.positional.to(device=x.device, dtype=x.dtype)
        residual = self.output(self.backbone(x)).to(dtype=noisy.dtype)
        return anchor + residual


def euclidean_flow_sample(model, anchor, target_index, shape, steps=20,
                          seed=42):
    """Euler sample for a model predicting clean x0 on a Euclidean path."""
    if steps < 1:
        raise ValueError("steps must be positive")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    z = torch.randn(
        shape, generator=generator, dtype=torch.float32).to(anchor.device)
    grid = torch.linspace(0, 1, steps + 1, device=anchor.device)
    for left, right in zip(grid[:-1], grid[1:]):
        t = torch.full((shape[0],), float(left), device=anchor.device)
        x0 = model(z, anchor, target_index, t).float()
        velocity = (x0 - z) / max(1.0 - float(left), 1e-6)
        z = z + velocity * (float(right) - float(left))
    return z
