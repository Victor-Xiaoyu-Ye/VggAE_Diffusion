"""R7-to-Wan native token bridge for teacher-alignment probes."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class R7WanTeacherBridge(nn.Module):
    """Align R7 spatial-temporal tokens with frozen Wan patch hidden tokens.

    The bridge is intentionally independent of Wan VAE at inference. Teacher
    tokens are consumed only by ``alignment_loss`` during training/probing.
    """

    def __init__(self, r7_dim=192, wan_dim=1536, depth=2):
        super().__init__()
        layers = [nn.LayerNorm(r7_dim), nn.Linear(r7_dim, wan_dim), nn.SiLU()]
        for _ in range(max(depth - 1, 0)):
            layers.extend([
                nn.LayerNorm(wan_dim), nn.Linear(wan_dim, wan_dim), nn.SiLU()])
        self.projection = nn.Sequential(*layers)

    def forward(self, r7):
        if r7.ndim != 4:
            raise ValueError(f"expected R7 [B,T,N,D], got {tuple(r7.shape)}")
        return self.projection(r7)

    @staticmethod
    def pool_tokens(tokens, frames, height, width):
        """Pool arbitrary Wan patch grids to one hidden vector per R7 token.

        Wan teacher tensors arrive as [B,C,F,H,W]. Adaptive pooling keeps this
        probe agnostic to native VAE spatial/temporal patch sizes.
        """
        if tokens.ndim != 5:
            raise ValueError(
                f"expected Wan patch hidden [B,C,F,H,W], got {tuple(tokens.shape)}")
        pooled = F.adaptive_avg_pool3d(tokens.float(), (frames, height, width))
        return pooled.permute(0, 2, 3, 4, 1).flatten(2, 3)

    def alignment_loss(self, r7, teacher):
        predicted = self(r7)
        if r7.shape[2] <= 0:
            raise ValueError("R7 token count must be positive")
        side = int(r7.shape[2] ** 0.5)
        if side * side != r7.shape[2]:
            raise ValueError("R7 spatial token count must be a square grid")
        target = self.pool_tokens(teacher, r7.shape[1], side, side)
        if predicted.shape != target.shape:
            raise RuntimeError(
                f"bridge/teacher shape mismatch {predicted.shape} != {target.shape}")
        pred_flat = predicted.flatten(0, 2)
        target_flat = target.flatten(0, 2)
        feature = 1.0 - F.cosine_similarity(
            pred_flat, target_flat, dim=1).mean()
        mean = F.mse_loss(pred_flat.mean(0), target_flat.mean(0))
        pred_center = pred_flat - pred_flat.mean(0)
        target_center = target_flat - target_flat.mean(0)
        variance = F.mse_loss(
            pred_center.square().mean(0).sqrt(),
            target_center.square().mean(0).sqrt())
        return feature + 0.1 * mean + 0.1 * variance, {
            "feature": feature.detach(), "mean": mean.detach(),
            "variance": variance.detach()}
