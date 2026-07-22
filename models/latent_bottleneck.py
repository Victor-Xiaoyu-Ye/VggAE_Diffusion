"""MIRA-style latent bottleneck: dual-stream 512ch -> compact diffusion space.

Why
---
MIRA's decisive tokenizer choice is a narrow learned bottleneck (DINOv3 1024
-> 32ch) so the world model's width/latent ratio is huge (64x). Our E7 setup
diffuses the raw 512ch dual-stream latent with a 768-wide DiT (1.5x) —
right at the RAE width-law floor. Measured effective ranks (z_geo 55/256,
z_tex 82/256) say ~4x compression is available losslessly.

The bottleneck is trained as a finetune ON TOP of the frozen R5 tokenizer
(compressor + tex_encoder stay frozen; decoder finetunes to read expanded
latents). Diffusion then operates on the 128ch compressed tokens.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LatentBottleneck(nn.Module):
    """Per-token linear compress/expand around the diffusion space.

    forward(z) -> (z_rec, z_comp): z [..., full_dim] -> compressed
    [..., comp_dim] -> reconstructed [..., full_dim].

    Linear maps (MIRA's bottleneck is likewise linear): the latent already
    has low effective rank, the bottleneck only needs to find the subspace;
    nonlinearity here would entangle the diffusion space with the decoder.
    """

    def __init__(self, full_dim: int = 512, comp_dim: int = 128):
        super().__init__()
        self.full_dim = full_dim
        self.comp_dim = comp_dim
        self.norm = nn.LayerNorm(full_dim)
        self.compress = nn.Linear(full_dim, comp_dim)
        self.expand = nn.Linear(comp_dim, full_dim)

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        return self.compress(self.norm(z))

    def decode(self, z_comp: torch.Tensor) -> torch.Tensor:
        return self.expand(z_comp)

    def forward(self, z: torch.Tensor):
        z_comp = self.encode(z)
        return self.decode(z_comp), z_comp

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
