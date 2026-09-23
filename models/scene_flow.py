"""Shared image/video flow: narrow trunk, wide denoising head, clean memory.

Uses our existing tested attention/flow implementations. Architectural ideas
are informed by RAE/DDT and GAE; this is not a copy of the GAE implementation.
"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from models.r7_window_dit import WindowBlock, position, sinusoid


class SceneFlow(nn.Module):
    def __init__(self, channels=256, grid=18, width=768, depth=24,
                 head_width=1536, head_depth=6, heads=12, text_dim=4096,
                 checkpoint_blocks=True):
        super().__init__()
        self.args = dict(channels=channels, grid=grid, width=width, depth=depth,
                        head_width=head_width, head_depth=head_depth, heads=heads,
                        text_dim=text_dim, checkpoint_blocks=checkpoint_blocks)
        self.grid, self.channels = grid, channels
        self.input = nn.Linear(channels, width)
        self.camera = nn.Sequential(nn.Linear(8, 128), nn.SiLU(), nn.Linear(128, width))
        nn.init.zeros_(self.camera[-1].weight)
        nn.init.zeros_(self.camera[-1].bias)
        self.ref = nn.Linear(channels, width)
        self.null_ref = nn.Parameter(torch.zeros(1, 1, width))
        self.text = nn.Linear(text_dim, width) if text_dim else None
        self.time = nn.Sequential(nn.Linear(256, width), nn.SiLU(), nn.Linear(width, width))
        self.trunk = nn.ModuleList([WindowBlock(width, heads, text_dim) for _ in range(depth)])
        self.widen = nn.Linear(width, head_width)
        self.ref_widen = nn.Linear(width, head_width)
        self.text_widen = nn.Linear(width, head_width) if text_dim else None
        self.time_widen = nn.Linear(width, head_width)
        self.wide = nn.ModuleList([WindowBlock(head_width, heads, text_dim) for _ in range(head_depth)])
        self.output = nn.Sequential(nn.LayerNorm(head_width), nn.Linear(head_width, channels))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)
        self.checkpoint_blocks = checkpoint_blocks

    def forward(self, noisy, u, anchor, text=None, text_valid=None, ref_present=None, rays=None):
        b, t, n, c = noisy.shape
        if n != self.grid**2 or c != self.channels:
            raise ValueError('SceneFlow latent layout mismatch')
        pos = position(t, self.grid, self.args['width']).to(noisy.device)
        x = self.input(noisy.float().flatten(1, 2)) + pos
        if rays is None:
            rays = noisy.new_zeros(b, t, n, 8)
        x = x + self.camera(rays.float().flatten(1, 2))
        memory = self.ref(anchor[:, 0].float())
        if ref_present is None:
            ref_present = torch.ones(b, device=noisy.device, dtype=torch.bool)
        memory = torch.where(ref_present[:, None, None], memory, self.null_ref)
        memory = memory + position(1, self.grid, self.args['width'], anchor=True).to(noisy.device)
        context = self.text(text.float()) if self.text is not None else None
        time = self.time(sinusoid(u*1000, 256))[:, None]
        for blocks in (self.trunk, self.wide):
            for block in blocks:
                if self.training and self.checkpoint_blocks:
                    x = checkpoint(block, x, time, memory, context, text_valid, use_reentrant=False)
                else:
                    x = block(x, time, memory, context, text_valid)
            if blocks is self.trunk:
                x, memory, time = self.widen(x), self.ref_widen(memory), self.time_widen(time)
                if context is not None:
                    context = self.text_widen(context)
        return self.output(x).reshape(b, t, n, c).float()
