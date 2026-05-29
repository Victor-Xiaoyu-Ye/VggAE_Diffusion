#!/usr/bin/env python3
"""Systematic verification of the diffusion pipeline.

Each experiment isolates one component to identify where things break.
Run on a single GPU. Results inform architecture decisions.

Verification Matrix:
  V1 — Single-token flow matching (MLP): can we learn E[v|zt,t] in 2048-dim?
  V2 — Time embedding ablation: is t-conditioning working correctly?
  V3 — Architecture comparison: concat vs adaLN conditioning
  V4 — Spatial structure: does spatial attention help?
  V5 — Multi-level target: single level vs mean-of-levels
  V6 — Decoder sensitivity: how accurate must tokens be?
  V7 — Spatiotemporal DiT: full 8-frame factorized spatial-temporal model
  V8 — Frame count: compare seq_len=1/4/8/16
  V4 — Spatial structure necessity: does spatial context help?
  V5 — Multi-level target: single level vs mean-of-levels
  V6 — Decoder sensitivity: how accurate must predicted tokens be?

Usage:
  python verify_diffusion.py --exp v1          # Run V1 only
  python verify_diffusion.py --exp all         # Run all experiments
  python verify_diffusion.py --exp v1,v6       # Run V1 and V6
"""

import argparse
import os, sys, time, json
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Project imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'Wan2.1'))

from streamvggt.models.streamvggt import StreamVGGT
from models.flow_matching import OTCFM, logit_normal_sampling
from data.video_dataset import SpatialVidDataset, collate_fn
from data.token_utils import (
    DPT_LEVELS, DEFAULT_BOUNDARY_LEVEL,
    load_token_stats, normalize_tokens, select_levels, select_levels_mean,
    strip_special_tokens, build_decoder_tokens_from_generated,
)
from utils.decoder_loader import load_decoder

# ---------------------------------------------------------------------------
# Common utilities
# ---------------------------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    return torch.device('cpu')


def load_encoder(ckpt_path, device):
    encoder = StreamVGGT(img_size=518, patch_size=14, embed_dim=1024)
    state = torch.load(ckpt_path, map_location='cpu')
    encoder.load_state_dict(state, strict=False)
    encoder = encoder.to(device=device, dtype=torch.bfloat16).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


def get_token_batch(encoder, dataset, device, dtype=torch.bfloat16):
    """Extract one batch of tokens from encoder."""
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, collate_fn=collate_fn,
        num_workers=2, drop_last=True,
    )
    frames, tokens_list = None, None
    for batch in loader:
        frames = batch['frames'].to(device=device, dtype=dtype)
        with torch.no_grad():
            tl, psi = encoder(frames)
            tl = strip_special_tokens(tl, psi)
        tokens_list = tl
        break
    return frames, tokens_list


# ---------------------------------------------------------------------------
# V1: Single-token flow matching
# ---------------------------------------------------------------------------

class TokenMLP(nn.Module):
    """Simple MLP for single-token flow matching (no spatial/temporal context)."""

    def __init__(self, token_dim=2048, hidden_dim=1024, num_layers=4,
                 time_emb_dim=256, time_mode='sinusoidal', cond_mode='concat'):
        super().__init__()
        self.token_dim = token_dim
        self.time_emb_dim = time_emb_dim
        self.time_mode = time_mode
        self.cond_mode = cond_mode

        # Time embedding
        if time_mode == 'sinusoidal':
            self.time_mlp = nn.Sequential(
                nn.Linear(time_emb_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        elif time_mode == 'fourier':
            # Use log-spaced frequencies up to Nyquist-like limit
            self.fourier_dim = time_emb_dim
            self.time_mlp = nn.Sequential(
                nn.Linear(time_emb_dim * 2, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        elif time_mode == 'learned':
            self.time_embed = nn.Embedding(1000, time_emb_dim)
            self.time_mlp = nn.Sequential(
                nn.Linear(time_emb_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        elif time_mode == 'direct':
            # Just use t directly as a feature
            self.time_mlp = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        if cond_mode == 'concat':
            # Main network: concat(token, time_emb) → velocity
            in_dim = token_dim + hidden_dim
            layers = []
            for i in range(num_layers):
                out_dim = hidden_dim if i < num_layers - 1 else token_dim
                layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, out_dim))
                if i < num_layers - 1:
                    layers.append(nn.LayerNorm(out_dim))
                    layers.append(nn.SiLU())
            self.net = nn.Sequential(*layers)
            # Zero-init output
            nn.init.zeros_(layers[-1].weight)
            nn.init.zeros_(layers[-1].bias)
        elif cond_mode == 'adaln':
            # adaLN-style: each layer has scale+shift conditioned on time
            self.adaln_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * num_layers * 4),
                nn.SiLU(),
                nn.Linear(hidden_dim * num_layers * 4, hidden_dim * num_layers * 4),
            )
            self.blocks = nn.ModuleList([
                AdaLNMLPBlock(token_dim if i == 0 else hidden_dim,
                              hidden_dim if i < num_layers - 1 else token_dim,
                              i < num_layers - 1)
                for i in range(num_layers)
            ])
            self.num_layers = num_layers
            self.hidden_dim = hidden_dim
            self.token_dim = token_dim

    def _time_embed(self, t):
        """t: [B] in [0, 1]."""
        if self.time_mode == 'sinusoidal':
            half = self.time_emb_dim // 2
            emb = torch.exp(
                torch.arange(half, device=t.device, dtype=torch.float32) *
                (-math.log(10000) / (half - 1))
            )
            emb = t.float().unsqueeze(1) * emb.unsqueeze(0)  # [B, half]
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
            return self.time_mlp(emb.to(t.dtype))
        elif self.time_mode == 'fourier':
            max_freq_exp = min(self.fourier_dim - 1, 10)
            freqs = 2.0 ** torch.arange(0, max_freq_exp + 1, device=t.device, dtype=torch.float32)
            emb = t.float().unsqueeze(1) * freqs.unsqueeze(0) * math.pi
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
            if emb.shape[1] < self.fourier_dim * 2:
                pad = torch.zeros(emb.shape[0], self.fourier_dim * 2 - emb.shape[1],
                                  device=t.device, dtype=emb.dtype)
                emb = torch.cat([emb, pad], dim=1)
            else:
                emb = emb[:, :self.fourier_dim * 2]
            return self.time_mlp(emb.to(t.dtype))
        elif self.time_mode == 'learned':
            idx = (t * 999).long().clamp(0, 999)
            emb = self.time_embed(idx)
            return self.time_mlp(emb)
        elif self.time_mode == 'direct':
            return self.time_mlp(t.unsqueeze(1))

    def forward(self, z, t, cond=None, text_emb=None):
        """z: [B, D], t: [B] in [0,1]."""
        t_emb = self._time_embed(t)  # [B, hidden_dim]

        if self.cond_mode == 'concat':
            x = torch.cat([z, t_emb], dim=-1)
            return self.net(x)
        elif self.cond_mode == 'adaln':
            # adaLN: project time to per-layer modulation params
            mod = self.adaln_proj(t_emb)  # [B, num_layers * 4 * hidden]
            mod = mod.chunk(self.num_layers * 4, dim=-1)  # list of [B, hidden]
            x = z
            for i, block in enumerate(self.blocks):
                scale1 = mod[i * 4]
                shift1 = mod[i * 4 + 1]
                scale2 = mod[i * 4 + 2]
                shift2 = mod[i * 4 + 3]
                x = block(x, scale1, shift1, scale2, shift2)
            return x


class AdaLNMLPBlock(nn.Module):
    """Single MLP block with adaLN modulation."""

    def __init__(self, in_dim, out_dim, has_activation):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.has_activation = has_activation
        if has_activation:
            self.norm1 = nn.LayerNorm(out_dim)
            self.norm2 = nn.LayerNorm(out_dim)
            self.mlp = nn.Sequential(
                nn.Linear(out_dim, out_dim * 4),
                nn.SiLU(),
                nn.Linear(out_dim * 4, out_dim),
            )
            # Zero-init the second linear of MLP
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)
        else:
            # Final layer: zero-init
            nn.init.zeros_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)

    def forward(self, x, scale1, shift1, scale2, shift2):
        if self.has_activation:
            x = self.linear(x)
            x = self.norm1(x) * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
            x = x + self.mlp(self.norm2(x) * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1))
            return x
        else:
            return self.linear(x)


class SpatialMLP(nn.Module):
    """Per-token MLP for full spatial grid (no spatial interaction, no temporal)."""

    def __init__(self, token_dim=2048, hidden_dim=1024, num_layers=4, time_emb_dim=256):
        super().__init__()
        self.mlp = TokenMLP(token_dim, hidden_dim, num_layers, time_emb_dim)

    def forward(self, z, t, cond=None, text_emb=None):
        """z: [B, S, N, D] → [B, S, N, D]."""
        B, S, N, D = z.shape
        z_flat = z.reshape(-1, D)
        t_repeat = t.repeat_interleave(S * N)
        v_flat = self.mlp(z_flat, t_repeat)
        return v_flat.reshape(B, S, N, D)


class TinyDiT(nn.Module):
    """Minimal DiT for single-frame spatial token grid with adaLN."""

    def __init__(self, token_dim=2048, hidden_dim=512, num_layers=4, num_heads=8,
                 num_tokens=1369, time_emb_dim=256):
        super().__init__()
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 6),
        )

        # Input projection
        self.input_proj = nn.Linear(token_dim, hidden_dim)

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, hidden_dim) * 0.02)

        # Transformer blocks with adaLN
        self.blocks = nn.ModuleList([
            AdaLNBlock(hidden_dim, num_heads) for _ in range(num_layers)
        ])

        # Output projection (zero-init)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, token_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        self.time_emb_dim = time_emb_dim

    def _time_embed(self, t):
        half = self.time_emb_dim // 2
        emb = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32) *
            (-math.log(10000) / (half - 1))
        )
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return self.time_mlp(emb.to(t.dtype))  # [B, hidden*6]

    def forward(self, z, t, cond=None, text_emb=None):
        """z: [B, S=1, N, D] → [B, S=1, N, D]."""
        B, S, N, D = z.shape
        z = z.squeeze(1)  # [B, N, D]
        x = self.input_proj(z)  # [B, N, hidden]

        # Add positional embedding
        if x.size(1) == self.pos_embed.size(1):
            x = x + self.pos_embed

        t_emb = self._time_embed(t)  # [B, hidden*6]

        for block in self.blocks:
            x = block(x, t_emb)

        x = self.output_norm(x)
        x = self.output_proj(x)  # [B, N, D]
        return x.unsqueeze(1)  # [B, 1, N, D]


class AdaLNBlock(nn.Module):
    """Transformer block with adaptive layer norm."""

    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        # adaLN: each block gets scale+shift for norm1, norm2, and residual scales
        # 6 params: scale1, shift1, scale_attn, scale2, shift2, scale_mlp
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim * 6, dim * 6),
        )

    def forward(self, x, t_emb):
        modulation = self.adaLN_modulation(t_emb).chunk(6, dim=-1)
        # modulation[i]: [B, dim]
        scale1 = modulation[0].unsqueeze(1)
        shift1 = modulation[1].unsqueeze(1)
        scale_attn = modulation[2].unsqueeze(1)
        scale2 = modulation[3].unsqueeze(1)
        shift2 = modulation[4].unsqueeze(1)
        scale_mlp = modulation[5].unsqueeze(1)

        # Self-attention with adaLN
        x_norm = self.norm1(x) * (1 + scale1) + shift1
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out * scale_attn

        # MLP with adaLN
        x_norm = self.norm2(x) * (1 + scale2) + shift2
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out * scale_mlp

        return x


# ---------------------------------------------------------------------------
# V1-V3: Single-token & architecture experiments
# ---------------------------------------------------------------------------

import math


def run_single_token_experiment(encoder, level_stats, dataset, device,
                                 num_steps=2000, batch_size=64,
                                 model_type='mlp', time_mode='sinusoidal',
                                 token_dim=2048, use_single_frame=True,
                                 cond_mode='concat', num_layers=4, hidden_dim=1024):
    """V1-V3: Train flow matching on tokens, return loss history and diagnostics."""

    # Prepare data: collect tokens from multiple batches
    print(f"  Collecting token data...")
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, collate_fn=collate_fn,
        num_workers=2, drop_last=True,
    )
    all_tokens = []
    max_batches = 50  # collect up to 200 samples
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        frames = batch['frames'].to(device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            tl, psi = encoder(frames)
            tl = strip_special_tokens(tl, psi)
            tl = normalize_tokens(tl, level_stats)
        # Use single level (11) or multi-level mean
        x1 = select_levels(tl, levels=[11]).to(dtype=torch.float32)  # [B, S, 1369, 2048]
        if use_single_frame:
            x1 = x1[:, 0:1]  # [B, 1, 1369, 2048] — first frame only
        all_tokens.append(x1.cpu())
    all_tokens = torch.cat(all_tokens, dim=0)  # [total_B, S, 1369, 2048]
    print(f"  Collected {all_tokens.shape[0]} samples × {all_tokens.shape[1]} frames × {all_tokens.shape[2]} tokens")

    # For single-token experiment: pick one token position
    # Pick center token (idx 684 = 37*18+18 ≈ center)
    token_idx = 684  # center token
    x1_single = all_tokens[:, 0, token_idx, :].contiguous()  # [N_samples, 2048]
    print(f"  Single-token data: {x1_single.shape}")

    # Build model
    if model_type == 'mlp':
        model = TokenMLP(token_dim=token_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                         time_emb_dim=256, time_mode=time_mode, cond_mode=cond_mode).to(device)
    elif model_type == 'spatial_mlp':
        model = SpatialMLP(token_dim=token_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    elif model_type == 'tiny_dit':
        model = TinyDiT(token_dim=token_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                        num_heads=8, num_tokens=1369).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    flow = OTCFM(model)

    total_p = sum(p.numel() for p in model.parameters())
    print(f"  Model: {model_type}/{time_mode}/{cond_mode}, layers={num_layers}, hidden={hidden_dim}, params: {total_p:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)

    # Training loop
    loss_history = []
    sample_stats_history = []  # track generated token statistics

    # Create a fixed eval set for consistent tracking
    eval_indices = torch.randperm(len(x1_single))[:min(64, len(x1_single))]
    x1_eval = x1_single[eval_indices].to(device)

    model.train()
    pbar = tqdm(range(num_steps), desc=f"  Training {model_type}/{time_mode}")
    for step in pbar:
        # Sample batch
        idx = torch.randint(0, len(x1_single), (batch_size,))
        x1_batch = x1_single[idx].to(device)

        loss = flow.compute_loss(x1_batch)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        loss_history.append(loss.item())

        if step % 200 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Evaluate every 500 steps
        if step % 500 == 0 or step == num_steps - 1:
            model.eval()
            with torch.no_grad():
                # Compute loss on eval set
                eval_loss = flow.compute_loss(x1_eval)
                # Sample and check statistics
                z_sample = torch.randn(64, token_dim, device=device)
                dt = 1.0 / 20
                for i in range(20):
                    t_val = torch.full((64,), i / 20., device=device)
                    v = model(z_sample, t_val)
                    z_sample = z_sample + v * dt
                sample_std = z_sample.std().item()
                sample_mean = z_sample.mean().item()
                sample_stats_history.append({
                    'step': step,
                    'eval_loss': eval_loss.item(),
                    'sample_std': sample_std,
                    'sample_mean': sample_mean,
                })
                print(f"  [step {step}] eval_loss={eval_loss.item():.4f}, "
                      f"sample_std={sample_std:.4f}, sample_mean={sample_mean:.4f}")
            model.train()

    # Final diagnostics
    model.eval()
    with torch.no_grad():
        # Check: does model output vary with t for the same z?
        z_fixed = torch.randn(16, token_dim, device=device)
        outputs_at_t = []
        for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
            t = torch.full((16,), t_val, device=device)
            v = model(z_fixed, t)
            outputs_at_t.append(v.cpu())
        # Compute variance of output across t (should be large if time-conditioning works)
        stacked = torch.stack(outputs_at_t, dim=0)  # [5, 16, D]
        time_variance = stacked.var(dim=0).mean().item()
        print(f"  Time-conditioning variance: {time_variance:.6f}")
        print(f"  (Higher = model responds to t. Near 0 = time embedding broken)")

        # Check: what does the model predict at t=1 (should be close to x1)?
        x1_test = x1_eval[:16].to(device)
        v_at_t1 = model(x1_test, torch.ones(16, device=device))
        expected_v_at_t1 = x1_test  # E[v|z=x1, t=1] = E[x1 - x0 | x1] = x1

    return {
        'loss_history': loss_history,
        'sample_stats': sample_stats_history,
        'time_variance': time_variance,
        'final_eval_loss': sample_stats_history[-1]['eval_loss'] if sample_stats_history else None,
        'initial_loss_approx': loss_history[0] if loss_history else None,
    }


# ---------------------------------------------------------------------------
# V4: Spatial structure necessity
# ---------------------------------------------------------------------------

def run_spatial_experiment(encoder, level_stats, dataset, device,
                            num_steps=1500, batch_size=8,
                            use_spatial=True):
    """V4: Train full-grid (1 frame, 1369 tokens), compare per-token vs spatial."""
    print(f"  Collecting token data...")
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, collate_fn=collate_fn,
        num_workers=2, drop_last=True,
    )
    all_tokens = []
    max_batches = 60
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        frames = batch['frames'].to(device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            tl, psi = encoder(frames)
            tl = strip_special_tokens(tl, psi)
            tl = normalize_tokens(tl, level_stats)
        x1 = select_levels(tl, levels=[11]).to(dtype=torch.float32)
        x1 = x1[:, 0:1]  # single frame
        all_tokens.append(x1.cpu())
    all_tokens = torch.cat(all_tokens, dim=0)  # [N, 1, 1369, 2048]
    print(f"  Collected {all_tokens.shape[0]} samples × 1369 tokens")

    if use_spatial:
        model = TinyDiT(token_dim=2048, hidden_dim=512, num_layers=4, num_heads=8).to(device)
        model_type = 'tiny_dit'
    else:
        model = SpatialMLP(token_dim=2048, hidden_dim=1024, num_layers=4).to(device)
        model_type = 'per_token_mlp'

    flow = OTCFM(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)

    # Eval set
    eval_idx = torch.randperm(len(all_tokens))[:min(16, len(all_tokens))]
    x1_eval = all_tokens[eval_idx].to(device)

    loss_history = []
    model.train()
    pbar = tqdm(range(num_steps), desc=f"  Training {model_type}")
    for step in pbar:
        idx = torch.randint(0, len(all_tokens), (batch_size,))
        x1_batch = all_tokens[idx].to(device)

        loss = flow.compute_loss(x1_batch)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        loss_history.append(loss.item())
        if step % 200 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    # Final eval
    model.eval()
    with torch.no_grad():
        eval_loss = flow.compute_loss(x1_eval)
        print(f"  Final eval loss ({model_type}): {eval_loss.item():.4f}")

    return {'loss_history': loss_history, 'final_eval_loss': eval_loss.item()}


# ---------------------------------------------------------------------------
# V5: Multi-level target comparison
# ---------------------------------------------------------------------------

def run_multilevel_experiment(encoder, level_stats, dataset, device,
                               num_steps=1500, batch_size=64):
    """V5: Compare single-level vs multi-level mean as flow matching target."""
    print(f"  Collecting token data...")
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, collate_fn=collate_fn,
        num_workers=2, drop_last=True,
    )
    all_single = []
    all_mean = []
    max_batches = 50
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        frames = batch['frames'].to(device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            tl, psi = encoder(frames)
            tl = strip_special_tokens(tl, psi)
            tl = normalize_tokens(tl, level_stats)
        single = select_levels(tl, levels=[11]).to(dtype=torch.float32)
        mean_lvl = select_levels_mean(tl, levels=[4, 11, 17, 23], downsample=0).to(dtype=torch.float32)
        # Single token, first frame
        all_single.append(single[:, 0:1, 684, :].cpu())
        all_mean.append(mean_lvl[:, 0:1, 684, :].cpu())
    all_single = torch.cat(all_single, dim=0).squeeze(1)  # [N, 2048]
    all_mean = torch.cat(all_mean, dim=0).squeeze(1)  # [N, 2048]

    print(f"  Single-level data: {all_single.shape}, std={all_single.std():.4f}")
    print(f"  Multi-level mean data: {all_mean.shape}, std={all_mean.std():.4f}")

    results = {}
    for label, data in [('single_level', all_single), ('multi_level_mean', all_mean)]:
        model = TokenMLP(token_dim=2048, hidden_dim=1024, num_layers=4).to(device)
        flow = OTCFM(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)

        # Eval
        eval_idx = torch.randperm(len(data))[:min(64, len(data))]
        x1_eval = data[eval_idx].to(device)

        loss_history = []
        model.train()
        pbar = tqdm(range(num_steps), desc=f"  Training {label}")
        for step in pbar:
            idx = torch.randint(0, len(data), (batch_size,))
            x1_batch = data[idx].to(device)
            loss = flow.compute_loss(x1_batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            loss_history.append(loss.item())
            if step % 200 == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        with torch.no_grad():
            eval_loss = flow.compute_loss(x1_eval)
            print(f"  {label}: final_eval_loss={eval_loss.item():.4f}")

            # Check target variance (higher var = harder task? or more signal?)
            v_target = x1_eval - torch.randn_like(x1_eval)
            target_var = v_target.var().item()
            print(f"  {label}: target_velocity_variance={target_var:.4f}")

        results[label] = {
            'loss_history': loss_history,
            'final_eval_loss': eval_loss.item(),
            'target_variance': target_var,
            'data_std': data.std().item(),
        }

    return results


# ---------------------------------------------------------------------------
# V6: Decoder sensitivity
# ---------------------------------------------------------------------------

def run_decoder_sensitivity(encoder, level_stats, dataset, device,
                             decoder_ckpt_path):
    """V6: Measure how decoder PSNR degrades with token noise.

    This determines the tolerance: how accurate must diffusion-generated tokens be?
    """
    print(f"  Loading decoder from {decoder_ckpt_path}")
    decoder = load_decoder(decoder_ckpt_path, device, decoder_type='auto')

    # Get one batch of tokens + frames
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, collate_fn=collate_fn,
        num_workers=2, drop_last=True,
    )
    for batch in loader:
        frames = batch['frames'].to(device=device, dtype=torch.bfloat16)
        break

    with torch.no_grad():
        tl, psi = encoder(frames)
        tl = strip_special_tokens(tl, psi)
        tl_clean = normalize_tokens(tl, level_stats)  # keep normalized
        tl_unnorm = list(tl)  # unnormalized for decoder (decoder expects raw tokens)

    # Get clean reconstruction
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            z_clean = select_levels(tl_unnorm, levels=[11]).float()
            tokens_for_decoder = build_decoder_tokens_from_generated(
                z_clean, levels=[11], seq_len=8,
            )
            result = decoder(tokens_for_decoder, images=frames.float(),
                           patch_start_idx=0, frames_chunk_size=8)
        if getattr(decoder, 'output_depth', False):
            preds_clean, _, _, _ = result
        else:
            preds_clean, _ = result

    recon_clean = preds_clean.permute(0, 1, 4, 2, 3).contiguous()[:, :, :3].clamp(0, 1)
    clean_psnr = -10 * torch.log10(F.mse_loss(recon_clean, frames.float().clamp(0, 1))).item()
    print(f"  Clean reconstruction PSNR: {clean_psnr:.2f} dB")

    # Test with increasing noise on tokens
    noise_levels = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    psnr_results = []

    for noise_std in noise_levels:
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                z_noisy = z_clean + torch.randn_like(z_clean) * noise_std
                tokens_for_decoder = build_decoder_tokens_from_generated(
                    z_noisy, levels=[11], seq_len=8,
                )
                result = decoder(tokens_for_decoder, images=frames.float(),
                               patch_start_idx=0, frames_chunk_size=8)
            if getattr(decoder, 'output_depth', False):
                preds, _, _, _ = result
            else:
                preds, _ = result

        recon = preds.permute(0, 1, 4, 2, 3).contiguous()[:, :, :3].clamp(0, 1)
        psnr = -10 * torch.log10(F.mse_loss(recon, frames.float().clamp(0, 1))).item()
        psnr_results.append((noise_std, psnr))
        print(f"  noise_std={noise_std:.3f}: PSNR={psnr:.2f} dB")

    # Also test: what happens if we use mean-of-levels tokens at level 11?
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            z_mean = select_levels_mean(tl_unnorm, levels=[4, 11, 17, 23], downsample=0).float()
            tokens_for_decoder = build_decoder_tokens_from_generated(
                z_mean, levels=[11], seq_len=8,
            )
            result = decoder(tokens_for_decoder, images=frames.float(),
                           patch_start_idx=0, frames_chunk_size=8)
        if getattr(decoder, 'output_depth', False):
            preds, _, _, _ = result
        else:
            preds, _ = result
    recon = preds.permute(0, 1, 4, 2, 3).contiguous()[:, :, :3].clamp(0, 1)
    mean_psnr = -10 * torch.log10(F.mse_loss(recon, frames.float().clamp(0, 1))).item()
    print(f"  Multi-level mean as level-11: PSNR={mean_psnr:.2f} dB")

    # Find acceptable noise level (PSNR drop < 2dB)
    acceptable_noise = None
    for noise_std, psnr in psnr_results:
        if clean_psnr - psnr < 2.0 and acceptable_noise is None:
            acceptable_noise = noise_std

    return {
        'clean_psnr': clean_psnr,
        'psnr_vs_noise': psnr_results,
        'mean_as_level11_psnr': mean_psnr,
        'acceptable_noise_std': acceptable_noise,
    }


# ---------------------------------------------------------------------------
# V7/V8: Spatiotemporal DiT
# ---------------------------------------------------------------------------

class FactorizedSpatioTemporalDiT(nn.Module):
    """Factorized spatial-temporal DiT with concat time conditioning.

    Alternates spatial attention (within-frame) and temporal attention
    (across-frame at same position). Complexity: O(S*N² + N*S²) instead
    of O((S*N)²) for full attention.

    No adaLN — uses simple concat of time embedding with token features.
    """

    def __init__(self, token_dim=2048, hidden_dim=512, spatial_depth=4,
                 temporal_depth=2, num_heads=8, seq_len=8, num_tokens=1369,
                 time_emb_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_emb_dim = time_emb_dim
        self.seq_len = seq_len
        self.num_tokens = num_tokens

        # Time embedding (sinusoidal + MLP → concat to tokens)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Input projection: token + time → hidden_dim
        self.input_proj = nn.Linear(token_dim + hidden_dim, hidden_dim)

        # Positional embeddings
        self.spatial_pos = nn.Parameter(
            torch.randn(1, 1, num_tokens, hidden_dim) * 0.02)
        self.temporal_pos = nn.Parameter(
            torch.randn(1, seq_len, 1, hidden_dim) * 0.02)

        # Spatial attention blocks
        self.spatial_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads)
            for _ in range(spatial_depth)
        ])

        # Temporal attention blocks
        self.temporal_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads)
            for _ in range(temporal_depth)
        ])

        # Output projection (zero-init)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, token_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def _time_embed(self, t):
        half = self.time_emb_dim // 2
        emb = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32) *
            (-math.log(10000) / (half - 1))
        )
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return self.time_mlp(emb.to(t.dtype))  # [B, hidden_dim]

    def forward(self, z, t, cond=None, text_emb=None):
        B, S, N, D = z.shape

        # Time embedding → broadcast to all tokens
        t_emb = self._time_embed(t)  # [B, hidden_dim]
        t_emb = t_emb.unsqueeze(1).unsqueeze(1).expand(B, S, N, -1)  # [B, S, N, hidden_dim]

        # Concat token + time → project
        x = torch.cat([z, t_emb], dim=-1)  # [B, S, N, D + hidden_dim]
        x = self.input_proj(x)  # [B, S, N, hidden_dim]

        # Add positional embeddings (truncate/pad as needed)
        x = x + self.spatial_pos[:, :, :N, :] + self.temporal_pos[:, :S, :, :]

        # Spatial blocks: attend within each frame
        for block in self.spatial_blocks:
            x_flat = x.reshape(B * S, N, self.hidden_dim)
            x_flat = block(x_flat)
            x = x_flat.reshape(B, S, N, self.hidden_dim)

        # Temporal blocks: attend across frames at each position
        for block in self.temporal_blocks:
            x_flat = x.permute(0, 2, 1, 3).reshape(B * N, S, self.hidden_dim)
            x_flat = block(x_flat)
            x = x_flat.reshape(B, N, S, self.hidden_dim).permute(0, 2, 1, 3)

        # Output
        x = self.output_norm(x)
        x = self.output_proj(x)  # [B, S, N, D]
        return x


class TransformerBlock(nn.Module):
    """Standard transformer block: self-attention + MLP, pre-norm."""

    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        # Self-attention
        y = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + y
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


class PerTokenMLPFull(nn.Module):
    """Per-token MLP applied to all tokens independently (no spatial/temporal interaction)."""

    def __init__(self, token_dim=2048, hidden_dim=1024, num_layers=4, time_emb_dim=256):
        super().__init__()
        self.token_mlp = TokenMLP(token_dim, hidden_dim, num_layers, time_emb_dim, 'sinusoidal', 'concat')
        self.token_dim = token_dim

    def forward(self, z, t, cond=None, text_emb=None):
        B, S, N, D = z.shape
        z_flat = z.reshape(-1, D)
        t_repeat = t.repeat_interleave(S * N)
        v_flat = self.token_mlp(z_flat, t_repeat)
        return v_flat.reshape(B, S, N, D)


class SpatialOnlyDiT(nn.Module):
    """Spatial DiT that processes each frame independently (no temporal interaction)."""

    def __init__(self, token_dim=2048, hidden_dim=512, depth=6, num_heads=8,
                 num_tokens=1369, time_emb_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_emb_dim = time_emb_dim
        self.num_tokens = num_tokens

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.input_proj = nn.Linear(token_dim + hidden_dim, hidden_dim)
        self.spatial_pos = nn.Parameter(
            torch.randn(1, 1, num_tokens, hidden_dim) * 0.02)
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads) for _ in range(depth)
        ])
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, token_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def _time_embed(self, t):
        half = self.time_emb_dim // 2
        emb = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32) *
            (-math.log(10000) / (half - 1))
        )
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return self.time_mlp(emb.to(t.dtype))

    def forward(self, z, t, cond=None, text_emb=None):
        B, S, N, D = z.shape
        t_emb = self._time_embed(t).unsqueeze(1).unsqueeze(1).expand(B, S, N, -1)
        x = torch.cat([z, t_emb], dim=-1)
        x = self.input_proj(x)
        x = x + self.spatial_pos[:, :, :N, :]
        x_flat = x.reshape(B * S, N, self.hidden_dim)
        for block in self.blocks:
            x_flat = block(x_flat)
        x = x_flat.reshape(B, S, N, self.hidden_dim)
        x = self.output_norm(x)
        x = self.output_proj(x)
        return x


def run_spatiotemporal_experiment(encoder, level_stats, dataset, device,
                                    num_steps=1500, seq_len=8, batch_size=4):
    """V7/V8: Compare per-token MLP, spatial-only DiT, spatiotemporal DiT."""
    print(f"  Collecting token data (seq_len={seq_len})...")
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, collate_fn=collate_fn,
        num_workers=2, drop_last=True,
    )
    all_tokens = []
    max_batches = 80
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        frames = batch['frames'].to(device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            tl, psi = encoder(frames)
            tl = strip_special_tokens(tl, psi)
            tl = normalize_tokens(tl, level_stats)
        x1 = select_levels(tl, levels=[11]).to(dtype=torch.float32)
        x1 = x1[:, :seq_len]  # take first seq_len frames
        all_tokens.append(x1.cpu())
    all_tokens = torch.cat(all_tokens, dim=0)
    print(f"  Collected {all_tokens.shape[0]} samples × {all_tokens.shape[1]} frames × 1369 tokens")

    results = {}

    # Model configs to test
    configs = [
        ('per_token_mlp', lambda: PerTokenMLPFull(token_dim=2048, hidden_dim=1024, num_layers=4).to(device)),
        ('spatial_only', lambda: SpatialOnlyDiT(token_dim=2048, hidden_dim=512, depth=6,
                                                  num_heads=8).to(device)),
        ('spatiotemporal', lambda: FactorizedSpatioTemporalDiT(
            token_dim=2048, hidden_dim=768, spatial_depth=4, temporal_depth=2,
            num_heads=12, seq_len=seq_len).to(device)),
    ]

    # Eval set
    eval_idx = torch.randperm(len(all_tokens))[:min(8, len(all_tokens))]
    x1_eval = all_tokens[eval_idx].to(device)

    for label, build_model in configs:
        model = build_model()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n  --- {label}: {n_params:,} params ---")

        flow = OTCFM(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)

        model.train()
        loss_history = []
        pbar = tqdm(range(num_steps), desc=f"  {label}")
        for step in pbar:
            idx = torch.randint(0, len(all_tokens), (batch_size,))
            x1_batch = all_tokens[idx].to(device)

            loss = flow.compute_loss(x1_batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            loss_history.append(loss.item())
            if step % 300 == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        with torch.no_grad():
            eval_loss = flow.compute_loss(x1_eval)
            print(f"  {label}: final_eval_loss={eval_loss.item():.4f}")

        results[label] = {
            'final_eval_loss': eval_loss.item(),
            'params': n_params,
        }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Verify diffusion pipeline components')
    p.add_argument('--exp', type=str, default='v1',
                   help='Experiments to run: v1,v2,v3,v4,v5,v6,v7,v8 or "all"')
    p.add_argument('--encoder_ckpt', type=str,
                   default='/home/yexiaoyu/work/4DLangVGGT/ckpt/streamvggt/checkpoints.pth')
    p.add_argument('--token_stats', type=str,
                   default='/home/yexiaoyu/work/VggAE-Diffusion/ckpts/token_stats.pt')
    p.add_argument('--decoder_ckpt', type=str,
                   default='/home/yexiaoyu/work/VggAE-Diffusion/ckpts/decoder_dpt/exp-5-dpt/decoder_final.pt')
    p.add_argument('--csv', type=str,
                   default='/public2/LiZhen/yexiaoyu/dataset/spatial-vid-hq-oft/data/train/SpatialVID_HQ_metadata.csv')
    p.add_argument('--video_root', type=str,
                   default='/public2/LiZhen/yexiaoyu/dataset/spatial-vid-hq-oft/videos/SpatialVID/videos')
    p.add_argument('--max_videos', type=int, default=200,
                   help='Max videos for verification (keep small for speed)')
    p.add_argument('--steps', type=int, default=2000,
                   help='Training steps per experiment')
    p.add_argument('--output', type=str, default='',
                   help='Output JSON for results')
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print(f"=== Diffusion Pipeline Verification ===")
    print(f"Device: {device}")
    print(f"Max videos: {args.max_videos}, Steps: {args.steps}")

    exps = args.exp.split(',') if args.exp != 'all' else ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8']
    all_results = {}

    # Load common resources (shared across experiments)
    print(f"\n{'='*60}")
    print(f"Loading shared resources...")
    print(f"{'='*60}")

    print(f"  Encoder: {args.encoder_ckpt}")
    encoder = load_encoder(args.encoder_ckpt, device)

    level_stats = load_token_stats(args.token_stats, device, dtype=torch.float32)
    print(f"  Token stats: {args.token_stats}")

    dataset = SpatialVidDataset(
        csv_path=args.csv, video_root=args.video_root,
        seq_len=8, target_size=518,
        max_videos=args.max_videos,
        num_frames_per_video=8,
    )
    print(f"  Dataset: {len(dataset)} videos")

    # -------------------------------------------------------------------
    # V1: Single-token flow matching
    # -------------------------------------------------------------------
    if 'v1' in exps:
        print(f"\n{'='*60}")
        print(f"V1: Single-token flow matching (MLP, center token)")
        print(f"  Question: Can a simple MLP learn E[v|zt,t] in 2048-dim?")
        print(f"  If NO → flow matching in token space is fundamentally broken.")
        print(f"  If YES → problem is architecture/Wan/spatial, not the paradigm.")
        print(f"{'='*60}")

        r = run_single_token_experiment(
            encoder, level_stats, dataset, device,
            num_steps=args.steps, model_type='mlp', time_mode='sinusoidal',
        )
        all_results['v1'] = {
            'initial_loss': r['initial_loss_approx'],
            'final_eval_loss': r['final_eval_loss'],
            'loss_reduction': (r['initial_loss_approx'] - r['final_eval_loss']) / r['initial_loss_approx'] * 100
                if r['initial_loss_approx'] and r['final_eval_loss'] else None,
            'time_variance': r['time_variance'],
            'final_sample_std': r['sample_stats'][-1]['sample_std'] if r['sample_stats'] else None,
        }
        print(f"\n  V1 RESULT: initial_loss={r['initial_loss_approx']:.4f}, "
              f"final_eval_loss={r['final_eval_loss']:.4f}, "
              f"reduction={all_results['v1']['loss_reduction']:.1f}%")
        print(f"  Time variance: {r['time_variance']:.6f} "
              f"({'GOOD' if r['time_variance'] > 0.01 else 'BAD — time embedding broken'})")

    # -------------------------------------------------------------------
    # V2: Time embedding ablation
    # -------------------------------------------------------------------
    if 'v2' in exps:
        print(f"\n{'='*60}")
        print(f"V2: Time embedding ablation")
        print(f"  Question: Which time embedding works best?")
        print(f"  Compares: sinusoidal (current), fourier, learned, direct")
        print(f"{'='*60}")

        time_results = {}
        for mode in ['sinusoidal', 'fourier', 'learned', 'direct']:
            print(f"\n  --- Time mode: {mode} ---")
            r = run_single_token_experiment(
                encoder, level_stats, dataset, device,
                num_steps=args.steps, model_type='mlp', time_mode=mode,
            )
            time_results[mode] = {
                'final_eval_loss': r['final_eval_loss'],
                'time_variance': r['time_variance'],
            }
            print(f"  {mode}: loss={r['final_eval_loss']:.4f}, time_var={r['time_variance']:.6f}")

        all_results['v2'] = time_results

        best = min(time_results, key=lambda k: time_results[k]['final_eval_loss'])
        print(f"\n  V2 RESULT: Best time embedding = {best}")

    # -------------------------------------------------------------------
    # V3: Architecture comparison (single token)
    # -------------------------------------------------------------------
    if 'v3' in exps:
        print(f"\n{'='*60}")
        print(f"V3: Architecture comparison (single token)")
        print(f"  Question: Which architecture + conditioning works best?")
        print(f"  mlp_concat (baseline) vs mlp_adaln vs mlp_deep vs mlp_wide")
        print(f"{'='*60}")

        arch_results = {}
        configs = [
            ('mlp_concat', 'mlp', 'sinusoidal', 'concat', 4, 1024),
            ('mlp_adaln',  'mlp', 'sinusoidal', 'adaln',  4, 1024),
            ('mlp_deep',   'mlp', 'sinusoidal', 'concat', 8, 1024),
            ('mlp_wide',   'mlp', 'sinusoidal', 'concat', 4, 2048),
        ]
        for label, mtype, tmode, cmode, nlayers, hdim in configs:
            print(f"\n  --- {label} (layers={nlayers}, hidden={hdim}, cond={cmode}) ---")
            r = run_single_token_experiment(
                encoder, level_stats, dataset, device,
                num_steps=args.steps, model_type=mtype, time_mode=tmode,
                cond_mode=cmode, num_layers=nlayers, hidden_dim=hdim,
            )
            arch_results[label] = {
                'final_eval_loss': r['final_eval_loss'],
                'time_variance': r['time_variance'],
                'params': sum(p.numel() for p in
                    TokenMLP(token_dim=2048, hidden_dim=hdim, num_layers=nlayers,
                             time_mode=tmode, cond_mode=cmode).parameters()),
            }
            print(f"  {label}: loss={r['final_eval_loss']:.4f}, time_var={r['time_variance']:.6f}")

        all_results['v3'] = arch_results

        best = min(arch_results, key=lambda k: arch_results[k]['final_eval_loss'])
        print(f"\n  V3 RESULT: Best architecture = {best}")

    # -------------------------------------------------------------------
    # V4: Spatial structure (single frame, full grid)
    # -------------------------------------------------------------------
    if 'v4' in exps:
        print(f"\n{'='*60}")
        print(f"V4: Spatial structure necessity")
        print(f"  Question: Does spatial attention help over per-token MLP?")
        print(f"  If per-token MLP ≈ spatial DiT → tokens are spatially independent")
        print(f"{'='*60}")

        spatial_results = {}
        for use_spatial in [False, True]:
            label = 'spatial_dit' if use_spatial else 'per_token_mlp'
            print(f"\n  --- Model: {label} ---")
            r = run_spatial_experiment(
                encoder, level_stats, dataset, device,
                num_steps=args.steps,
                use_spatial=use_spatial,
            )
            spatial_results[label] = {'final_eval_loss': r['final_eval_loss']}
            print(f"  {label}: loss={r['final_eval_loss']:.4f}")

        all_results['v4'] = spatial_results

        gap = (spatial_results['per_token_mlp']['final_eval_loss'] -
               spatial_results['spatial_dit']['final_eval_loss'])
        print(f"\n  V4 RESULT: Spatial DiT vs per-token MLP gap = {gap:+.4f}")
        print(f"  {'Spatial context HELPS' if gap > 0.02 else 'Spatial context does NOT help significantly'}")

    # -------------------------------------------------------------------
    # V5: Multi-level target comparison
    # -------------------------------------------------------------------
    if 'v5' in exps:
        print(f"\n{'='*60}")
        print(f"V5: Multi-level target comparison")
        print(f"  Question: Single level vs mean([4,11,17,23]) — which is easier to learn?")
        print(f"{'='*60}")

        r = run_multilevel_experiment(
            encoder, level_stats, dataset, device,
            num_steps=args.steps,
        )
        all_results['v5'] = r

        for label in ['single_level', 'multi_level_mean']:
            d = r[label]
            print(f"  {label}: loss={d['final_eval_loss']:.4f}, "
                  f"target_var={d['target_variance']:.4f}, "
                  f"data_std={d['data_std']:.4f}")

        # Normalize loss by target variance for fair comparison
        for label in ['single_level', 'multi_level_mean']:
            d = r[label]
            normalized = d['final_eval_loss'] / d['target_variance']
            print(f"  {label}: loss/target_var = {normalized:.4f}")

    # -------------------------------------------------------------------
    # V6: Decoder sensitivity
    # -------------------------------------------------------------------
    if 'v6' in exps:
        print(f"\n{'='*60}")
        print(f"V6: Decoder sensitivity to token noise")
        print(f"  Question: How accurate must generated tokens be?")
        print(f"{'='*60}")

        r = run_decoder_sensitivity(
            encoder, level_stats, dataset, device,
            decoder_ckpt_path=args.decoder_ckpt,
        )
        all_results['v6'] = r

        print(f"\n  V6 RESULT: Clean PSNR = {r['clean_psnr']:.2f} dB")
        print(f"  Acceptable noise std (PSNR drop < 2dB): {r['acceptable_noise_std']}")
        print(f"  Multi-level mean as level-11 PSNR: {r['mean_as_level11_psnr']:.2f} dB")

    # -------------------------------------------------------------------
    # V7: Spatiotemporal DiT (full 8 frames)
    # -------------------------------------------------------------------
    if 'v7' in exps:
        print(f"\n{'='*60}")
        print(f"V7: Spatiotemporal DiT (factorized spatial-temporal attention)")
        print(f"  Question: Does temporal attention help?")
        print(f"  Compare: per-token MLP vs spatial-only vs spatial+temporal")
        print(f"{'='*60}")

        r = run_spatiotemporal_experiment(
            encoder, level_stats, dataset, device,
            num_steps=args.steps, seq_len=8, batch_size=4,
        )
        all_results['v7'] = r

        for label in ['per_token_mlp', 'spatial_only', 'spatiotemporal']:
            d = r[label]
            print(f"  {label}: loss={d['final_eval_loss']:.4f}, params={d['params']:,}")

        gap = r['spatial_only']['final_eval_loss'] - r['spatiotemporal']['final_eval_loss']
        print(f"\n  V7 RESULT: Spatial-temporal vs spatial-only gap = {gap:+.4f}")
        print(f"  {'Temporal attention HELPS' if gap > 0.02 else 'Temporal attention does NOT help significantly'}")

    # -------------------------------------------------------------------
    # V8: Frame count comparison
    # -------------------------------------------------------------------
    if 'v8' in exps:
        print(f"\n{'='*60}")
        print(f"V8: Frame count comparison")
        print(f"  Question: What seq_len works best?")
        print(f"  Compare: 1, 4, 8, 16 frames")
        print(f"{'='*60}")

        frame_results = {}
        for seq_len in [1, 4, 8, 16]:
            print(f"\n  --- seq_len={seq_len} ---")
            r = run_spatiotemporal_experiment(
                encoder, level_stats, dataset, device,
                num_steps=args.steps, seq_len=seq_len, batch_size=max(1, 8 // seq_len),
            )
            key = f'S={seq_len}'
            frame_results[key] = r['spatiotemporal']
            print(f"  {key}: loss={r['spatiotemporal']['final_eval_loss']:.4f}")

        all_results['v8'] = frame_results

        # Check how loss scales with frame count
        print(f"\n  V8 RESULT: Loss vs frame count:")
        for k, v in frame_results.items():
            print(f"    {k}: {v['final_eval_loss']:.4f}")

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"VERIFICATION SUMMARY")
    print(f"{'='*60}")
    print(json.dumps(all_results, indent=2, default=str))

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
