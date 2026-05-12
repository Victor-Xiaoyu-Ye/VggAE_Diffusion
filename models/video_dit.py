"""Video DiT for flow matching in StreamVGGT token space.

Architecture overview:
  Input  : [B, T, N, D]   (T = temporal, N = spatial patches, D = token_dim)
  Output : [B, T, N, D]   (predicted velocity)

  1. Token projection:   D -> hidden_dim
  2. Add spatial + temporal (+ level) positional embeddings
  3. N DiT blocks (spatial attn -> cross attn -> temporal attn -> MLP, AdaLN-Zero)
  4. Output projection:  hidden_dim -> D

Each DiT block:
  - Spatial self-attention   (within each frame, across all patches)
  - Cross-attention          (to CLIP text embeddings, optional)
  - Temporal self-attention  (across T positions, per spatial location)
  - SwiGLU MLP
  All conditioned on timestep via AdaLN-Zero.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# Positional embeddings
# ======================================================================

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([args.cos(), args.sin()], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.sinusoidal = SinusoidalEmbedding(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        dtype = self.mlp[0].weight.dtype
        return self.mlp(self.sinusoidal(t).to(dtype))


# ======================================================================
# AdaLN-Zero modulation
# ======================================================================

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class AdaLNZeroModulation(nn.Module):
    def __init__(self, hidden_dim: int, num_vectors: int = 9):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_vectors = num_vectors
        self.linear = nn.Linear(hidden_dim, num_vectors * hidden_dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, cond: torch.Tensor):
        out = self.linear(F.silu(cond))
        return out.chunk(self.num_vectors, dim=-1)


# ======================================================================
# Attention
# ======================================================================

class Attention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, causal: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.causal = causal
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_mask = None
        if self.causal and L > 1:
            attn_mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)

        try:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=self.scale)
        except Exception:
            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if attn_mask is not None:
                attn = attn.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            attn = F.softmax(attn, dim=-1)
            out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(B, L, D)
        return self.proj(out)


class CrossAttention(nn.Module):
    """Cross-attention from spatial tokens to CLIP text embeddings."""

    def __init__(self, hidden_dim: int, num_heads: int, text_dim: int = 768):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(text_dim, hidden_dim)
        self.v = nn.Linear(text_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """x: [BT, N, D], text_emb: [B, L_text, text_dim]"""
        BT, N, D = x.shape
        B = text_emb.shape[0]
        T = BT // B
        Hd = self.head_dim
        L_text = text_emb.shape[1]

        q = self.q(x).reshape(BT, N, self.num_heads, Hd).transpose(1, 2)  # [BT, H, N, Hd]
        kt = self.k(text_emb).reshape(B, L_text, self.num_heads, Hd).transpose(1, 2)  # [B, H, L_text, Hd]
        vt = self.v(text_emb).reshape(B, L_text, self.num_heads, Hd).transpose(1, 2)  # [B, H, L_text, Hd]

        # Expand k,v across T: [B, H, L_text, Hd] -> [B*T, H, L_text, Hd]
        kt = kt.unsqueeze(1).expand(B, T, self.num_heads, L_text, Hd).contiguous().reshape(BT, self.num_heads, L_text, Hd)
        vt = vt.unsqueeze(1).expand(B, T, self.num_heads, L_text, Hd).contiguous().reshape(BT, self.num_heads, L_text, Hd)

        out = F.scaled_dot_product_attention(q, kt, vt, scale=self.scale)
        out = out.transpose(1, 2).reshape(BT, N, D)
        return self.proj(out)


# ======================================================================
# MLP (SwiGLU)
# ======================================================================

class SwiGLUMLP(nn.Module):
    def __init__(self, hidden_dim: int, mlp_ratio: int = 4):
        super().__init__()
        inner = hidden_dim * mlp_ratio
        self.fc1 = nn.Linear(hidden_dim, inner)
        self.fc2 = nn.Linear(inner, hidden_dim)
        self.gate = nn.Linear(hidden_dim, inner)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.silu(self.fc1(x)) * self.gate(x))


# ======================================================================
# DiT Block
# ======================================================================

class DiTBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: int = 4,
                 causal_temporal: bool = False, use_cross_attn: bool = False,
                 text_dim: int = 768):
        super().__init__()
        self.use_cross_attn = use_cross_attn

        self.spatial_attn_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.spatial_attn = Attention(hidden_dim, num_heads)

        if use_cross_attn:
            self.cross_attn_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
            self.cross_attn = CrossAttention(hidden_dim, num_heads, text_dim=text_dim)
            self.cross_attn_gate = nn.Parameter(torch.zeros(1))

        self.temporal_attn_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.temporal_attn = Attention(hidden_dim, num_heads, causal=causal_temporal)
        self.mlp_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = SwiGLUMLP(hidden_dim, mlp_ratio)

        # 9 vectors (no cross-attn) or 12 vectors (with cross-attn)
        num_mod = 12 if use_cross_attn else 9
        self.modulation = AdaLNZeroModulation(hidden_dim, num_vectors=num_mod)

    def forward(self, x: torch.Tensor, cond: torch.Tensor,
                text_emb: torch.Tensor = None) -> torch.Tensor:
        """x: [B, T, N, D], cond: [B, D]"""
        B, T, N, D = x.shape

        mods = self.modulation(cond)
        if self.use_cross_attn:
            (s1, sc1, g1, s_x, sc_x, g_x, s2, sc2, g2, s3, sc3, g3) = mods
        else:
            (s1, sc1, g1, s2, sc2, g2, s3, sc3, g3) = mods

        # Broadcast modulation from [B, D] to [B, 1, 1, D]
        def bcast(v):
            return v[:, None, None]

        # --- Spatial self-attention ---
        x_s = x.reshape(B * T, N, D)
        x_s = self.spatial_attn_norm(x_s).reshape(B, T, N, D)
        x_s = x_s * (1 + bcast(sc1)) + bcast(s1)
        x_s = self.spatial_attn(x_s.reshape(B * T, N, D)).reshape(B, T, N, D)
        x = x + g1[:, None, None] * x_s

        # --- Cross-attention (optional) ---
        if self.use_cross_attn and text_emb is not None:
            x_c = self.cross_attn_norm(x.reshape(B * T, N, D)).reshape(B, T, N, D)
            x_c = x_c * (1 + bcast(sc_x)) + bcast(s_x)
            x_c = self.cross_attn(x_c.reshape(B * T, N, D), text_emb).reshape(B, T, N, D)
            x = x + self.cross_attn_gate * x_c

        # --- Temporal self-attention ---
        x_t = x.permute(0, 2, 1, 3).reshape(B * N, T, D)
        x_t = self.temporal_attn_norm(x_t).reshape(B, N, T, D)
        x_t = x_t * (1 + bcast(sc2)) + bcast(s2)
        x_t = self.temporal_attn(x_t.reshape(B * N, T, D)).reshape(B, N, T, D).permute(0, 2, 1, 3)
        x = x + g2[:, None, None] * x_t

        # --- MLP ---
        x_m = x.reshape(B * T * N, D)
        x_m = self.mlp_norm(x_m).reshape(B, T, N, D)
        x_m = x_m * (1 + bcast(sc3)) + bcast(s3)
        x_m = self.mlp(x_m.reshape(B * T * N, D)).reshape(B, T, N, D)
        x = x + g3[:, None, None] * x_m

        return x


# ======================================================================
# Full VideoDiT
# ======================================================================

class VideoDiT(nn.Module):
    def __init__(
        self,
        token_dim: int = 2048,
        hidden_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: int = 4,
        num_levels: int = 1,
        seq_len: int = 8,
        patch_size: int = 14,
        img_size: int = 518,
        causal_temporal: bool = False,
        use_cross_attn: bool = False,
        text_dim: int = 768,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        self.seq_len = seq_len
        self.use_cross_attn = use_cross_attn
        self.use_checkpoint = use_checkpoint

        patch_h = img_size // patch_size
        patch_w = img_size // patch_size
        self.num_patches = patch_h * patch_w
        self.T = num_levels * seq_len

        # Projections
        self.input_proj = nn.Linear(token_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, token_dim)

        # Timestep embedder
        self.t_embedder = TimestepEmbedder(hidden_dim)

        # Positional embeddings
        self.spatial_pos_emb = nn.Parameter(torch.zeros(1, 1, self.num_patches, hidden_dim))
        nn.init.trunc_normal_(self.spatial_pos_emb, std=0.02)
        self.temporal_pos_emb = nn.Parameter(torch.zeros(1, self.T, 1, hidden_dim))
        nn.init.trunc_normal_(self.temporal_pos_emb, std=0.02)

        # Level embedding (for multi-level)
        if num_levels > 1:
            self.level_emb = nn.Parameter(torch.zeros(1, num_levels, 1, hidden_dim))
            nn.init.trunc_normal_(self.level_emb, std=0.02)

        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, mlp_ratio, causal_temporal, use_cross_attn, text_dim)
            for _ in range(num_layers)
        ])

        # Final norm
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)

        # Init
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if "output_proj" in name or "modulation" in name:
                    continue
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor, t: torch.Tensor,
                cond: torch.Tensor = None, text_emb: torch.Tensor = None) -> torch.Tensor:
        """Predict velocity. z: [B, T, N, D], t: [B], text_emb: [B, L, text_dim]"""
        B = z.shape[0]
        assert z.shape == (B, self.T, self.num_patches, self.token_dim), \
            f"Expected {(B, self.T, self.num_patches, self.token_dim)}, got {z.shape}"

        t_emb = self.t_embedder(t).to(z.dtype)

        x = self.input_proj(z)
        x = x + self.spatial_pos_emb + self.temporal_pos_emb
        if self.num_levels > 1:
            # select_levels uses level-major order: [lvl0_t0..tS, lvl1_t0..tS, ...].
            level_emb = self.level_emb.repeat_interleave(self.seq_len, dim=1)
            x = x + level_emb

        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, t_emb, text_emb, use_reentrant=False,
                )
            else:
                x = block(x, t_emb, text_emb=text_emb)

        x = self.final_norm(x)
        return self.output_proj(x)


def build_video_dit(token_dim=2048, hidden_dim=768, num_layers=12, num_heads=12,
                    num_levels=1, seq_len=8, patch_size=14, img_size=518,
                    causal=False, use_cross_attn=False, text_dim=768):
    return VideoDiT(
        token_dim=token_dim, hidden_dim=hidden_dim, num_layers=num_layers,
        num_heads=num_heads, num_levels=num_levels, seq_len=seq_len,
        patch_size=patch_size, img_size=img_size, causal_temporal=causal,
        use_cross_attn=use_cross_attn, text_dim=text_dim,
    )
