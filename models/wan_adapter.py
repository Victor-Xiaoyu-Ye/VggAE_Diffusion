"""Wan2.1 1.3B adapter: maps VGGT tokens through Wan's pretrained DiT backbone.

Two modes:
  Full fine-tune: all Wan params trainable (1.43B, need FSDP)
  LoRA: attention QKV adapted, FFN frozen (~66M trainable)

Flow:
  VGGT [B, S=8, N=1369, 2048]
    → Input Proj (2048→1536) [full training]
    → Wan 30-layer DiT [LoRA on attn QKV]
    → Output Proj (1536→2048) [full training]
"""

import torch
import torch.nn as nn
import torch.amp as amp

import sys, os
_wan_root = os.path.join(os.path.dirname(__file__), '..', 'Wan2.1')
if _wan_root not in sys.path:
    sys.path.insert(0, _wan_root)
from wan.modules.model import WanModel, sinusoidal_embedding_1d


# ---------------------------------------------------------------------------
# Minimal LoRA wrapper (no peft dependency)
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """LoRA adapter wrapping a frozen nn.Linear. y = Wx + (alpha/r) * BAx."""

    def __init__(self, linear: nn.Linear, rank: int = 64, alpha: float = 128):
        super().__init__()
        self.linear = linear           # frozen original
        self.rank = rank
        self.alpha = alpha
        in_dim, out_dim = linear.in_features, linear.out_features

        # LoRA params
        self.lora_A = nn.Parameter(torch.randn(rank, in_dim) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_dim, rank))

        # Freeze original
        linear.weight.requires_grad_(False)
        if linear.bias is not None:
            linear.bias.requires_grad_(False)

        self.scaling = alpha / rank

    def forward(self, x):
        y = self.linear(x)
        delta = (x @ self.lora_A.T) @ self.lora_B.T
        return y + delta * self.scaling


# ---------------------------------------------------------------------------
# Wan VGGT Adapter
# ---------------------------------------------------------------------------
class WanVGGTAdapter(nn.Module):

    def __init__(self, wan_checkpoint_dir, vggt_token_dim=2048, seq_len=8,
                 img_size=518, patch_size=14, lora_rank=64, lora_alpha=128):
        super().__init__()

        # Load pretrained Wan 1.3B
        self.wan = WanModel.from_pretrained(wan_checkpoint_dir)
        wan_dim = self.wan.dim          # 1536
        freq_dim = self.wan.freq_dim    # 256

        self.seq_len = seq_len
        self.num_patches = (img_size // patch_size) ** 2  # 1369
        self.grid_size = int(self.num_patches ** 0.5)     # 37
        self.token_dim = vggt_token_dim
        self.wan_dim = wan_dim
        self.freq_dim = freq_dim
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

        # ---- Input adapter: VGGT 2048 → Wan 1536 [full training] ----
        self.input_proj = nn.Sequential(
            nn.Linear(vggt_token_dim, wan_dim),
            nn.LayerNorm(wan_dim),
            nn.SiLU(),
            nn.Linear(wan_dim, wan_dim),
        )

        # ---- Output adapter: Wan 1536 → VGGT 2048 [full training] ----
        self.output_norm = nn.LayerNorm(wan_dim)
        self.output_proj = nn.Linear(wan_dim, vggt_token_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        # ---- Text projection: CLIP 768 → Wan 1536 ----
        self.text_proj = nn.Sequential(
            nn.Linear(768, wan_dim),
            nn.GELU(),
            nn.Linear(wan_dim, wan_dim),
        )
        self.text_dim = 768

        # ---- Freeze all non-adapter Wan params ----
        self._freeze_wan()

        # ---- Apply LoRA to attention QKV ----
        if lora_rank > 0:
            self._apply_lora(lora_rank, lora_alpha)

        self._time_emb_converted = False

    def _freeze_wan(self):
        """Freeze all Wan parameters except those we'll explicitly unfreeze."""
        for p in self.wan.parameters():
            p.requires_grad_(False)

    def _apply_lora(self, rank, alpha):
        """Wrap attention QKV Linear layers in each Wan block with LoRA."""
        for blk in self.wan.blocks:
            attn = blk.self_attn
            attn.q = LoRALinear(attn.q, rank=rank, alpha=alpha)
            attn.k = LoRALinear(attn.k, rank=rank, alpha=alpha)
            attn.v = LoRALinear(attn.v, rank=rank, alpha=alpha)

    def _ensure_time_emb_float32(self):
        if self._time_emb_converted:
            return
        for m in [self.wan.time_embedding, self.wan.time_projection]:
            for p in m.parameters():
                p.data = p.data.float()
        self._time_emb_converted = True

    def _time_embed(self, t):
        self._ensure_time_emb_float32()
        with amp.autocast(device_type='cuda', enabled=False):
            e = sinusoidal_embedding_1d(self.freq_dim, t).float().to(device=t.device)
            e = self.wan.time_embedding(e.float())
            e = self.wan.time_projection(e.float())
        return e.unflatten(1, (6, self.wan_dim)).float()

    def set_lora_trainable(self):
        """Set only LoRA params + i/o adapters + time embedding as trainable."""
        for p in self.parameters():
            p.requires_grad_(False)
        for p in self.input_proj.parameters():
            p.requires_grad_(True)
        for p in self.output_norm.parameters():
            p.requires_grad_(True)
        for p in self.output_proj.parameters():
            p.requires_grad_(True)
        for n, p in self.named_parameters():
            if 'lora_A' in n or 'lora_B' in n:
                p.requires_grad_(True)
        # Also train time embedding + text projection
        for p in self.wan.time_embedding.parameters():
            p.requires_grad_(True)
        for p in self.wan.time_projection.parameters():
            p.requires_grad_(True)
        for p in self.text_proj.parameters():
            p.requires_grad_(True)

    def forward(self, z, t, cond=None, text_emb=None):
        B, S, N, D = z.shape

        # ---- Input projection ----
        x = self.input_proj(z)                              # [B, S, N, wan_dim]
        x = x.reshape(B, S * N, self.wan_dim)               # [B, S*N, wan_dim]

        # ---- Time embedding ----
        t_wan = (t * 1000).to(device=z.device)
        e = self._time_embed(t_wan)

        # ---- Grid setup ----
        grid_sizes = torch.tensor(
            [[S, self.grid_size, self.grid_size]],
            device=z.device, dtype=torch.long
        ).repeat(B, 1)
        seq_lens = torch.full((B,), S * N, device=z.device, dtype=torch.long)

        # ---- Text conditioning (CLIP 768 → wan_dim 1536) ----
        context, context_lens = None, None
        if text_emb is not None:
            context = self.text_proj(text_emb.to(x.dtype))
            context_lens = torch.full((B,), context.shape[1], device=x.device, dtype=torch.long)

        # ---- Wan DiT blocks ----
        if self.wan.freqs.device != x.device:
            self.wan.freqs = self.wan.freqs.to(x.device)

        for block in self.wan.blocks:
            e_dtype = x.dtype
            e6 = (block.modulation.to(e_dtype) + e.to(e_dtype)).chunk(6, dim=1)

            # self-attention
            y = block.self_attn(
                block.norm1(x) * (1 + e6[1]) + e6[0],
                seq_lens, grid_sizes, self.wan.freqs)
            x = x + y * e6[2].to(x.dtype)

            # cross-attn (skip if no text)
            if context is not None:
                x = x + block.cross_attn(block.norm3(x), context, context_lens)

            # FFN
            y = block.ffn(block.norm2(x) * (1 + e6[4].to(x.dtype)) + e6[3].to(x.dtype))
            x = x + y * e6[5].to(x.dtype)

        # ---- Output projection ----
        x = self.output_norm(x)
        x = self.output_proj(x)                             # [B, S*N, token_dim]
        x = x.reshape(B, S, N, self.token_dim)              # [B, S, N, 2048]

        return x
