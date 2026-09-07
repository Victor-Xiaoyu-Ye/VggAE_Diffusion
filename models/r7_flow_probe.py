"""Small conditional flow model; R7 remains the only generation/decoding space."""
from __future__ import annotations

import math
import torch
import torch.nn as nn

FLOW_SCHEMA = 'r7-prefix-flow-v1'


def coefficients(t):
    d = t.square() + (1-t).square()
    return d, d.rsqrt(), t / d, (1-t) * d.rsqrt()


def network_target(data, noise, t, prediction):
    if prediction == 'plain_x0':
        return data
    if prediction != 'preconditioned':
        raise ValueError('unknown prediction contract')
    te = t.float().view(-1, 1, 1, 1)
    d, _, _, _ = coefficients(te)
    return ((1-te)*data.float() - te*noise.float()) / d.sqrt()


def clean_prediction(output, noisy, t, prediction):
    if prediction == 'plain_x0':
        return output.float()
    te = t.float().view(-1, 1, 1, 1)
    _, _, skip, out = coefficients(te)
    return skip*noisy.float() + out*output.float()


def velocity_prediction(output, noisy, t, prediction):
    te = t.float().view(-1, 1, 1, 1)
    if prediction == 'plain_x0':
        return (output.float()-noisy.float()) / (1-te).clamp_min(1e-6)
    d, _, _, _ = coefficients(te)
    # Algebraic form avoids xhat-x cancellation and division by 1-t at t=1.
    return ((2*te-1)/d)*noisy.float() + output.float()/d.sqrt()


def time_features(t, dim=128):
    freq = torch.exp(torch.arange(dim//2, device=t.device, dtype=torch.float32)
                     * (-math.log(10000.) / (dim//2-1)))
    phase = 1000*t.float()[:, None]*freq[None]
    return torch.cat((phase.sin(), phase.cos()), -1)


class FlowBlock(nn.Module):
    def __init__(self, width, heads, text_cond):
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
        self.ffnorm = nn.LayerNorm(width, elementwise_affine=False)
        self.ff = nn.Sequential(nn.Linear(width, width*4), nn.GELU(), nn.Linear(width*4, width))
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(width, width*6))
        self.cross = (nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
                      if text_cond else None)
        self.crossnorm = nn.LayerNorm(width) if text_cond else None
        nn.init.zeros_(self.mod[-1].weight)
        nn.init.zeros_(self.mod[-1].bias)

    def forward(self, x, time, text=None, text_mask=None):
        shift, scale, gate, fs, fc, fg = self.mod(time).chunk(6, -1)
        q = self.norm(x)*(1+scale) + shift
        x = x + gate*self.attn(q, q, q, need_weights=False)[0]
        if self.cross is not None:
            x = x + self.cross(self.crossnorm(x), text, text,
                               key_padding_mask=text_mask, need_weights=False)[0]
        return x + fg*self.ff(self.ffnorm(x)*(1+fc)+fs)


class R7FlowProbe(nn.Module):
    def __init__(self, latent_dim=192, num_tokens=324, future_frames=1,
                 hidden_dim=384, depth=4, num_heads=6,
                 prediction='plain_x0', text_cond=False):
        super().__init__()
        if hidden_dim % num_heads or future_frames not in (1, 2, 4, 8):
            raise ValueError('invalid heads or future prefix length')
        if prediction not in ('plain_x0', 'preconditioned'):
            raise ValueError('unknown prediction contract')
        self.latent_dim, self.num_tokens = latent_dim, num_tokens
        self.future_frames, self.prediction = future_frames, prediction
        self.text_cond = text_cond
        self.input = nn.Linear(latent_dim, hidden_dim)
        self.anchor = nn.Linear(latent_dim, hidden_dim)
        self.spatial_pos = nn.Parameter(torch.randn(1, 1, num_tokens, hidden_dim)*.02)
        self.temporal_pos = nn.Parameter(torch.randn(1, 1+future_frames, 1, hidden_dim)*.02)
        self.token_type = nn.Embedding(2, hidden_dim)
        self.time = nn.Sequential(nn.Linear(128, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.text_proj = nn.Linear(4096, hidden_dim) if text_cond else None
        self.blocks = nn.ModuleList([FlowBlock(hidden_dim, num_heads, text_cond) for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, latent_dim)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, noisy, t, anchor, text=None, text_mask=None):
        b, frames, n, d = noisy.shape
        if (frames, n, d) != (self.future_frames, self.num_tokens, self.latent_dim):
            raise ValueError('noisy target does not match model contract')
        if anchor.shape != (b, 1, n, d):
            raise ValueError('anchor must be [B,1,N,D]')
        if self.text_cond and (text is None or text.shape[0] != b or text.shape[-1] != 4096):
            raise ValueError('text-conditioned model requires UMT5 embeddings')
        xt = noisy.float()
        if self.prediction == 'preconditioned':
            _, ci, _, _ = coefficients(t.float().view(-1, 1, 1, 1))
            xt = xt*ci
        x = torch.cat((self.anchor(anchor.float()), self.input(xt)), 1)
        ids = torch.tensor([0]+[1]*frames, device=x.device)
        x = x+self.spatial_pos+self.temporal_pos+self.token_type(ids)[None, :, None]
        t_embed = self.time(time_features(t))
        clean_embed = self.time(time_features(torch.ones_like(t)))
        mod = torch.cat((clean_embed[:, None, None].expand(-1, 1, n, -1),
                         t_embed[:, None, None].expand(-1, frames, n, -1)), 1)
        x, mod = x.flatten(1, 2), mod.flatten(1, 2)
        context = self.text_proj(text.float()) if self.text_cond else None
        for block in self.blocks:
            x = block(x, mod, context, text_mask)
        return self.head(self.norm(x)).reshape(b, frames+1, n, d)[:, 1:].float()


@torch.no_grad()
def sample_flow(model, anchor, noise, steps=30, text=None, text_mask=None,
                dtype=torch.float32, grid_alpha=1.):
    if steps < 1 or grid_alpha <= 0:
        raise ValueError('positive sampling steps and grid alpha required')
    z = noise.float().clone()
    u = torch.linspace(0, 1, steps+1, device=z.device, dtype=torch.float32)
    grid = grid_alpha*u/(1+(grid_alpha-1)*u)
    for left, right in zip(grid[:-1], grid[1:]):
        t = torch.full((z.shape[0],), float(left), device=z.device)
        with torch.autocast(device_type=z.device.type, dtype=dtype, enabled=dtype != torch.float32):
            out = model(z, t, anchor, text, text_mask)
        z = z+(right-left)*velocity_prediction(out, z, t, model.prediction)
    return z
