"""Full-window conditional R7 DiT, independent of the unfinished n1 probes."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def sinusoid(value, width):
    freq = torch.exp(-math.log(10000.) * torch.arange(width//2, device=value.device).float()
                     / max(1, width//2-1))
    phase = value.float()[..., None]*freq
    return torch.cat((phase.sin(), phase.cos()), -1)


def position(frames, grid, width, factor=1, anchor=False):
    t = torch.zeros(1) if anchor else torch.arange(1, frames+1).float()*factor
    coords = torch.meshgrid(t, torch.arange(grid).float(), torch.arange(grid).float(), indexing='ij')
    axis = 2*(width//6)
    pos = torch.cat([sinusoid(c.flatten(), axis) for c in coords], -1)
    return F.pad(pos, (0, width-pos.shape[-1]))[None]


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads, self.head_dim = heads, width//heads
        self.q = nn.Linear(width, width)
        self.kv = nn.Linear(width, width*2)
        self.out = nn.Linear(width, width)

    def forward(self, x, context=None, valid=None):
        context = x if context is None else context
        b, n, w = x.shape
        q = self.q(x).reshape(b, n, self.heads, self.head_dim).transpose(1, 2)
        k, v = self.kv(context).reshape(b, -1, 2, self.heads, self.head_dim).unbind(2)
        mask = None if valid is None else valid[:, None, None, :]
        value = F.scaled_dot_product_attention(q, k.transpose(1, 2), v.transpose(1, 2),
                                               attn_mask=mask, dropout_p=0.)
        return self.out(value.transpose(1, 2).reshape(b, n, w))


class WindowBlock(nn.Module):
    def __init__(self, width, heads, text_dim):
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.self_attn = Attention(width, heads)
        self.anchor_attn = Attention(width, heads)
        self.text_attn = Attention(width, heads) if text_dim else None
        self.ff = nn.Sequential(nn.Linear(width, width*4), nn.GELU(), nn.Linear(width*4, width))
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(width, width*9))
        nn.init.zeros_(self.mod[-1].weight)
        nn.init.zeros_(self.mod[-1].bias)

    def forward(self, x, time, anchor, text=None, valid=None):
        s, c, gate, fs, fc, fg, ag, ts, tg = self.mod(time).chunk(9, -1)
        x = x+gate*self.self_attn(self.norm(x)*(1+c)+s)
        x = x+ag*self.anchor_attn(self.norm(x), anchor)
        if self.text_attn is not None:
            x = x+tg*self.text_attn(self.norm(x)+ts, text, valid)
        return x+fg*self.ff(self.norm(x)*(1+fc)+fs)


class R7WindowDiT(nn.Module):
    def __init__(self, channels=192, grid=18, future=4, temporal_factor=2,
                 width=768, depth=12, heads=12, text_dim=4096, checkpoint_blocks=True, aux_layer=0):
        super().__init__()
        if min(channels, grid, future, width, depth, heads) < 1 or width % heads:
            raise ValueError('positive dimensions and divisible heads required')
        self.channels, self.grid, self.future = channels, grid, future
        self.text_dim, self.checkpoint_blocks = text_dim, checkpoint_blocks
        self.input = nn.Linear(channels, width)
        self.anchor = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, width))
        self.text = nn.Linear(text_dim, width) if text_dim else None
        self.time = nn.Sequential(nn.Linear(256, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList([WindowBlock(width, heads, text_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, channels)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.register_buffer('future_pos', position(future, grid, width, temporal_factor), persistent=False)
        self.register_buffer('anchor_pos', position(1, grid, width, anchor=True), persistent=False)
        if aux_layer and not 1 <= aux_layer < depth:
            raise ValueError('aux_layer must be an intermediate block (1-based)')
        self.aux_layer = aux_layer
        if aux_layer:
            # Preserve baseline initialization and global random stream.
            with torch.random.fork_rng(devices=[]):
                self.aux_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, channels))
                nn.init.zeros_(self.aux_head[-1].weight)
                nn.init.zeros_(self.aux_head[-1].bias)

    def forward(self, noisy, u, anchor, text=None, text_valid=None, return_aux=False):
        if return_aux and not self.aux_layer:
            raise ValueError('auxiliary head is disabled')
        b = noisy.shape[0]
        if noisy.shape != (b, self.future, self.grid**2, self.channels):
            raise ValueError('future shape mismatch')
        if anchor.shape != (b, 1, self.grid**2, self.channels) or u.shape != (b,):
            raise ValueError('anchor/time shape mismatch')
        if self.text_dim and (text is None or text_valid is None):
            raise ValueError('caption model requires embeddings and valid-token mask')
        x = self.input(noisy.flatten(1, 2).float())+self.future_pos
        memory = self.anchor(anchor[:, 0].float())+self.anchor_pos
        context = self.text(text.float()) if self.text_dim else None
        time = self.time(sinusoid(u*1000, 256))[:, None]
        auxiliary = None
        for index, block in enumerate(self.blocks, 1):
            if self.training and self.checkpoint_blocks:
                x = checkpoint(block, x, time, memory, context, text_valid, use_reentrant=False)
            else:
                x = block(x, time, memory, context, text_valid)
            if return_aux and index == self.aux_layer:
                auxiliary = self.aux_head(x).reshape_as(noisy).float()
        output = self.head(self.norm(x)).reshape_as(noisy).float()
        return (output, auxiliary) if return_aux else output
