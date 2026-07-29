"""R7 causal spatiotemporal tokenizer over dual geometry/texture latents."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.causal_temporal_codec import CausalSpatiotemporalCodec


class StreamProjection(nn.Module):
    def __init__(self, input_dim=256, latent_dim=96):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.norm = nn.LayerNorm(input_dim)
        self.compress = nn.Linear(input_dim, latent_dim)
        self.expand = nn.Linear(latent_dim, input_dim)

    def encode(self, x):
        return self.compress(self.norm(x))

    def decode(self, z):
        return self.expand(z)


class CausalDualTokenizerCore(nn.Module):
    """Compress/decompress `[z_geo | z_tex]`; spatial RGB decode stays outside."""

    def __init__(self, geo_dim=256, tex_dim=256, geo_latent_dim=96,
                 tex_latent_dim=96, temporal_factor=2, temporal_depth=3):
        super().__init__()
        self.geo_dim = geo_dim
        self.tex_dim = tex_dim
        self.geo_latent_dim = geo_latent_dim
        self.tex_latent_dim = tex_latent_dim
        self.latent_dim = geo_latent_dim + tex_latent_dim
        self.temporal_factor = temporal_factor
        self.temporal = CausalSpatiotemporalCodec(
            self.latent_dim, temporal_factor, temporal_depth)
        self.geo_projection = StreamProjection(geo_dim, geo_latent_dim)
        self.tex_projection = StreamProjection(tex_dim, tex_latent_dim)

    @staticmethod
    def _to_bcthw(x):
        return x.permute(0, 4, 1, 2, 3).contiguous()

    @staticmethod
    def _to_bthwc(x):
        return x.permute(0, 2, 3, 4, 1).contiguous()

    def encode(self, z_geo, z_tex):
        compressed = torch.cat([
            self.geo_projection.encode(z_geo),
            self.tex_projection.encode(z_tex),
        ], dim=-1)
        temporal = self.temporal.encode(self._to_bcthw(compressed))
        return self._to_bthwc(temporal)

    def decode(self, z):
        full = self.temporal.decode(self._to_bcthw(z))
        full = self._to_bthwc(full)
        geo, tex = full.split(
            [self.geo_latent_dim, self.tex_latent_dim], dim=-1)
        return self.geo_projection.decode(geo), self.tex_projection.decode(tex)

    def forward(self, z_geo, z_tex):
        z = self.encode(z_geo, z_tex)
        geo_rec, tex_rec = self.decode(z)
        return geo_rec, tex_rec, z


class CausalDualTokenizerAE(nn.Module):
    def __init__(self, tokenizer, decoder):
        super().__init__()
        self.tokenizer = tokenizer
        self.decoder = decoder

    def forward(self, z_geo, z_tex, frames_chunk_size=None):
        geo_rec, tex_rec, z = self.tokenizer(z_geo, z_tex)
        rgb = self.decoder(
            geo_rec, tex_rec, frames_chunk_size=frames_chunk_size)
        return rgb, geo_rec, tex_rec, z
