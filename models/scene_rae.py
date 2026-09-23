"""Per-frame VGGT dual-stream RAE. New identity; never load R7 temporal weights.

The texture channels are part of the generated state, not an RGB decoder skip.
Only the VGGT aggregator is frozen. Fusion, bottleneck and RGB head all train.
"""
from dataclasses import asdict
import torch
from torch import nn
from torch.nn import functional as F
from models.causal_dual_tokenizer import StreamProjection
from utils.r7_representation import R7Config, build_modules, load_r7_modules


class SceneRAE(nn.Module):
    def __init__(self, source_config, channels=256):
        super().__init__()
        self.source_config = dict(source_config)
        self.config = R7Config(**source_config)
        self.channels = channels
        self.grid = self.config.latent_grid
        self.compressor, self.tex_encoder, _, self.decoder = build_modules(
            self.config, include_tokenizer=False)
        self.geo_projection = StreamProjection(self.config.geo_dim, channels//2)
        self.tex_projection = StreamProjection(self.config.tex_dim, channels//2)
        self.restore_features = nn.Linear(channels, len(self.config.levels)*self.config.token_dim)

    @classmethod
    def from_r7(cls, artifact, channels=256):
        cfg, compressor, texture, tokenizer, decoder, _ = load_r7_modules(artifact)
        model = cls(asdict(cfg), channels)
        model.compressor.load_state_dict(compressor.state_dict())
        model.tex_encoder.load_state_dict(texture.state_dict())
        model.decoder.load_state_dict(decoder.state_dict())
        # Preserve the known spatial projection functions, widen with initially
        # zero decoder contribution. No reinterpretation of legacy time-GN.
        for name in ('geo_projection', 'tex_projection'):
            old, new = getattr(tokenizer, name), getattr(model, name)
            n = old.latent_dim
            if n > new.latent_dim:
                raise ValueError('Warm-start widening cannot truncate old channels')
            with torch.no_grad():
                new.norm.load_state_dict(old.norm.state_dict())
                new.compress.weight[:n].copy_(old.compress.weight)
                new.compress.bias[:n].copy_(old.compress.bias)
                new.expand.weight.zero_()
                new.expand.weight[:, :n].copy_(old.expand.weight)
                new.expand.bias.copy_(old.expand.bias)
        return model

    def specification(self):
        return dict(schema='scene-rae-v1', source_config=self.source_config,
                    channels=self.channels, temporal_factor=1, rgb_skip=False)

    def encode(self, tokens, frames):
        geo = self.compressor(tokens).permute(0, 1, 3, 4, 2)
        texture = self.tex_encoder(frames.float())
        return torch.cat((self.geo_projection.encode(geo),
                          self.tex_projection.encode(texture)), -1)

    def decode(self, z):
        geo, texture = z.split(self.channels//2, dim=-1)
        return self.decoder(self.geo_projection.decode(geo), self.tex_projection.decode(texture))

    def feature_targets(self, tokens):
        out = []
        for level in self.config.levels:
            token = tokens[level].float()
            b, t, n, c = token.shape
            token = F.layer_norm(token, (c,))
            token = F.adaptive_avg_pool2d(token.reshape(b*t, self.config.input_grid,
                self.config.input_grid, c).permute(0, 3, 1, 2), self.grid)
            out.append(token.permute(0, 2, 3, 1).reshape(b, t, self.grid, self.grid, c))
        return torch.cat(out, -1).detach()

    def forward(self, tokens, frames, jitter=0., anchor_tokens=None):
        z = self.encode(tokens, frames)
        target = self.feature_targets(tokens)
        if anchor_tokens is not None:
            anchor = self.encode(anchor_tokens, frames[:, :1])
            z = torch.cat((anchor, z[:, 1:]), 1)
            target = torch.cat((self.feature_targets(anchor_tokens), target[:, 1:]), 1)
        rgb_z = z
        if jitter:
            scale = z.detach().float().flatten(0, -2).std(0, unbiased=False).clamp_min(.01)
            # Feature reconstruction and relation losses always see clean z.
            rgb_z = z + torch.randn_like(z)*scale*jitter
        return self.decode(rgb_z), z, self.restore_features(z), target


def relation_loss(z, teacher):
    """Direct latent spatial relations, per frame; no detached student/projection."""
    a = F.normalize(z.float().flatten(2, 3), dim=-1)
    b = F.normalize(teacher.float().flatten(2, 3), dim=-1)
    return F.mse_loss(a @ a.transpose(-1, -2), b @ b.transpose(-1, -2))


def reconstruction_loss(rgb, frames, z, restored, target, perceptual=None,
                        temporal_valid=False, relation_weight=.1):
    truth = frames.permute(0, 1, 3, 4, 2)
    losses = dict(rgb=F.l1_loss(rgb.float(), truth.float()),
        features=F.mse_loss(restored.float(), target), relation=relation_loss(z, target))
    losses['perceptual'] = rgb.new_zeros(())
    if perceptual is not None:
        from utils.scene_losses import lpips_chunked
        losses['perceptual'] = lpips_chunked(perceptual, rgb, truth, chunk_size=1, resize=256)
    losses['temporal'] = rgb.new_zeros(())
    if temporal_valid and rgb.shape[1] > 1:
        losses['temporal'] = F.l1_loss(rgb[:, 1:]-rgb[:, :-1], truth[:, 1:]-truth[:, :-1])
    total = losses['rgb'] + .5*losses['perceptual'] + .1*losses['features'] \
        + relation_weight*losses['relation'] + .1*losses['temporal']
    return total, losses
