"""DINO-based discriminator for GAN training (inspired by GLD)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DinoDiscriminator(nn.Module):
    """Frozen DINO ViT-S/8 backbone + trainable conv1d heads for real/fake."""

    def __init__(self, model_name="facebook/dinov2-small", device="cuda"):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(model_name).to(device).eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        # DINOv2-small: embed_dim=384, 12 blocks
        # Extract features at blocks [2, 5, 8, 11]
        self.out_indices = [2, 5, 8, 11]
        embed_dim = 384
        head_dim = 64

        # Small conv1d heads per feature level
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.utils.spectral_norm(nn.Conv1d(embed_dim, head_dim, 1)),
                nn.LeakyReLU(0.2),
                nn.utils.spectral_norm(nn.Conv1d(head_dim, 1, 1)),
            ) for _ in range(len(self.out_indices) + 1)  # +1 for final layer
        ])

    def forward(self, x):
        """x: [B, 3, H, W] in [-1, 1]"""
        with torch.no_grad():
            outputs = self.backbone(x, output_hidden_states=True)
            feats = [outputs.hidden_states[i] for i in self.out_indices]  # [B, N, 384]
            feats.append(outputs.last_hidden_state)  # add final layer

        logits = []
        for feat, head in zip(feats, self.heads):
            feat = feat.transpose(1, 2)  # [B, 384, N]
            logit = head(feat)            # [B, 1, N]
            logits.append(logit.mean(dim=-1))  # [B, 1]

        return torch.cat(logits, dim=1).mean(dim=1, keepdim=True)  # [B, 1]


def hinge_d_loss(logits_real, logits_fake):
    loss_real = F.relu(1.0 - logits_real).mean()
    loss_fake = F.relu(1.0 + logits_fake).mean()
    return 0.5 * (loss_real + loss_fake)


def vanilla_g_loss(logits_fake):
    return -logits_fake.mean()


def diff_augment(x):
    """Lightweight differentiable augmentation: color jitter + small translation."""
    B, C, H, W = x.shape
    # Color jitter
    brightness = 1.0 + 0.1 * (torch.rand(B, 1, 1, 1, device=x.device) * 2 - 1)
    x = x * brightness
    contrast = 0.9 + 0.2 * torch.rand(B, 1, 1, 1, device=x.device)
    x = (x - 0.5) * contrast + 0.5
    # Small random crop
    if H > 224:
        top = torch.randint(0, H - 224, (1,), device=x.device).item()
        left = torch.randint(0, W - 224, (1,), device=x.device).item()
        x = x[:, :, top:top+224, left:left+224]
    return x.clamp(-1, 1)
