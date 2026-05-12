import torch
import torch.nn as nn
from typing import Optional, List
from dataclasses import dataclass

from streamvggt.models.aggregator import Aggregator


@dataclass
class StreamVGGTOutput:
    ress: Optional[List[dict]] = None
    views: Optional[List] = None


class StreamVGGT(nn.Module):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024):
        super().__init__()
        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)

    def forward(self, images):
        """Extract aggregated tokens from images.

        Args:
            images: [B, S, 3, H, W] tensor in [0, 1]

        Returns:
            aggregated_tokens_list: list of 24 tensors, each [B, S, P, 2*embed_dim]
            patch_start_idx: int (number of special tokens prepended)
        """
        return self.aggregator(images)
