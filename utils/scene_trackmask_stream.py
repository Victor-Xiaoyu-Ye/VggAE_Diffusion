"""Diagnostic-only full StreamVGGT heads with the production causal backbone.

Head constructors match wzzheng/StreamVGGT at the commit in
third_party/streamvggt_trackmask/UPSTREAM.json. The production feature-only
StreamVGGT wrapper and its training callers remain unchanged. No hub mixin,
weight download, point head, or one-frame inference API is needed here.
"""
from functools import lru_cache

import torch
from torch import nn

from streamvggt.models.aggregator import Aggregator
from utils.scene_trackmask_vendor import import_stream_vendor

CameraHead = import_stream_vendor('heads.camera_head').CameraHead
DPTHead = import_stream_vendor('heads.dpt_head').DPTHead
TrackHead = import_stream_vendor('heads.track_head').TrackHead
_head_utils = import_stream_vendor('heads.utils')
create_uv_grid = _head_utils.create_uv_grid
position_grid_to_embed = _head_utils.position_grid_to_embed


@lru_cache(maxsize=8)
def _cpu_position_embedding(patch_w, patch_h, image_w, image_h, channels, dtype, ratio):
    # Upstream sinusoidal frequencies use FP64, unsupported by Ascend. These
    # geometry-independent constants are computed on CPU with the same formula,
    # then cast by upstream to FP32 before copying to the inference device.
    grid = create_uv_grid(patch_w, patch_h, aspect_ratio=image_w / image_h,
                          dtype=dtype, device='cpu')
    embedding = position_grid_to_embed(grid, channels) * ratio
    return embedding.permute(2, 0, 1)[None]


class DeviceCompatibleDPTHead(DPTHead):
    def _apply_pos_embed(self, x, W, H, ratio=0.1):
        position = _cpu_position_embedding(x.shape[-1], x.shape[-2], W, H,
                                            x.shape[1], x.dtype, ratio)
        return x + position.to(device=x.device).expand(x.shape[0], -1, -1, -1)


class StreamVGGTTrackModel(nn.Module):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024, **unused):
        super().__init__()
        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size,
                                     embed_dim=embed_dim)
        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.depth_head = DeviceCompatibleDPTHead(dim_in=2 * embed_dim, output_dim=2,
                                                  activation='exp', conf_activation='expp1')
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size)
