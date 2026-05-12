"""Shared decoder loader supporting both DPTHead and ViTDecoder."""

import torch
from streamvggt.heads.dpt_head import DPTHead
from models.vit_decoder import ViTDecoder


def load_decoder(path, device, decoder_type="auto", **overrides):
    """Load DPTHead or ViTDecoder checkpoint.

    Args:
        path: checkpoint path
        device: torch device
        decoder_type: "dpt", "vit", or "auto" (detect from checkpoint keys)
        **overrides: passed to decoder constructor

    Returns:
        decoder module in eval mode with frozen params
    """
    state = torch.load(path, map_location="cpu")

    # Extract state_dict
    if isinstance(state, dict):
        for key in ("model_state_dict", "ema_state_dict", "model", "ema"):
            if key in state:
                state = state[key]
                break

    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    # Auto-detect
    if decoder_type == "auto":
        has_dpt_keys = any("scratch.refinenet" in k for k in state)
        has_vit_keys = any("blocks.0" in k for k in state)
        if has_vit_keys and not has_dpt_keys:
            decoder_type = "vit"
        else:
            decoder_type = "dpt"

    # Build decoder
    if decoder_type == "vit":
        decoder = ViTDecoder(
            dim=overrides.get("token_dim", 2048),
            decoder_dim=overrides.get("decoder_dim", 512),
            num_levels=overrides.get("num_levels", 4),
            depth=overrides.get("depth", overrides.get("vit_depth", 4)),
            num_heads=overrides.get("num_heads", overrides.get("vit_heads", 8)),
            patch_size=overrides.get("patch_size", 14),
            img_size=overrides.get("img_size", 518),
            output_dim=overrides.get("output_dim", 3),
        )
    else:
        has_depth_keys = any("depth_output_conv2" in k for k in state)
        decoder = DPTHead(
            dim_in=overrides.get("dim_in", overrides.get("token_dim", 2048)),
            patch_size=overrides.get("patch_size", 14),
            output_dim=overrides.get("output_dim", 4),
            activation=overrides.get("activation", "sigmoid"),
            conf_activation=overrides.get("conf_activation", "sigmoid"),
            output_depth=has_depth_keys or overrides.get("output_depth", False),
        )

    decoder = decoder.to(device=device, dtype=torch.float32)
    decoder.load_state_dict(state, strict=True)
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad_(False)

    return decoder
