import torch

DPT_LEVELS = [4, 11, 17, 23]
DEFAULT_BOUNDARY_LEVEL = 11


def strip_special_tokens(tokens_list, patch_start_idx):
    """Strip camera/register tokens from all levels. DPTHead ignores them anyway.

    Returns a new list of 24 tensors, each [B, S, N, D] (patch tokens only).
    """
    return [t[:, :, patch_start_idx:] for t in tokens_list]


def load_token_stats(path, device, dtype=torch.float32):
    stats = torch.load(path, map_location="cpu")
    level_stats = {}
    for lvl in DPT_LEVELS:
        mean = stats[f"mean_{lvl}"].to(device=device, dtype=dtype)
        var = stats[f"var_{lvl}"].to(device=device, dtype=dtype)
        level_stats[lvl] = (mean, var)
    return level_stats


def normalize_tokens(tokens_list, level_stats, eps=1e-5):
    """Normalize per-level patch tokens (already stripped of special tokens).

    tokens_list: list of 24 tensors [B, S, N, D]
    level_stats: dict {level: (mean, var)}
    Returns: new list of 24 tensors (does not mutate input).
    """
    out = list(tokens_list)
    for lvl, (mean, var) in level_stats.items():
        tokens = out[lvl]
        shape = [1] * (tokens.dim() - mean.dim())
        m = mean.view(*([1] * len(shape)), *mean.shape)
        v = var.view(*([1] * len(shape)), *var.shape)
        out[lvl] = (tokens - m) / torch.sqrt(v + eps)
    return out


def select_levels(tokens_list, levels):
    """Select DPT levels from (already stripped) aggregated tokens.

    Returns: [B, L*S, N, D] where L=len(levels).
    For a single level returns [B, S, N, D].
    """
    selected = [tokens_list[lvl] for lvl in levels]
    if len(selected) == 1:
        return selected[0]
    B, S, N, D = selected[0].shape
    return torch.stack(selected, dim=1).reshape(B, len(levels) * S, N, D)


def augment_tokens_for_decoder(
    tokens_list,
    levels=DPT_LEVELS,
    boundary_level=DEFAULT_BOUNDARY_LEVEL,
    level_dropout=0.0,
    boundary_only_prob=0.0,
    token_noise_std=0.0,
):
    """Apply decoder-side robustness augmentations to (already stripped) tokens.

    The RGB decoder must tolerate generated latents, not only exact encoder
    latents. With `boundary_only_prob`, the decoder sometimes sees only the
    boundary level used by diffusion.
    """
    if not levels:
        return list(tokens_list)

    out = list(tokens_list)
    device = out[levels[0]].device
    force_boundary = boundary_level in levels and torch.rand((), device=device).item() < boundary_only_prob

    keep_levels = set()
    if force_boundary:
        keep_levels.add(boundary_level)
    else:
        for lvl in levels:
            if level_dropout <= 0 or torch.rand((), device=device).item() >= level_dropout:
                keep_levels.add(lvl)
        if not keep_levels:
            keep_levels.add(boundary_level if boundary_level in levels else levels[len(levels) // 2])

    for lvl in levels:
        tokens = out[lvl]
        if token_noise_std > 0:
            tokens = tokens + torch.randn_like(tokens) * token_noise_std
        if lvl not in keep_levels:
            tokens = torch.zeros_like(tokens)
        out[lvl] = tokens

    return out


def build_decoder_tokens_from_generated(
    z,
    levels,
    seq_len,
    total_levels=24,
    dtype=torch.float32,
):
    """Convert generated patch tokens to DPTHead's 24-entry token list.

    All tokens are patch tokens (special tokens already stripped).
    Non-generated levels are filled with zeros.

    Args:
        z: [B, S, N, D] for one level, or [B, L*S, N, D] for legacy
           level-major multi-level generation.
        levels: generated DPT level indices.
        seq_len: number of video frames.
        total_levels: length of DPTHead token list.
        dtype: output dtype for decoder.

    Returns:
        List of `total_levels` tensors, each [B, S, N, D].
    """
    if not levels:
        raise ValueError("levels must contain at least one DPT level")

    B, T, N, D = z.shape
    num_levels = len(levels)
    if num_levels == 1:
        if T != seq_len:
            raise ValueError(f"single-level z has T={T}, expected seq_len={seq_len}")
        z_per_level = z[:, None]
    else:
        expected_t = num_levels * seq_len
        if T != expected_t:
            raise ValueError(f"multi-level z has T={T}, expected {expected_t}")
        z_per_level = z.reshape(B, num_levels, seq_len, N, D)

    z_per_level = z_per_level.to(dtype=dtype)
    zero_level = torch.zeros(B, seq_len, N, D, device=z.device, dtype=dtype)
    out = [zero_level.clone() for _ in range(total_levels)]
    for pos, lvl in enumerate(levels):
        out[lvl] = z_per_level[:, pos]
    return out
