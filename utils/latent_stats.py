"""Moment accumulation and broadcasting for compact video latents."""

import torch
import torch.distributed as dist


def create_moments(num_frames, latent_dim, device):
    return {
        "sum": torch.zeros(
            num_frames, latent_dim, device=device, dtype=torch.float64),
        "sum_sq": torch.zeros(
            num_frames, latent_dim, device=device, dtype=torch.float64),
        "count": torch.zeros(
            num_frames, 1, device=device, dtype=torch.float64),
    }


def update_moments(moments, tensor):
    """Accumulate [B,S,N,D] values while preserving frame and channel axes."""
    values = tensor.float().double()
    if values.ndim != 4:
        raise ValueError(f"Expected [B,S,N,D], got {tuple(values.shape)}")
    if values.shape[1] != moments["sum"].shape[0]:
        raise ValueError(
            f"Frame mismatch: values={values.shape[1]}, "
            f"stats={moments['sum'].shape[0]}")
    if values.shape[-1] != moments["sum"].shape[1]:
        raise ValueError(
            f"Channel mismatch: values={values.shape[-1]}, "
            f"stats={moments['sum'].shape[1]}")
    moments["sum"] += values.sum(dim=(0, 2))
    moments["sum_sq"] += values.square().sum(dim=(0, 2))
    moments["count"] += values.shape[0] * values.shape[2]


def reduce_moments(moments):
    if dist.is_available() and dist.is_initialized():
        for value in moments.values():
            dist.all_reduce(value, op=dist.ReduceOp.SUM)


def finalize_moments(moments, min_std=1e-6):
    count = moments["count"].clamp_min(1)
    mean = moments["sum"] / count
    variance = moments["sum_sq"] / count - mean.square()
    return {
        "mean": mean.float().cpu(),
        "std": variance.clamp_min(min_std ** 2).sqrt().float().cpu(),
        "count": moments["count"].squeeze(-1).long().cpu(),
    }


def validate_latent_stats(
        stats, num_frames, latent_dim, name="latent", allow_legacy=True):
    """Validate a saved normalization contract before broadcasting it."""
    if not isinstance(stats, dict) or "mean" not in stats or "std" not in stats:
        raise ValueError(f"{name} stats must contain mean and std tensors")

    mean = torch.as_tensor(stats["mean"])
    std = torch.as_tensor(stats["std"])
    if mean.shape != std.shape:
        raise ValueError(
            f"{name} mean/std shape mismatch: {tuple(mean.shape)} vs "
            f"{tuple(std.shape)}")
    if mean.ndim == 1:
        if not allow_legacy:
            raise ValueError(
                f"{name} uses legacy [D] stats; frame-aware [S,D] stats "
                "are required")
        frame_count, channel_count = 1, mean.shape[0]
    elif mean.ndim == 2:
        frame_count, channel_count = mean.shape
    else:
        raise ValueError(
            f"{name} stats must have shape [D] or [S,D], got "
            f"{tuple(mean.shape)}")

    if channel_count != latent_dim:
        raise ValueError(
            f"{name} stats channels {channel_count} != {latent_dim}")
    if frame_count not in (1, num_frames):
        raise ValueError(
            f"{name} stats frames {frame_count} are not compatible with "
            f"{num_frames}")
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError(f"{name} stats contain non-finite values")
    if torch.any(std <= 0):
        raise ValueError(f"{name} stats contain non-positive std values")


def broadcast_stat(value, tensor):
    """Convert legacy [D] or frame-aware [S,D] stats to [1,S,1,D]."""
    value = value.to(device=tensor.device, dtype=torch.float32)
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2:
        raise ValueError(
            f"Latent stats must have shape [D] or [S,D], got {tuple(value.shape)}")
    if value.shape[-1] != tensor.shape[-1]:
        raise ValueError(
            f"Stats channels {value.shape[-1]} != tensor channels "
            f"{tensor.shape[-1]}")
    if value.shape[0] not in (1, tensor.shape[1]):
        raise ValueError(
            f"Stats frames {value.shape[0]} are not broadcastable to "
            f"{tensor.shape[1]}")
    return value.view(1, value.shape[0], 1, value.shape[1])


def normalize_latent(tensor, stats):
    mean = broadcast_stat(stats["mean"], tensor)
    std = broadcast_stat(stats["std"], tensor).clamp_min(1e-6)
    return (tensor.float() - mean) / std


def denormalize_latent(tensor, stats):
    mean = broadcast_stat(stats["mean"], tensor)
    std = broadcast_stat(stats["std"], tensor).clamp_min(1e-6)
    return tensor.float() * std + mean
