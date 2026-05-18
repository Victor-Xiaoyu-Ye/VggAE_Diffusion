"""Unified device management for CUDA / Ascend NPU."""

import torch


def get_device_name():
    """Return 'cuda' or 'npu' depending on what's available."""
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return 'npu'
    return 'cuda'


def get_device(local_rank=0):
    """Return torch.device for current backend."""
    name = get_device_name()
    return torch.device(f'{name}:{local_rank}' if local_rank >= 0 else name)


def autocast_dtype():
    """Return the optimal autocast dtype for current backend.
    CUDA → bfloat16, NPU → float16 (910B doesn't fully support bf16).
    """
    if get_device_name() == 'npu':
        return torch.float16
    return torch.bfloat16


def is_available():
    """Check if any accelerator is available."""
    return torch.cuda.is_available() or (
        hasattr(torch, 'npu') and torch.npu.is_available()
    )


def empty_cache():
    """Clear device cache."""
    if get_device_name() == 'npu':
        if hasattr(torch, 'npu'):
            torch.npu.empty_cache()
    else:
        torch.cuda.empty_cache()


def manual_seed_all(seed):
    """Set seed for all devices."""
    if get_device_name() == 'npu':
        if hasattr(torch, 'npu'):
            torch.npu.manual_seed_all(seed)
    else:
        torch.cuda.manual_seed_all(seed)
