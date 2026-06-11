"""Unified accelerator management for CUDA and Ascend NPU."""

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None


def get_device_name():
    """Return the available accelerator backend."""
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return 'npu'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def get_device(local_rank=0):
    """Return torch.device for current backend."""
    name = get_device_name()
    if name == 'cpu':
        return torch.device('cpu')
    return torch.device(f'{name}:{local_rank}' if local_rank >= 0 else name)


def resolve_dtype(requested='bf16'):
    """Resolve a requested training dtype for the active backend."""
    if requested == 'fp32' or get_device_name() == 'cpu':
        return torch.float32
    if requested == 'fp16' or get_device_name() == 'npu':
        return torch.float16
    return torch.bfloat16


def autocast_dtype():
    """Return the preferred autocast dtype for the active backend."""
    return resolve_dtype('bf16')


def create_grad_scaler(enabled=True):
    """Create a GradScaler across torch_npu and generic torch AMP versions."""
    if get_device_name() == 'npu' and hasattr(torch.npu, 'amp'):
        scaler_class = getattr(torch.npu.amp, 'GradScaler', None)
        if scaler_class is not None:
            return scaler_class(enabled=enabled)
    try:
        return torch.amp.GradScaler(
            device=get_device_name(), enabled=enabled)
    except TypeError:
        return torch.amp.GradScaler(enabled=enabled)


def is_available():
    """Check if any accelerator is available."""
    return torch.cuda.is_available() or (
        hasattr(torch, 'npu') and torch.npu.is_available()
    )


def empty_cache():
    """Clear device cache."""
    if get_device_name() == 'npu':
        torch.npu.empty_cache()
    elif get_device_name() == 'cuda':
        torch.cuda.empty_cache()


def manual_seed_all(seed):
    """Set seed for all devices."""
    if get_device_name() == 'npu':
        torch.npu.manual_seed_all(seed)
    elif get_device_name() == 'cuda':
        torch.cuda.manual_seed_all(seed)
