import os
import torch
import torch.distributed as dist


def _get_backend():
    """Return nccl (CUDA) or hccl (Ascend NPU)."""
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return 'hccl'
    return 'nccl'


def setup_ddp():
    if "RANK" in os.environ:
        backend = _get_backend()
        dist.init_process_group(backend)
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
        if backend == 'hccl':
            torch.npu.set_device(local_rank)
        else:
            torch.cuda.set_device(local_rank)
        return True, rank, local_rank, world_size
    return False, 0, 0, 1


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0
