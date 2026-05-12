import os
import torch
import torch.distributed as dist

def setup_ddp():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        return True, rank, local_rank, world_size
    return False, 0, 0, 1

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0
