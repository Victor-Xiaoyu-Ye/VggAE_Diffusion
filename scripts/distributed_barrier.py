#!/usr/bin/env python3
"""Synchronize ModelArts nodes between chained torchrun stages."""

import torch.distributed as dist

from utils.distributed import setup_ddp


def main():
    use_ddp, _, _, _ = setup_ddp()
    if use_ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
