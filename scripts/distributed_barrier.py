#!/usr/bin/env python3
"""Synchronize ModelArts nodes between chained torchrun stages."""

from __future__ import annotations

import os
import sys

# Executing ``PROJECT/scripts/foo.py`` puts only ``PROJECT/scripts`` on
# sys.path. Add the repository root so project packages resolve from any cwd.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch.distributed as dist

from utils.device import get_device_name
from utils.distributed import setup_ddp


def main():
    get_device_name()  # Import/register torch_npu before selecting HCCL.
    use_ddp, _, _, _ = setup_ddp()
    if use_ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
