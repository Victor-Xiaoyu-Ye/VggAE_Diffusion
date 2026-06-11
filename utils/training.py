import json
import os
import random

import numpy as np
import torch
import torch.nn as nn


class EMA:
    def __init__(self, model, decay=0.9999, dtype=None):
        self.decay = decay
        self.dtype = dtype
        self.shadow = {}
        for k, v in model.state_dict().items():
            if v.is_floating_point():
                shadow = v.clone().detach()
                if dtype is not None:
                    shadow = shadow.to(dtype=dtype)
                self.shadow[k] = shadow
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k] = self.shadow[k].to(v.device)
                value = v.to(dtype=self.shadow[k].dtype)
                self.shadow[k].mul_(self.decay).add_(value, alpha=1 - self.decay)
    def state_dict(self):
        return self.shadow
    def load_state_dict(self, state_dict):
        self.shadow = state_dict
    def to(self, device):
        self.shadow = {k: v.to(device) for k, v in self.shadow.items()}
        return self


def atomic_torch_save(payload, path):
    """Write a checkpoint atomically so interrupted saves are not resumable."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary_path = f"{path}.tmp-{os.getpid()}"
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def append_metrics(path, metrics):
    """Append durable scalar metrics alongside TensorBoard event files."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    serializable = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().item()
        if isinstance(value, np.generic):
            value = value.item()
        serializable[key] = value
    with open(path, "a") as output:
        output.write(json.dumps(serializable, sort_keys=True) + "\n")


def build_optimizer(model, lr, wd):
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "norm" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return torch.optim.AdamW([
        {"params": decay_params, "weight_decay": wd},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=lr, betas=(0.9, 0.95), eps=1e-8)

def build_scheduler(optimizer, warmup_steps, total_steps, min_lr=1e-6):
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-4, end_factor=1.0, total_iters=warmup_steps)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=min_lr)
    return torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])
