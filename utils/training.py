import torch
import torch.nn as nn

class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {}
        for k, v in model.state_dict().items():
            if v.is_floating_point():
                self.shadow[k] = v.clone().detach()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)
    def state_dict(self):
        return self.shadow
    def load_state_dict(self, state_dict):
        self.shadow = state_dict
    def to(self, device):
        self.shadow = {k: v.to(device) for k, v in self.shadow.items()}
        return self

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
