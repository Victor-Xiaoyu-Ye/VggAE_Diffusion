"""Short-window overlap helpers for R7 I2V training and sampling."""

from __future__ import annotations

import torch


def overlapping_windows(future_chunks, window=2, overlap=1):
    """Return deterministic [start,end) windows covering all future chunks."""
    if future_chunks < 1 or window < 1 or not 0 <= overlap < window:
        raise ValueError("invalid future/window/overlap")
    stride = window - overlap
    result = []
    start = 0
    while start < future_chunks:
        end = min(start + window, future_chunks)
        start = max(0, end - window)
        pair = (start, end)
        if not result or result[-1] != pair:
            result.append(pair)
        if end == future_chunks:
            break
        start += stride
    return result


def generated_context_mix(clean, generated, probability, generator=None):
    """Per-sample scheduled-context replacement with a detached prediction."""
    if clean.shape != generated.shape:
        raise ValueError("clean and generated context shapes differ")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0,1]")
    if probability == 0:
        return clean
    if probability == 1:
        return generated.detach()
    mask = torch.rand(
        clean.shape[0], device=clean.device, generator=generator) < probability
    shape = [clean.shape[0]] + [1] * (clean.ndim - 1)
    return torch.where(mask.reshape(shape), generated.detach(), clean)


def scheduled_context_probability(step, start, ramp, maximum):
    if maximum < 0 or maximum > 1 or start < 0 or ramp < 0:
        raise ValueError("invalid scheduled-context configuration")
    if step < start:
        return 0.0
    if ramp == 0:
        return maximum
    return maximum * min(1.0, (step - start + 1) / ramp)
