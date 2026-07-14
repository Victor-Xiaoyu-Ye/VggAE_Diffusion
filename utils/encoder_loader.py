"""Verified checkpoint loading for the frozen StreamVGGT encoder.

Why this exists
---------------
Several entry points used to do::

    state = torch.load(ckpt, map_location='cpu')
    encoder.load_state_dict(state, strict=False)

If the checkpoint is wrapped (``{'model_state_dict': ...}``) or carries a
``module.`` prefix, ``strict=False`` loads ZERO weights and raises nothing —
the probe then trains against a randomly initialized encoder and the metrics
look plausible but are meaningless. Every encoder load must go through
:func:`load_encoder_checkpoint`, which unwraps common containers and asserts
that a minimum fraction of parameters actually matched.
"""

from __future__ import annotations

import torch

_WRAPPER_KEYS = ("model_state_dict", "model", "state_dict", "ema_state_dict")


def unwrap_state_dict(state):
    """Unwrap common checkpoint containers and strip ``module.`` prefixes."""
    if isinstance(state, dict):
        for key in _WRAPPER_KEYS:
            inner = state.get(key)
            if isinstance(inner, dict) and inner and all(
                    torch.is_tensor(v) for v in list(inner.values())[:4]):
                state = inner
                break
    if isinstance(state, dict) and any(k.startswith("module.") for k in state):
        state = {
            (k[len("module."):] if k.startswith("module.") else k): v
            for k, v in state.items()
        }
    return state


def load_encoder_checkpoint(encoder, ckpt_path, min_match=0.9, verbose=True):
    """Load a StreamVGGT checkpoint into ``encoder`` and verify coverage.

    Raises RuntimeError if fewer than ``min_match`` of the encoder's
    parameters were matched by the checkpoint (the silent-failure mode of
    ``strict=False`` on a wrapped/foreign state_dict).
    """
    # weights_only=False: training-container checkpoints commonly carry
    # pickled metadata (args namespace, epoch info) that the torch>=2.6
    # weights_only default refuses to load.
    state = unwrap_state_dict(
        torch.load(ckpt_path, map_location="cpu", weights_only=False))
    result = encoder.load_state_dict(state, strict=False)
    total = len(encoder.state_dict())
    matched = total - len(result.missing_keys)
    ratio = matched / max(total, 1)
    if verbose:
        print(f"[encoder] {ckpt_path}: matched {matched}/{total} keys "
              f"({ratio:.1%}), missing={len(result.missing_keys)}, "
              f"unexpected={len(result.unexpected_keys)}")
    if ratio < min_match:
        raise RuntimeError(
            f"Encoder checkpoint {ckpt_path} matched only {matched}/{total} "
            f"({ratio:.1%}) StreamVGGT keys — wrapped differently or not a "
            f"StreamVGGT state_dict. missing[:5]={result.missing_keys[:5]} "
            f"unexpected[:5]={result.unexpected_keys[:5]}")
    return result
