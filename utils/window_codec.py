"""Explicit execution semantics for historical frozen R7 checkpoints.

GroupNorm weight shapes do not identify whether time participates in moments.
Every window cache must carry this separate runtime contract.
"""
from models.causal_temporal_codec import FramewiseGroupNorm
import math


def runtime_contract(mode):
    if mode not in ('legacy', 'framewise'):
        raise ValueError('select legacy or framewise temporal normalization explicitly')
    return dict(schema='r7-window-codec-runtime-v1', temporal_norm=mode,
                condition='independent_first_frame', future='joint_full_window')


def configure_codec(tokenizer, mode):
    contract = runtime_contract(mode)
    count = 0
    for module in tokenizer.modules():
        if isinstance(module, FramewiseGroupNorm):
            module.temporal_norm = mode
            count += 1
    if not count: raise ValueError('R7 tokenizer has no recognized temporal normalization layers')
    return contract


def validate_runtime(representation):
    value = representation.get('window_codec_runtime')
    if not isinstance(value, dict) or value != runtime_contract(value.get('temporal_norm')):
        raise ValueError('missing/incompatible codec runtime; rebuild a versioned window cache')
    return value['temporal_norm']


def reconstruction_gate(values, minimum):
    if not math.isfinite(minimum) or minimum<=0: raise ValueError('invalid AE PSNR minimum')
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError('missing or nonfinite AE reconstruction measurements')
    mean=sum(values)/len(values)
    return mean,mean>=minimum
