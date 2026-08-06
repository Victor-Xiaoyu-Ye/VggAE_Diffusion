#!/usr/bin/env python3
"""Production fixed-window R7 diffusion training from cached tar shards.

The cache contract is deliberately narrow: every member contains ``cond``
``[1,N,192]`` and absolute ``target`` ``[4,N,192]``.  Normalization v2 is
frame/position-aware (``[1,D]`` and ``[4,D]``) and all normalization arithmetic
is performed in fp32.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import pickle
import random
from typing import Any, Mapping

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.latent_shard_dataset import LatentShardDataset, latent_collate_fn
from models.compact_dit import CompactLatentDiT
from utils.device import (configure_backend_compatibility, create_grad_scaler,
                          get_device, get_device_name, manual_seed_all,
                          resolve_dtype)
from utils.distributed import is_main_process, setup_ddp
from utils.training import (EMA, ThroughputMeter, append_metrics,
                            atomic_torch_save, build_optimizer,
                            build_scheduler, capture_rng_state,
                            count_latent_tokens, restore_rng_state)

CONTEXT_CHUNKS = 1
FUTURE_CHUNKS = 4
LATENT_DIM = 192
VMSE_BUCKETS = (0.1, 0.3, 0.5, 0.7, 0.9)
STRICT_MODEL_ARGS = ("latent_dim", "latent_grid", "model_dim", "spatial_depth",
                     "temporal_depth", "num_heads", "time_scale",
                     "time_shift_alpha", "normalization_mode")
STRICT_TRAIN_ARGS = ("lambda_motion", "lambda_accel", "lambda_geo_motion",
                     "aux_warmup_steps", "aux_ramp_steps", "aux_t_min",
                     "batch_size", "accum_steps", "lr", "wd", "warmup_steps",
                     "ema_decay", "max_grad_norm", "dtype",
                     "require_rgb_lpips")
ARCHITECTURE_CONTRACT = {
    "class": "CompactLatentDiT",
    "clean_frame0": True,
    "i0_condition": False,
    "block_schedule": "interleaved",
    "seq_len": FUTURE_CHUNKS,
    "text_cond": False,
}
OBJECTIVE_SCHEMA = "r7-fixed-window-shifted-ot-cfm-v1"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Production fixed-window R7 diffusion")
    p.add_argument("--manifest", required=True)
    p.add_argument("--stats", required=True)
    p.add_argument("--eval_manifest", required=True)
    p.add_argument("--eval_stats", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--resume", default="")
    p.add_argument("--r7_ckpt", default="",
                   help="strict R7 tokenizer/decoder load for geo aux and RGB eval")
    p.add_argument("--latent_dim", type=int, default=LATENT_DIM)
    p.add_argument("--latent_grid", type=int, default=18)
    p.add_argument("--context_chunks", type=int, default=CONTEXT_CHUNKS)
    p.add_argument("--future_chunks", type=int, default=FUTURE_CHUNKS)
    p.add_argument("--model_dim", type=int, default=1152)
    p.add_argument("--spatial_depth", type=int, default=10)
    p.add_argument("--temporal_depth", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=16)
    p.add_argument("--time_scale", type=float, default=1.0)
    p.add_argument("--time_shift_alpha", type=float, default=1.0)
    p.add_argument("--normalization_mode", choices=("zscore", "none"),
                   default="zscore")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--accum_steps", type=int, default=4)
    p.add_argument("--max_steps", type=int, default=6000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--warmup_steps", type=int, default=300)
    p.add_argument("--extension_lr", type=float, default=0.0,
                   help="required peak LR when extending a completed run")
    p.add_argument("--extension_warmup_steps", type=int, default=200)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="bf16")
    p.add_argument("--lambda_motion", type=float, default=0.10)
    p.add_argument("--lambda_accel", type=float, default=0.05)
    p.add_argument("--lambda_geo_motion", type=float, default=0.05)
    p.add_argument("--aux_warmup_steps", type=int, default=1000)
    p.add_argument("--aux_ramp_steps", type=int, default=1000)
    p.add_argument("--aux_t_min", type=float, default=0.6)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--shuffle_buffer", type=int, default=256)
    p.add_argument("--eval_clips", type=int, default=16)
    p.add_argument("--sample_clips", type=int, default=4)
    p.add_argument("--sample_steps", type=int, default=30)
    p.add_argument("--decode_chunk_size", type=int, default=0,
                   help="0 decodes all 9 frames together; chunking is forbidden "
                        "for decoders with temporal attention")
    p.add_argument("--require_rgb_lpips", action="store_true",
                   help="fail closed if decoded RGB LPIPS cannot be evaluated")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--throughput_divisor", type=float, default=1.0,
                   help="Divide reported DI_throughput by this value")
    p.add_argument("--early_stop_min_steps", type=int, default=6000)
    p.add_argument("--patience", type=int, default=8,
                   help="number of non-improving evaluations after min steps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--local_rank", type=int, default=0)
    return p.parse_args(argv)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    manual_seed_all(seed)


def load_torch_artifact(path):
    """Load a local or OBS torch artifact without mutating the cache."""
    if isinstance(path, str) and path.startswith(("obs://", "s3://")):
        try:
            from utils.moxing_io import read_bytes
        except ImportError as exc:
            raise RuntimeError("OBS stats require utils.moxing_io.read_bytes") from exc
        return torch.load(io.BytesIO(read_bytes(path)), map_location="cpu",
                          weights_only=False)
    return torch.load(path, map_location="cpu", weights_only=False)


def _jsonable(value):
    if torch.is_tensor(value):
        return {"dtype": str(value.dtype), "shape": list(value.shape),
                "sha256": hashlib.sha256(value.detach().cpu().contiguous()
                                           .numpy().tobytes()).hexdigest()}
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def normalization_signature(stats):
    payload = {"normalization_version": stats.get("normalization_version"),
               "cond": stats.get("cond"), "target": stats.get("target"),
               "representation": stats.get("representation")}
    raw = json.dumps(_jsonable(payload), sort_keys=True,
                     separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def exact_equal(a: Any, b: Any) -> bool:
    if torch.is_tensor(a) or torch.is_tensor(b):
        return torch.is_tensor(a) and torch.is_tensor(b) and \
            a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, Mapping) or isinstance(b, Mapping):
        return isinstance(a, Mapping) and isinstance(b, Mapping) and \
            set(a) == set(b) and all(exact_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and \
            all(exact_equal(x, y) for x, y in zip(a, b))
    return a == b


def validate_stats(stats, label):
    if not isinstance(stats, Mapping):
        raise ValueError(f"{label}: stats.pt must contain a mapping")
    if stats.get("normalization_version") != 2:
        raise ValueError(f"{label}: normalization_version must be exactly 2")
    if not isinstance(stats.get("representation"), dict):
        raise ValueError(f"{label}: stats.pt requires representation dict")
    for name, frames in (("cond", CONTEXT_CHUNKS), ("target", FUTURE_CHUNKS)):
        group = stats.get(name)
        if not isinstance(group, dict):
            raise ValueError(f"{label}: missing {name} normalization")
        mean, std = torch.as_tensor(group.get("mean")), torch.as_tensor(group.get("std"))
        if mean.shape != (frames, LATENT_DIM) or std.shape != mean.shape:
            raise ValueError(f"{label}: {name} mean/std must be [{frames},{LATENT_DIM}], "
                             f"got {tuple(mean.shape)}/{tuple(std.shape)}")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or \
                (std <= 0).any():
            raise ValueError(f"{label}: invalid {name} mean/std")
    rep = stats["representation"]
    from utils.r7_representation import R7_CONTRACT_SCHEMA
    if rep.get("schema") != R7_CONTRACT_SCHEMA:
        raise ValueError(f"{label}: unsupported R7 representation schema")
    config = rep.get("config") or {}
    layout = rep.get("layout") or {}
    if (int(config.get("temporal_factor", -1)),
            int(config.get("geo_latent_dim", -1))
            + int(config.get("tex_latent_dim", -1)),
            int(config.get("seq_len", -1))) != (2, 192, 9):
        raise ValueError(f"{label}: expected t2/c192/seq9 R7 config")
    if (int(config.get("latent_grid", -1)) != 18
            or layout.get("anchor_chunks") != [1, 4]
            or layout.get("channel_split") != [96, 96]):
        raise ValueError(f"{label}: invalid R7 grid/chunk/channel layout")
    signatures = rep.get("signatures") or {}
    missing = [name for name in ("streamvggt", "source_dual_ae", "r7")
               if name not in signatures]
    if missing:
        raise ValueError(f"{label}: missing representation signatures {missing}")
    return normalization_signature(stats)


def assert_same_representation(train_stats, eval_stats):
    if not exact_equal(train_stats.get("representation"),
                       eval_stats.get("representation")):
        raise ValueError("evaluation representation differs exactly from training")


def stats_tensors(stats, device):
    return {name: {key: torch.as_tensor(stats[name][key], device=device,
                                        dtype=torch.float32)
                   for key in ("mean", "std")}
            for name in ("cond", "target")}


def normalize_fp32(x, group, mode):
    x = x.float()
    if mode == "none": return x
    return (x - group["mean"][None, :, None]) / group["std"][None, :, None]


def inverse_fp32(x, group, mode):
    x = x.float()
    if mode == "none": return x
    return x * group["std"][None, :, None] + group["mean"][None, :, None]


def shift_time(t, alpha):
    return t if alpha == 1.0 else alpha * t / (1.0 + (alpha - 1.0) * t)


def architecture_contract(args):
    contract = dict(ARCHITECTURE_CONTRACT)
    contract.update({
        "latent_dim": int(args.latent_dim),
        "latent_grid": int(args.latent_grid),
        "model_dim": int(args.model_dim),
        "spatial_depth": int(args.spatial_depth),
        "temporal_depth": int(args.temporal_depth),
        "num_heads": int(args.num_heads),
        "time_scale": float(args.time_scale),
    })
    return contract


def objective_contract(args):
    return {
        "schema": OBJECTIVE_SCHEMA,
        "prediction": "velocity",
        "path": "x_t=(1-t)*noise+t*absolute_target",
        "time_shift_alpha": float(args.time_shift_alpha),
        "normalization_mode": args.normalization_mode,
        "lambda_motion": float(args.lambda_motion),
        "lambda_accel": float(args.lambda_accel),
        "lambda_geo_motion": float(args.lambda_geo_motion),
        "aux_warmup_steps": int(args.aux_warmup_steps),
        "aux_ramp_steps": int(args.aux_ramp_steps),
        "aux_t_min": float(args.aux_t_min),
        "batch_size": int(args.batch_size),
        "accum_steps": int(args.accum_steps),
        "optimizer": {
            "name": "AdamW", "lr": float(args.lr), "weight_decay": float(args.wd),
            "betas": [0.9, 0.95], "eps": 1e-8,
        },
    }


def build_model(args):
    return CompactLatentDiT(
        latent_dim=args.latent_dim, num_tokens=args.latent_grid ** 2,
        model_dim=args.model_dim, spatial_depth=args.spatial_depth,
        temporal_depth=args.temporal_depth, num_heads=args.num_heads,
        seq_len=FUTURE_CHUNKS, text_cond=False, i0_condition=False,
        clean_frame0=True, block_schedule="interleaved",
        time_scale=args.time_scale)


def validate_batch(batch, args):
    cond, target = batch["cond"], batch["target"]
    expected_tail = (args.latent_grid ** 2, LATENT_DIM)
    if tuple(cond.shape[1:]) != (CONTEXT_CHUNKS, *expected_tail):
        raise ValueError(f"cache cond must be [B,1,N,192], got {tuple(cond.shape)}")
    if tuple(target.shape[1:]) != (FUTURE_CHUNKS, *expected_tail):
        raise ValueError(f"cache target must be [B,4,N,192], got {tuple(target.shape)}")
    return cond, target


def flow_forward(model, x1, cond):
    x0 = torch.randn_like(x1)
    t = shift_time(torch.rand(x1.shape[0], device=x1.device),
                   flow_forward.time_shift_alpha).to(x1.dtype)
    te = t.view(-1, 1, 1, 1)
    xt = (1 - te) * x0 + te * x1
    target_v = x1 - x0
    pred_v = model(xt, t, cond=cond)
    return F.mse_loss(pred_v.float(), target_v.float()), {
        "t": t, "x1_pred": xt + (1 - te) * pred_v, "v_pred": pred_v,
        "v_target": target_v}
flow_forward.time_shift_alpha = 1.0


def masked_loss(error, t, threshold):
    per = error.float().reshape(error.shape[0], -1).mean(1)
    mask = (t.float() >= threshold).float()
    return (per * mask).sum() / mask.sum().clamp_min(1)


def aux_scale(step, args):
    if step < args.aux_warmup_steps: return 0.0
    if args.aux_ramp_steps <= 0: return 1.0
    return min(1.0, (step - args.aux_warmup_steps + 1) / args.aux_ramp_steps)


def decoder_temporal_blocks(config, decoder):
    value = getattr(config, "decoder_temporal_blocks", None)
    if value is None and hasattr(config, "__dict__"):
        value = config.__dict__.get("decoder_temporal_blocks")
    if value is not None: return int(value)
    blocks = getattr(decoder, "temporal_blocks", None)
    return len(blocks) if blocks is not None else 0


def checked_decode_chunk_size(config, decoder, requested):
    temporal = decoder_temporal_blocks(config, decoder)
    if requested < 0: raise ValueError("decode_chunk_size must be nonnegative")
    if temporal > 0 and 0 < requested < 9:
        raise ValueError("R7 decoder has temporal attention; decode_chunk_size < 9 "
                         "would break full-sequence temporal attention")
    return None if requested == 0 else requested


def validate_r7_artifact_representation(stats_representation, r7_path):
    from utils.file_signature import sampled_file_signature
    from utils.r7_representation import validate_contract

    artifact = load_torch_artifact(r7_path)
    checkpoint_representation = artifact.get("representation_contract")
    if not isinstance(checkpoint_representation, Mapping):
        raise ValueError("--r7_ckpt must contain representation_contract")
    # The cache adds the accepted R7 artifact's self-signature.  The checkpoint
    # contract predates that self-reference and normally signs only its encoder
    # and source dual AE, so compare via the representation contract validator.
    validate_contract(stats_representation, checkpoint_representation)
    expected_r7 = stats_representation.get("signatures", {}).get("r7")
    if expected_r7 is None:
        raise ValueError("cache representation must contain signatures.r7")
    if sampled_file_signature(r7_path) != expected_r7:
        raise ValueError("--r7_ckpt does not match cache representation signatures.r7")
    return artifact


def load_r7_stack(path, device, need_decoder=True):
    if not path:
        raise ValueError("--r7_ckpt is required for geo-motion/RGB decode")
    from utils.r7_representation import (
        load_checkpoint, load_r7_modules, load_r7_tokenizer)
    artifact = load_checkpoint(path)
    if need_decoder:
        config, _, _, tokenizer, decoder, _ = load_r7_modules(artifact)
        decoder = decoder.to(device).eval()
    else:
        config, tokenizer, _ = load_r7_tokenizer(artifact)
        decoder = None
    tokenizer = tokenizer.to(device).eval()
    for module in (tokenizer, decoder):
        if module is not None:
            for parameter in module.parameters(): parameter.requires_grad_(False)
    return config, tokenizer, decoder


def auxiliary_losses(pred_norm, target_raw, cond_raw, target_stats, mode,
                     t, args, tokenizer=None):
    pred_raw = inverse_fp32(pred_norm, target_stats, mode)
    pred_full = torch.cat((cond_raw.float(), pred_raw), 1)
    target_full = torch.cat((cond_raw.float(), target_raw.float()), 1)
    dp, dt = pred_full[:, 1:] - pred_full[:, :-1], \
             target_full[:, 1:] - target_full[:, :-1]
    motion = masked_loss((dp - dt).square(), t, args.aux_t_min)
    ap, at = dp[:, 1:] - dp[:, :-1], dt[:, 1:] - dt[:, :-1]
    accel = masked_loss((ap - at).square(), t, args.aux_t_min)
    geo = motion.new_zeros(())
    if args.lambda_geo_motion > 0:
        if tokenizer is None: raise RuntimeError("geo aux requires strict R7 tokenizer")
        b, s, n, c = pred_full.shape; g = args.latent_grid
        pg, _ = tokenizer.decode(pred_full.reshape(b, s, g, g, c))
        tg, _ = tokenizer.decode(target_full.reshape(b, s, g, g, c))
        geo = masked_loss(((pg[:, 1:] - pg[:, :-1]) -
                           (tg[:, 1:] - tg[:, :-1])).square(), t, args.aux_t_min)
    return motion, accel, geo


@contextlib.contextmanager
def ema_weights(model, ema):
    """Swap model/EMA tensors without allocating another device-side model."""
    params = dict(model.named_parameters())
    swapped = {}
    try:
        for name, shadow in ema.shadow.items():
            if name not in params:
                continue
            parameter = params[name]
            shadow = shadow.to(
                device=parameter.device, dtype=parameter.dtype)
            swapped[name] = parameter.data
            parameter.data = shadow
            ema.shadow[name] = swapped[name]
        yield
    finally:
        for name, train_value in swapped.items():
            ema.shadow[name] = params[name].data
            params[name].data = train_value


def sample_shifted(model, cond, shape, steps, alpha, noise=None,
                   device_type="cpu", compute_dtype=torch.float32):
    model_dtype = cond.dtype
    z = (noise if noise is not None else torch.randn(
        shape, device=cond.device, dtype=model_dtype)).float()
    grid = torch.linspace(0, 1, steps + 1, device=cond.device,
                          dtype=torch.float32)
    shifted = shift_time(grid, alpha)
    for i in range(steps):
        t = torch.full((shape[0],), float(shifted[i]), device=cond.device,
                       dtype=model_dtype)
        with torch.autocast(
                device_type=device_type, dtype=compute_dtype,
                enabled=compute_dtype != torch.float32):
            velocity = model(z.to(model_dtype), t, cond=cond).float()
        z = z + velocity * (shifted[i + 1] - shifted[i])
    return z


def _mean(values): return float(sum(values) / max(len(values), 1))


def latent_metrics(generated, target, cond_raw, target_stats, mode):
    gen_raw = inverse_fp32(generated, target_stats, mode)
    tgt_raw = target.float()
    row = {"eval/latent_mse": F.mse_loss(gen_raw, tgt_raw).item(),
           "eval/gen_std_ratio": gen_raw.std().item() /
                                  max(tgt_raw.std().item(), 1e-8)}
    for i in range(FUTURE_CHUNKS):
        row[f"eval/latent_mse_chunk{i+1}"] = F.mse_loss(
            gen_raw[:, i], tgt_raw[:, i]).item()
        row[f"eval/std_ratio_chunk{i+1}"] = gen_raw[:, i].std().item() / \
            max(tgt_raw[:, i].std().item(), 1e-8)
    gf = torch.cat((cond_raw.float(), gen_raw), 1)
    tf = torch.cat((cond_raw.float(), tgt_raw), 1)
    gm, tm = gf[:, 1:] - gf[:, :-1], tf[:, 1:] - tf[:, :-1]
    ratios = gm.flatten(2).norm(dim=2) / tm.flatten(2).norm(dim=2).clamp_min(1e-8)
    cosines = F.cosine_similarity(gm.flatten(2), tm.flatten(2), dim=2)
    row["eval/motion_ratio"] = ratios.mean().item()
    row["eval/motion_cosine"] = cosines.mean().item()
    for i in range(FUTURE_CHUNKS):
        row[f"eval/motion_ratio_chunk{i+1}"] = ratios[:, i].mean().item()
        row[f"eval/motion_cosine_chunk{i+1}"] = cosines[:, i].mean().item()
    return row, gen_raw


def expanded_geo_motion_metrics(gen_raw, target_raw, cond_raw, tokenizer, args):
    b, _, _, c = gen_raw.shape
    g = args.latent_grid
    generated = torch.cat((cond_raw.float(), gen_raw.float()), 1)
    target = torch.cat((cond_raw.float(), target_raw.float()), 1)
    generated_geo, _ = tokenizer.decode(generated.reshape(b, 5, g, g, c))
    target_geo, _ = tokenizer.decode(target.reshape(b, 5, g, g, c))
    gm = generated_geo[:, 1:] - generated_geo[:, :-1]
    tm = target_geo[:, 1:] - target_geo[:, :-1]
    ratio = gm.flatten(2).norm(dim=2) / tm.flatten(2).norm(dim=2).clamp_min(1e-8)
    cosine = F.cosine_similarity(gm.flatten(2), tm.flatten(2), dim=2)
    row = {
        "eval/expanded_geo_motion_mse": F.mse_loss(gm, tm).item(),
        "eval/expanded_geo_motion_ratio": ratio.mean().item(),
        "eval/expanded_geo_motion_cosine": cosine.mean().item(),
    }
    for i in range(FUTURE_CHUNKS):
        # R7 factor=2 expands each latent future chunk into two RGB-time
        # transitions. Aggregate exactly that pair rather than mislabelling
        # individual expanded-frame transitions as latent chunks.
        start, end = i * 2, (i + 1) * 2
        row[f"eval/expanded_geo_motion_ratio_chunk{i+1}"] = \
            ratio[:, start:end].mean().item()
        row[f"eval/expanded_geo_motion_cosine_chunk{i+1}"] = \
            cosine[:, start:end].mean().item()
    return row


def decode_rgb_metrics(gen_raw, target_raw, cond_raw, tokenizer, decoder,
                       decode_chunk_size, args, lpips_model=None,
                       device_type="cpu", compute_dtype=torch.float32):
    b, _, n, c = gen_raw.shape; g = args.latent_grid
    def decode(future):
        latent = torch.cat((cond_raw.float(), future.float()), 1)
        latent = latent.reshape(b, 5, g, g, c)
        with torch.autocast(
                device_type=device_type, dtype=compute_dtype,
                enabled=compute_dtype != torch.float32):
            geo, tex = tokenizer.decode(latent)
            rgb = decoder(
                geo, tex, frames_chunk_size=decode_chunk_size)[..., :3]
        return rgb.float().clamp(0, 1)
    gen = decode(gen_raw)
    gen_cpu = gen.cpu()
    del gen
    target = decode(target_raw)
    target_cpu = target.cpu()
    gen = gen_cpu.to(cond_raw.device)
    target = target_cpu.to(cond_raw.device)
    if gen.shape[1] != 9 or target.shape[1] != 9:
        raise RuntimeError(f"R7 decode must produce exactly 9 frames, got {gen.shape[1]}")
    mse_frame = (gen - target).square().mean((0, 2, 3, 4))
    psnr = -10 * torch.log10(mse_frame.clamp_min(1e-12))
    row = {"eval/rgb_psnr_future_vs_ae_target": psnr[1:].mean().item(),
           "eval/rgb_psnr_late_half_vs_ae_target": psnr[5:].mean().item(),
           "eval/rgb_psnr_boundary_vs_ae_target": psnr[1].item(),
           "eval/rgb_boundary_motion_mse_vs_ae_target": F.mse_loss(
               gen[:, 1] - gen[:, 0], target[:, 1] - target[:, 0]).item()}
    if lpips_model is not None:
        vals = []
        for i in range(1, 9):
            a = gen[:, i].permute(0, 3, 1, 2) * 2 - 1
            b_ = target[:, i].permute(0, 3, 1, 2) * 2 - 1
            vals.append(lpips_model(a, b_).mean())
        row["eval/rgb_lpips_future_vs_ae_target"] = torch.stack(vals).mean().item()
        row["eval/rgb_lpips_late_half_vs_ae_target"] = torch.stack(vals[4:]).mean().item()
    return row, gen.cpu(), target.cpu()


def load_eval_batches(manifest, count, args):
    dataset = LatentShardDataset(manifest, shuffle_buffer=1, seed=0,
                                 repeat=False, rank=0, world_size=1)
    loader = DataLoader(dataset, batch_size=1, num_workers=0,
                        collate_fn=latent_collate_fn)
    batches = []
    for batch in loader:
        validate_batch(batch, args); batches.append(batch)
        if len(batches) >= count: break
    if not batches: raise RuntimeError("evaluation manifest is empty")
    return batches


def build_extension_scheduler(optimizer, old_lrs, peak_lr, warmup, remaining):
    if peak_lr <= 0 or remaining <= 0:
        raise ValueError("extension requires positive --extension_lr and remaining steps")
    ratio = [lr / max(max(old_lrs), 1e-12) for lr in old_lrs]
    for group, value in zip(optimizer.param_groups, ratio):
        group["lr"] = peak_lr * value; group["initial_lr"] = peak_lr * value
    warmup = min(warmup, max(remaining - 1, 0)); schedules = []; milestones = []
    if warmup:
        schedules.append(torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=max(min(max(old_lrs) / peak_lr, 1), 1e-4),
            total_iters=warmup)); milestones.append(warmup)
    schedules.append(torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(remaining - warmup, 1), eta_min=1e-6))
    return schedules[0] if len(schedules) == 1 else \
        torch.optim.lr_scheduler.SequentialLR(optimizer, schedules, milestones)


def strict_resume(checkpoint, args, stats, signature, world_size):
    saved = checkpoint.get("args", {})
    saved_max = int(saved.get("max_steps", 1 << 60))
    saved_step = int(checkpoint.get("global_step", -1))
    if args.max_steps != saved_max:
        if saved_step < saved_max:
            raise ValueError(
                "max_steps may change only after the saved run completed")
        if (args.max_steps <= saved_step or args.max_steps < saved_max or
                args.max_steps != 12000 or args.extension_lr <= 0):
            raise ValueError(
                "extension requires completed saved run, target step 12000, "
                "and positive --extension_lr")
    for key in STRICT_MODEL_ARGS + STRICT_TRAIN_ARGS:
        if saved.get(key) != getattr(args, key):
            raise ValueError(f"resume mismatch {key}: {saved.get(key)!r} != {getattr(args,key)!r}")
    saved_effective = int(saved.get("batch_size", -1)) * \
        int(saved.get("accum_steps", -1)) * int(checkpoint.get("world_size", world_size))
    current_effective = args.batch_size * args.accum_steps * world_size
    if saved_effective != current_effective:
        raise ValueError(f"resume effective batch mismatch: {saved_effective} != {current_effective}")
    for key, expected in (("context_chunks", 1), ("future_chunks", 4)):
        if int(saved.get(key, -1)) != expected: raise ValueError(f"resume {key} mismatch")
    expected_architecture = architecture_contract(args)
    if not exact_equal(checkpoint.get("architecture"), expected_architecture):
        raise ValueError("resume architecture contract is not exactly identical")
    expected_objective = objective_contract(args)
    saved_objective = checkpoint.get("objective")
    if not exact_equal(saved_objective, expected_objective):
        raise ValueError("resume objective contract is not exactly identical")
    if checkpoint.get("normalization_signature") != signature or not exact_equal(
            checkpoint.get("normalization"), stats):
        raise ValueError("resume normalization/representation is not exactly identical")


def gather_rng(use_ddp, world_size, device):
    """Collect rank-local RNG as byte tensors on NCCL/HCCL-safe collectives."""
    state = capture_rng_state()
    if not use_ddp:
        return [state]
    payload = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
    local_size = torch.tensor([len(payload)], device=device, dtype=torch.long)
    sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(sizes, local_size)
    maximum = max(int(value.item()) for value in sizes)
    local = torch.zeros(maximum, device=device, dtype=torch.uint8)
    if payload:
        local[:len(payload)] = torch.tensor(
            list(payload), device=device, dtype=torch.uint8)
    buffers = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(buffers, local)
    if not is_main_process():
        return None
    return [pickle.loads(bytes(buffer[:int(size.item())].cpu().tolist()))
            for buffer, size in zip(buffers, sizes)]


def checkpoint_payload(core, ema, optimizer, scheduler, scaler, step, args, stats,
                       signature, rng_by_rank, best, world_size):
    return {"checkpoint_version": 4, "model": core.state_dict(),
            "ema": ema.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "global_step": step,
            "world_size": world_size,
            "rng_by_rank": rng_by_rank, "args": vars(args),
            "architecture": architecture_contract(args),
            "objective": objective_contract(args),
            "representation": stats["representation"],
            "normalization": stats, "normalization_signature": signature,
            "best": best,
            "production_mode": args.normalization_mode == "zscore"}


def composite_score(row):
    std_penalty = abs(np.log(max(row["eval/gen_std_ratio"], 1e-8)))
    # A perfect aligned trajectory contributes zero; low positive cosine is
    # penalized continuously rather than only penalizing negative values.
    cosine_penalty = 1.0 - float(np.clip(row["eval/motion_cosine"], -1.0, 1.0))
    motion_penalty = abs(np.log(max(row["eval/motion_ratio"], 1e-8))) + cosine_penalty
    guarded = not (0.5 <= row["eval/gen_std_ratio"] <= 1.5 and
                   0.5 <= row["eval/motion_ratio"] <= 1.5 and
                   row["eval/motion_cosine"] >= 0.2)
    if "eval/rgb_lpips_future_vs_ae_target" in row:
        source = "rgb_composite_vs_ae_target"
        base = (row["eval/rgb_lpips_future_vs_ae_target"]
                + 1.0 / max(row["eval/rgb_psnr_future_vs_ae_target"], 1e-8)
                + 1.0 / max(row["eval/rgb_psnr_late_half_vs_ae_target"], 1e-8)
                + row["eval/rgb_boundary_motion_mse_vs_ae_target"])
    elif "eval/rgb_psnr_future_vs_ae_target" in row:
        source = "rgb_psnr_boundary_vs_ae_target"
        base = (1.0 / max(row["eval/rgb_psnr_future_vs_ae_target"], 1e-8)
                + 1.0 / max(row["eval/rgb_psnr_late_half_vs_ae_target"], 1e-8)
                + row["eval/rgb_boundary_motion_mse_vs_ae_target"])
    else:
        source, base = "latent_surrogate", row["eval/latent_mse"]
    score = float(base + std_penalty + motion_penalty)
    return (float("inf") if guarded else score), source, guarded


def main(argv=None):
    args = parse_args(argv)
    if (args.context_chunks, args.future_chunks, args.latent_dim) != (1, 4, 192):
        raise ValueError("production R7 contract is strictly context=1, future=4, D=192")
    if args.require_rgb_lpips and not args.r7_ckpt:
        raise ValueError("--require_rgb_lpips requires --r7_ckpt")
    if args.normalization_mode == "none":
        print("[WARN] normalization=none is diagnostic only; checkpoint is non-production")
    for value in (args.accum_steps, args.eval_every, args.save_every,
                  args.sample_steps, args.eval_clips):
        if value < 1: raise ValueError("step/count arguments must be positive")
    if args.throughput_divisor <= 0:
        raise ValueError("--throughput_divisor must be positive")
    if args.warmup_steps >= args.max_steps: raise ValueError("warmup must be < max_steps")
    if args.early_stop_min_steps < 6000:
        raise ValueError("production early stopping cannot begin before step 6000")
    if args.max_steps > 12000:
        raise ValueError("production schedule is capped at 12000 steps")
    if any(v < 0 for v in (args.lambda_motion, args.lambda_accel,
                            args.lambda_geo_motion, args.extension_lr)):
        raise ValueError("loss weights and extension_lr must be nonnegative")

    use_ddp, rank, local_rank, world_size = setup_ddp()
    device_type = get_device_name(); configure_backend_compatibility(device_type)
    device = get_device(local_rank); set_seed(args.seed + rank)
    main_process = is_main_process()
    if main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "samples"), exist_ok=True)
    if use_ddp: dist.barrier()

    stats = load_torch_artifact(args.stats)
    eval_stats = load_torch_artifact(args.eval_stats)
    signature = validate_stats(stats, "train stats")
    validate_stats(eval_stats, "eval stats")
    assert_same_representation(stats, eval_stats)
    st = stats_tensors(stats, device)
    representation_config = stats["representation"].get("config") or {}
    if int(representation_config.get("latent_grid", -1)) != args.latent_grid:
        raise ValueError("latent_grid differs from representation")

    dataset = LatentShardDataset(args.manifest, args.shuffle_buffer, args.seed,
                                 True, rank, world_size)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        num_workers=args.num_workers, collate_fn=latent_collate_fn,
                        pin_memory=device_type == "cuda", drop_last=True,
                        persistent_workers=args.num_workers > 0)
    eval_batches = load_eval_batches(args.eval_manifest, args.eval_clips, args) \
        if main_process else None

    core = build_model(args).to(device=device, dtype=torch.float32)
    # Keep FP32 master weights/Adam/EMA, and use autocast for the expensive DiT
    # forward. Normalization, integration state, and losses remain FP32.
    compute_dtype = resolve_dtype(args.dtype)
    model_dtype = torch.float32
    use_scaler = compute_dtype == torch.float16
    ema = EMA(core, args.ema_decay, dtype=torch.float32).to(device)
    optimizer = build_optimizer(core, args.lr, args.wd)
    scheduler = build_scheduler(optimizer, args.warmup_steps, args.max_steps)
    scaler = create_grad_scaler(enabled=use_scaler)
    step = 0; best = {"score": float("inf"), "step": 0, "bad_evals": 0,
                      "source": "none", "guarded": True}
    if args.resume:
        checkpoint = load_torch_artifact(args.resume)
        strict_resume(checkpoint, args, stats, signature, world_size)
        core.load_state_dict(checkpoint["model"], strict=True)
        ema.load_state_dict(checkpoint["ema"]); ema.to(device)
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint: scaler.load_state_dict(checkpoint["scaler"])
        step = int(checkpoint["global_step"]); best = checkpoint.get("best", best)
        saved_max = int(checkpoint["args"]["max_steps"])
        if args.max_steps != saved_max:
            if step < saved_max:
                raise ValueError("extension is only allowed after the saved run completed "
                                 "(global_step >= saved max_steps)")
            if args.max_steps <= step or args.max_steps < saved_max or args.extension_lr <= 0:
                raise ValueError("changing max_steps is only allowed as an explicit completed-run "
                                 "extension with positive --extension_lr")
            scheduler = build_extension_scheduler(
                optimizer, [g["lr"] for g in optimizer.param_groups],
                args.extension_lr, args.extension_warmup_steps, args.max_steps-step)
        else:
            scheduler.load_state_dict(checkpoint["scheduler"])
        states = checkpoint.get("rng_by_rank") or []
        if rank < len(states) and states[rank] is not None: restore_rng_state(states[rank])
        else: set_seed(args.seed + rank + step * 1009)

    r7_representation = None
    tokenizer = decoder = r7_config = None
    rgb_decode_chunk_size = None
    decode_enabled = bool(args.r7_ckpt)
    if args.r7_ckpt:
        validate_r7_artifact_representation(stats["representation"], args.r7_ckpt)
    if args.lambda_geo_motion > 0:
        r7_config, tokenizer, decoder = load_r7_stack(
            args.r7_ckpt, device, need_decoder=main_process)
        if main_process:
            rgb_decode_chunk_size = checked_decode_chunk_size(
                r7_config, decoder, args.decode_chunk_size)
    elif decode_enabled and main_process:
        try:
            r7_config, tokenizer, decoder = load_r7_stack(args.r7_ckpt, device)
            rgb_decode_chunk_size = checked_decode_chunk_size(
                r7_config, decoder, args.decode_chunk_size)
        except ValueError:
            raise
        except Exception as exc:
            print(f"[WARN] R7 RGB decode unavailable: {exc}"); decode_enabled = False
    lpips_model = None
    if main_process and decode_enabled and decoder is not None:
        try:
            import lpips
            lpips_model = lpips.LPIPS(net="alex").to(device).eval()
        except Exception as exc:
            if args.require_rgb_lpips:
                raise RuntimeError(
                    "production RGB-LPIPS evaluation could not initialize") from exc
            print(f"[WARN] LPIPS unavailable; best uses RGB PSNR/boundary: {exc}")
    if main_process and args.require_rgb_lpips and (
            not decode_enabled or decoder is None or lpips_model is None):
        raise RuntimeError(
            "production best selection requires R7 RGB decode and LPIPS")

    model = nn.parallel.DistributedDataParallel(
        core,
        device_ids=[local_rank] if device_type != "cpu" else None,
        output_device=local_rank if device_type != "cpu" else None,
        find_unused_parameters=False) if use_ddp else core
    flow_forward.time_shift_alpha = args.time_shift_alpha
    writer = SummaryWriter(os.path.join(args.output_dir, "tb"),
                           purge_step=step or None) if main_process else None
    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")

    @torch.no_grad()
    def evaluate():
        nonlocal decode_enabled
        core.eval(); aggregate = []; packs = []
        for index, cpu_batch in enumerate(eval_batches):
            cond_raw, target_raw = validate_batch(cpu_batch, args)
            cond_raw = cond_raw.to(device).float(); target_raw = target_raw.to(device).float()
            cond = normalize_fp32(cond_raw, st["cond"], args.normalization_mode).to(model_dtype)
            target = normalize_fp32(target_raw, st["target"], args.normalization_mode).to(model_dtype)
            generator = torch.Generator(device="cpu"); generator.manual_seed(args.seed + 1009*index)
            noise = torch.randn(target.shape, generator=generator).to(device, model_dtype)
            row = {}
            for bucket in VMSE_BUCKETS:
                t = torch.full((target.shape[0],), bucket, device=device, dtype=model_dtype)
                te = t.view(-1, 1, 1, 1)
                with torch.autocast(
                        device_type=device_type, dtype=compute_dtype,
                        enabled=compute_dtype != torch.float32):
                    v = core((1-te)*noise + te*target, t, cond=cond)
                row[f"eval/velocity_mse_t{bucket:.1f}"] = F.mse_loss(
                    v.float(), (target-noise).float()).item()
            row["eval/velocity_mse"] = _mean([row[k] for k in row])
            generated = sample_shifted(
                core, cond, target.shape, args.sample_steps,
                args.time_shift_alpha, noise, device_type, compute_dtype)
            latent_row, gen_raw = latent_metrics(generated, target_raw, cond_raw,
                                                  st["target"], args.normalization_mode)
            row.update(latent_row)
            if tokenizer is not None:
                try:
                    row.update(expanded_geo_motion_metrics(
                        gen_raw, target_raw, cond_raw, tokenizer, args))
                except Exception as exc:
                    print(f"[WARN] expanded-geo metrics unavailable: {exc}")
            rgb_generated = rgb_target = None
            if decode_enabled and tokenizer is not None:
                try:
                    rgb_row, rgb_generated, rgb_target = decode_rgb_metrics(
                        gen_raw, target_raw, cond_raw, tokenizer, decoder,
                        rgb_decode_chunk_size, args, lpips_model,
                        device_type, compute_dtype)
                    row.update(rgb_row)
                except Exception as exc:
                    if args.require_rgb_lpips:
                        raise RuntimeError(
                            "production decoded RGB/LPIPS evaluation failed") from exc
                    decode_enabled = False
                    print(f"[WARN] RGB decode disabled after failure (NPU tolerant): {exc}")
            aggregate.append(row)
            if index < args.sample_clips:
                packs.append({"cond_anchor": cond_raw.cpu(),
                              "target_future": target_raw.cpu(),
                              "sampled_future": gen_raw.cpu(),
                              "sampled_normalized": generated.float().cpu(),
                              "rgb_generated_9f": rgb_generated,
                              "rgb_ae_target_9f": rgb_target,
                              "video_id": cpu_batch.get("video_id", [""])[0]})
        keys = set.intersection(*(set(row) for row in aggregate))
        result = {key: _mean([row[key] for row in aggregate]) for key in keys}
        if args.require_rgb_lpips and (
                len(aggregate) != len(eval_batches)
                or "eval/rgb_lpips_future_vs_ae_target" not in result):
            raise RuntimeError(
                "production eval did not produce RGB LPIPS for every clip")
        result.update({"step": step, "eval/weights": "ema",
                       "eval/normalization_signature": signature,
                       "eval/decode_available": decode_enabled})
        score, source, guarded = composite_score(result)
        result.update({"eval/composite": score, "eval/composite_source": source,
                       "eval/composite_guarded": guarded})
        atomic_torch_save({"schema": "r7-diffusion-samples-v1", "step": step,
                           "checkpoint_weights": "ema", "samples": packs,
                           "representation": stats["representation"],
                           "normalization": stats,
                           "normalization_signature": signature},
                          os.path.join(args.output_dir, "samples",
                                       f"samples_step{step:07d}.pt"))
        core.train(); return result

    def write_checkpoint(kind, rng_states):
        payload = checkpoint_payload(core, ema, optimizer, scheduler, scaler,
                                     step, args, stats, signature, rng_states,
                                     best, world_size)
        if kind == "periodic":
            atomic_torch_save(payload, os.path.join(
                args.output_dir, f"checkpoint_step{step:07d}.pt"))
            atomic_torch_save(payload, os.path.join(args.output_dir,
                                                    "checkpoint_latest.pt"))
        elif kind == "best":
            atomic_torch_save(payload, os.path.join(args.output_dir,
                                                    "checkpoint_best.pt"))
        elif kind == "final":
            atomic_torch_save(payload, os.path.join(args.output_dir,
                                                    "checkpoint_final.pt"))
        else:
            raise ValueError(f"unknown checkpoint kind {kind!r}")

    iterator = iter(loader); optimizer.zero_grad(set_to_none=True)
    stop = False; throughput_meter = ThroughputMeter()
    while step < args.max_steps and not stop:
        totals = torch.zeros(4, device=device)
        for micro in range(args.accum_steps):
            batch = next(iterator); cond_raw, target_raw = validate_batch(batch, args)
            cond_raw = cond_raw.to(device, non_blocking=True).float()
            target_raw = target_raw.to(device, non_blocking=True).float()
            throughput_meter.update(count_latent_tokens(target_raw))
            cond = normalize_fp32(cond_raw, st["cond"], args.normalization_mode).to(model_dtype)
            target = normalize_fp32(target_raw, st["target"], args.normalization_mode).to(model_dtype)
            sync = model.no_sync() if use_ddp and micro < args.accum_steps-1 \
                else contextlib.nullcontext()
            with sync:
                with torch.autocast(
                        device_type=device_type, dtype=compute_dtype,
                        enabled=compute_dtype != torch.float32):
                    flow_loss, out = flow_forward(model, target, cond)
                scale = aux_scale(step, args)
                zero = flow_loss.new_zeros(())
                motion = accel = geo = zero
                if scale and (args.lambda_motion or args.lambda_accel or args.lambda_geo_motion):
                    motion, accel, geo = auxiliary_losses(
                        out["x1_pred"], target_raw, cond_raw, st["target"],
                        args.normalization_mode, out["t"], args, tokenizer)
                total = flow_loss + scale * (args.lambda_motion*motion +
                    args.lambda_accel*accel + args.lambda_geo_motion*geo)
                if use_scaler:
                    scaler.scale(total / args.accum_steps).backward()
                else:
                    (total / args.accum_steps).backward()
            totals += torch.stack((total.detach(), motion.detach(), accel.detach(), geo.detach()))
        if use_scaler: scaler.unscale_(optimizer)
        grad = torch.nn.utils.clip_grad_norm_(core.parameters(), args.max_grad_norm)
        if use_scaler:
            scaler.step(optimizer); scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True); scheduler.step(); ema.update(core)
        step += 1; totals /= args.accum_steps
        if use_ddp: dist.all_reduce(totals); totals /= world_size
        if main_process and step % args.log_every == 0:
            raw_throughput = throughput_meter.rate()
            throughput = raw_throughput / args.throughput_divisor
            row = {"step": step, "train/loss": totals[0].item(),
                   "train/motion_loss": totals[1].item(),
                   "train/accel_loss": totals[2].item(),
                   "train/geo_motion_loss": totals[3].item(),
                   "train/grad_norm": float(grad),
                   "train/lr": optimizer.param_groups[0]["lr"],
                   "train/DI_throughput": throughput,
                   "train/raw_DI_throughput": raw_throughput}
            append_metrics(metrics_path, row)
            for key, val in row.items():
                if key != "step": writer.add_scalar(key, val, step)
            print(f"step={step} loss={row['train/loss']:.6f} "
                  f"DI_throughput: {throughput:.2f} tokens/s/npu")

        eval_due = step % args.eval_every == 0 or step >= args.max_steps
        save_due = step % args.save_every == 0 or step >= args.max_steps
        if eval_due or save_due:
            if use_ddp: dist.barrier()
            eval_row = None
            eval_error = ""
            if main_process and eval_due:
                try:
                    with ema_weights(core, ema):
                        eval_row = evaluate()
                    score = eval_row["eval/composite"]
                    improved = np.isfinite(score) and score < best["score"]
                    if improved:
                        best.update(score=score, step=step, bad_evals=0,
                                    source=eval_row["eval/composite_source"],
                                    guarded=eval_row["eval/composite_guarded"])
                    elif step >= args.early_stop_min_steps:
                        best["bad_evals"] += 1
                    append_metrics(metrics_path, eval_row)
                    for key, val in eval_row.items():
                        if isinstance(val, (int, float)) and key != "step":
                            writer.add_scalar(key, val, step)
                    writer.flush()
                except Exception as exc:
                    eval_error = repr(exc)
            if use_ddp:
                error_flag = torch.tensor(
                    int(bool(eval_error)), device=device, dtype=torch.int32)
                dist.broadcast(error_flag, src=0)
                if error_flag.item():
                    if main_process:
                        print(f"[ERROR] synchronized evaluation failure: {eval_error}")
                    dist.destroy_process_group()
                    raise RuntimeError(
                        "rank-0 evaluation failed; all ranks stopped together")
            elif eval_error:
                raise RuntimeError(f"evaluation failed: {eval_error}")
            checkpoint_for_best = bool(
                main_process and eval_row is not None and best["step"] == step)
            if use_ddp:
                best_flag = torch.tensor(
                    int(checkpoint_for_best), device=device, dtype=torch.int32)
                dist.broadcast(best_flag, src=0)
                checkpoint_for_best = bool(best_flag.item())
            need_rng = save_due or checkpoint_for_best
            rng_states = (gather_rng(use_ddp, world_size, device)
                          if need_rng else None)
            if main_process:
                if save_due:
                    write_checkpoint("periodic", rng_states)
                if checkpoint_for_best:
                    write_checkpoint("best", rng_states)
                stop = step >= args.early_stop_min_steps and \
                    best["bad_evals"] >= args.patience
            if use_ddp:
                flag = torch.tensor(int(stop), device=device)
                dist.broadcast(flag, 0)
                stop = bool(flag.item())
                dist.barrier()

    if use_ddp: dist.barrier()
    rng_states = gather_rng(use_ddp, world_size, device)
    if main_process:
        write_checkpoint("final", rng_states)
        # latest always denotes the most recent resumable state, including an
        # early-stopped/final step that was not aligned to save_every.
        payload = checkpoint_payload(core, ema, optimizer, scheduler, scaler,
                                     step, args, stats, signature, rng_states,
                                     best, world_size)
        atomic_torch_save(payload, os.path.join(args.output_dir,
                                                "checkpoint_latest.pt"))
        writer.close()
    if use_ddp:
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
