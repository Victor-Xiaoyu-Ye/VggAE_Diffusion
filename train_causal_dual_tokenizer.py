#!/usr/bin/env python3
"""Production R7 causal dual-tokenizer reconstruction training.

``codec`` trains only the tokenizer. ``joint`` trains tokenizer plus decoder.
StreamVGGT, CompactCompressor, and TextureEncoder are frozen in both phases.
"""
from __future__ import annotations

import argparse
import contextlib
import math
import os
from dataclasses import replace
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.loader_utils import multiprocessing_loader_kwargs
from data.video_dataset import SpatialVidDataset, collate_fn
from streamvggt.models.streamvggt import StreamVGGT
from utils.device import (configure_backend_compatibility, get_device,
                          get_device_name, manual_seed_all, resolve_dtype)
from utils.distributed import is_main_process, setup_ddp
from utils.encoder_loader import load_encoder_checkpoint
from utils.r7_representation import (
    R7Config, build_contract, build_modules, checkpoint_args, encode_dual,
    load_checkpoint, load_r7_modules, load_source_modules,
    merged_representation_state, validate_resume_contract,
    validate_source_artifact)
from utils.training import (ThroughputMeter, append_metrics, atomic_torch_save,
                            build_scheduler, capture_rng_state,
                            count_latent_tokens, restore_rng_state)

_LPIPS = None


def parse_args():
    p = argparse.ArgumentParser(description="R7 causal dual tokenizer")
    for name in ("csv", "video_root", "eval_csv", "encoder_ckpt",
                 "dual_ae_ckpt", "output_dir"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--temporal_factor", type=int, choices=[2, 4], default=2)
    p.add_argument("--geo_latent_dim", type=int, default=96)
    p.add_argument("--tex_latent_dim", type=int, default=96)
    p.add_argument("--temporal_depth", type=int, default=3)
    p.add_argument("--phase", choices=["codec", "joint"], default="codec")
    p.add_argument("--seq_len", type=int, default=9)
    p.add_argument("--target_size", type=int, default=518)
    p.add_argument("--clip_duration_seconds", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--accum_steps", type=int, default=2)
    p.add_argument("--max_steps", type=int, default=6000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--pretrained_lr", type=float, default=5e-5)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--extension_lr", type=float, default=0.0,
                   help="peak LR for extending a completed same-stage run")
    p.add_argument("--extension_warmup_steps", type=int, default=200)
    for name, default in (("l1", 1.0), ("lpips", 0.5), ("latent", 0.5),
                          ("temporal", 0.1), ("accel", 0.05),
                          ("geo_motion", 0.1), ("comp_reg", 0.01)):
        p.add_argument("--lambda_" + name, type=float, default=default)
    p.add_argument("--lambda_geo_motion_cosine", type=float, default=0.5)
    p.add_argument("--lpips_chunk_size", type=int, default=1)
    p.add_argument("--lpips_resize", type=int, default=256)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--decode_retries", type=int, default=8)
    p.add_argument("--eval_clips", type=int, default=16)
    p.add_argument("--frames_chunk_size", type=int, default=0,
                   help="0 decodes the full sequence; required for temporal decoder blocks")
    p.add_argument("--resume", default="", help="same-stage full resume")
    p.add_argument("--init_ckpt", default="",
                   help="weights-only initialization for a fresh stage")
    p.add_argument("--allow_legacy_checkpoint", action="store_true")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--min_steps", type=int, default=2000)
    p.add_argument("--early_stop_patience", type=int, default=4)
    p.add_argument("--early_stop_min_delta", type=float, default=1e-3)
    p.add_argument("--gate_psnr", type=float, default=23.9)
    p.add_argument("--gate_lpips", type=float, default=0.13)
    p.add_argument("--gate_boundary_ratio", type=float, default=1.10)
    p.add_argument("--gate_geo_motion_cosine", type=float, default=0.95)
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


class R7TrainingCore(nn.Module):
    """Single DDP boundary containing every trainable forward."""
    def __init__(self, tokenizer, decoder):
        super().__init__()
        self.tokenizer, self.decoder = tokenizer, decoder

    def forward(self, geo, tex, chunk=None):
        geo_rec, tex_rec, latent = self.tokenizer(geo, tex)
        rgb = self.decoder(geo_rec, tex_rec, frames_chunk_size=chunk)
        return rgb, geo_rec, tex_rec, latent


def get_lpips(device):
    global _LPIPS
    if _LPIPS is None:
        import lpips
        _LPIPS = lpips.LPIPS(net="vgg").to(device).eval()
        for parameter in _LPIPS.parameters():
            parameter.requires_grad_(False)
    return _LPIPS


def lpips_chunked(model, pred, target, chunk_size=1, resize=256):
    """Differentiable LPIPS for BTHWC RGB, chunked across frames."""
    b, t = pred.shape[:2]
    pred = pred.permute(0, 1, 4, 2, 3).reshape(b * t, 3, *pred.shape[2:4])
    target = target.permute(0, 1, 4, 2, 3).reshape(b * t, 3, *target.shape[2:4])
    if resize and pred.shape[-2:] != (resize, resize):
        pred = F.interpolate(pred, (resize, resize), mode="bilinear",
                             align_corners=False)
        target = F.interpolate(target, (resize, resize), mode="bilinear",
                               align_corners=False)
    total, count = pred.new_zeros(()), 0
    for start in range(0, len(pred), max(1, chunk_size)):
        end = min(start + max(1, chunk_size), len(pred))
        total = total + model(pred[start:end] * 2 - 1,
                              target[start:end] * 2 - 1).sum()
        count += end - start
    return total / max(count, 1)


def temporal_loss(pred, target):
    return F.l1_loss(pred[:, 1:] - pred[:, :-1],
                     target[:, 1:] - target[:, :-1])


def acceleration_loss(pred, target):
    pd, td = pred[:, 1:] - pred[:, :-1], target[:, 1:] - target[:, :-1]
    return F.l1_loss(pd[:, 1:] - pd[:, :-1], td[:, 1:] - td[:, :-1])


def latent_regularization(latent):
    flat = latent.float().reshape(-1, latent.shape[-1])
    return (flat.mean(0).square().mean()
            + (flat.std(0, unbiased=False) - 1).square().mean())


def motion_cosine(pred, target):
    pred = (pred[:, 1:] - pred[:, :-1]).float().flatten(2)
    target = (target[:, 1:] - target[:, :-1]).float().flatten(2)
    return F.cosine_similarity(pred, target, dim=-1).mean()


def motion_cosine_loss(pred, target):
    pred_delta = (pred[:, 1:] - pred[:, :-1]).float().flatten(2)
    target_delta = (target[:, 1:] - target[:, :-1]).float().flatten(2)
    cosine = F.cosine_similarity(pred_delta, target_delta, dim=-1)
    return (1.0 - cosine).mean()


def set_trainable(module, enabled):
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def build_optimizer(core, args):
    groups = [{"params": list(core.tokenizer.parameters()), "lr": args.lr,
               "weight_decay": args.wd, "name": "tokenizer"}]
    if args.phase == "joint":
        decay, no_decay = [], []
        for name, parameter in core.decoder.named_parameters():
            (no_decay if parameter.ndim < 2 or "norm" in name or "bias" in name
             else decay).append(parameter)
        for params, wd, name in ((decay, args.wd, "decoder"),
                                 (no_decay, 0.0, "decoder_no_decay")):
            if params:
                groups.append({"params": params, "lr": args.pretrained_lr,
                               "weight_decay": wd, "name": name})
    return torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8)


def log_scalars(writer, row, step):
    if writer:
        for key, value in row.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                writer.add_scalar(key, value, step)


def build_extension_scheduler(optimizer, peak_lr, pretrained_peak_lr,
                              warmup_steps, remaining_steps, min_lr=1e-6):
    if peak_lr <= 0 or remaining_steps <= 0:
        raise ValueError("extension requires positive LR and remaining steps")
    peak_lrs = []
    for group in optimizer.param_groups:
        family = group.get("name", "tokenizer")
        value = peak_lr if family == "tokenizer" else pretrained_peak_lr
        peak_lrs.append(value)
        group["lr"] = value
        group["initial_lr"] = value
    warmup_steps = min(warmup_steps, max(remaining_steps - 1, 0))
    schedulers, milestones = [], []
    if warmup_steps:
        schedulers.append(torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-2, end_factor=1.0,
            total_iters=warmup_steps))
        milestones.append(warmup_steps)
    schedulers.append(torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(remaining_steps - warmup_steps, 1),
        eta_min=min_lr))
    if len(schedulers) == 1:
        return schedulers[0]
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers, milestones=milestones)


def resume_objective(args):
    names = ("phase", "temporal_factor", "temporal_depth", "geo_latent_dim",
             "tex_latent_dim", "seq_len", "clip_duration_seconds",
             "lambda_l1", "lambda_lpips", "lambda_latent",
             "lambda_temporal", "lambda_accel", "lambda_geo_motion",
             "lambda_geo_motion_cosine", "lambda_comp_reg",
             "lpips_resize", "lpips_chunk_size",
             "batch_size", "accum_steps", "lr", "pretrained_lr", "wd")
    return {name: getattr(args, name) for name in names}


def main():
    args = parse_args()
    if args.resume and args.init_ckpt:
        raise ValueError("--resume and --init_ckpt are mutually exclusive")
    if min(args.accum_steps, args.max_steps, args.eval_every, args.save_every) <= 0:
        raise ValueError("step/count arguments must be positive")
    if args.eval_clips <= 0:
        raise ValueError("eval_clips must be positive")
    if args.early_stop_patience < 0 or args.early_stop_min_delta < 0:
        raise ValueError("early-stop settings must be nonnegative")
    if args.extension_lr < 0 or args.extension_warmup_steps < 0:
        raise ValueError("extension settings must be nonnegative")
    if args.lambda_geo_motion_cosine < 0:
        raise ValueError("--lambda_geo_motion_cosine must be nonnegative")

    use_ddp, rank, local_rank, world_size = setup_ddp()
    main_process = is_main_process()
    device, device_type = get_device(local_rank), get_device_name()
    encoder_dtype = resolve_dtype(args.dtype)
    configure_backend_compatibility(device_type)
    manual_seed_all(args.seed + rank)
    if main_process:
        os.makedirs(os.path.join(args.output_dir, "samples"), exist_ok=True)

    artifact_path = args.resume or args.init_ckpt
    artifact = load_checkpoint(artifact_path) if artifact_path else None
    if artifact:
        config, compressor, tex_encoder, tokenizer, decoder, matched = \
            load_r7_modules(artifact)
        # An initializer already embeds the frozen source modules. Verify that
        # the separately supplied source artifact is the same content before
        # signing it into the new representation contract.
        validate_source_artifact(artifact, args.dual_ae_ckpt)
        requested = replace(
            config,
            target_size=args.target_size,
            temporal_factor=args.temporal_factor,
            temporal_depth=args.temporal_depth,
            geo_latent_dim=args.geo_latent_dim,
            tex_latent_dim=args.tex_latent_dim,
            seq_len=args.seq_len,
            clip_duration_seconds=args.clip_duration_seconds,
        )
        requested.validate()
        if requested != config:
            mismatches = {key: (getattr(config, key), getattr(requested, key))
                          for key in config.__dataclass_fields__
                          if getattr(config, key) != getattr(requested, key)}
            raise RuntimeError(f"requested R7 config differs from checkpoint: {mismatches}")
    else:
        source = load_checkpoint(args.dual_ae_ckpt)
        source_cfg, compressor, tex_encoder, decoder, matched = \
            load_source_modules(source)
        config = replace(
            source_cfg,
            target_size=args.target_size,
            input_grid=args.target_size // 14,
            temporal_factor=args.temporal_factor,
            temporal_depth=args.temporal_depth,
            geo_latent_dim=args.geo_latent_dim,
            tex_latent_dim=args.tex_latent_dim,
            seq_len=args.seq_len,
            clip_duration_seconds=args.clip_duration_seconds,
        )
        config.validate()
        _, _, tokenizer, _ = build_modules(config)

    contract = build_contract(config, encoder_ckpt=args.encoder_ckpt,
                              dual_ae_ckpt=args.dual_ae_ckpt)
    if artifact:
        saved_contract = artifact.get("representation_contract")
        if saved_contract is None:
            if not args.allow_legacy_checkpoint:
                raise RuntimeError("legacy R7 checkpoint has no contract; pass "
                                   "--allow_legacy_checkpoint to migrate it")
            if main_process:
                print("[legacy strict migration]", matched)
        else:
            validate_resume_contract(saved_contract, contract)
        saved_phase = checkpoint_args(artifact).get("phase")
        if saved_phase is None:
            if not args.allow_legacy_checkpoint:
                raise RuntimeError("checkpoint has no phase metadata; legacy migration required")
            saved_phase = "codec"
        saved_phase = str(saved_phase)
        if args.resume and saved_phase != args.phase:
            raise RuntimeError("--resume phase mismatch; use --init_ckpt")
        if args.resume:
            saved_objective = artifact.get("resume_objective")
            current_objective = resume_objective(args)
            if saved_objective is None:
                raise RuntimeError(
                    "legacy checkpoints cannot be full-resumed safely; use "
                    "--init_ckpt to start a fresh stage")
            if saved_objective != current_objective:
                changed = {
                    key: (saved_objective.get(key), current_objective.get(key))
                    for key in current_objective
                    if saved_objective.get(key) != current_objective.get(key)
                }
                raise RuntimeError(f"resume objective mismatch: {changed}")

    encoder = StreamVGGT(img_size=config.target_size, patch_size=14,
                         embed_dim=1024)
    load_encoder_checkpoint(encoder, args.encoder_ckpt, verbose=main_process)
    encoder = encoder.to(device=device, dtype=encoder_dtype).eval()
    compressor, tex_encoder = compressor.to(device).eval(), tex_encoder.to(device).eval()
    core = R7TrainingCore(tokenizer.to(device), decoder.to(device)).to(device)
    for module in (encoder, compressor, tex_encoder):
        set_trainable(module, False)
    set_trainable(core.tokenizer, True)
    set_trainable(core.decoder, args.phase == "joint")
    core.train()  # frozen decoder still checkpoints activations in codec phase
    optimizer = build_optimizer(core, args)
    scheduler = build_scheduler(optimizer, args.warmup_steps, args.max_steps)
    model = (nn.parallel.DistributedDataParallel(
        core, device_ids=[local_rank], output_device=local_rank,
        find_unused_parameters=False) if use_ddp else core)

    common_data = dict(seq_len=config.seq_len, target_size=config.target_size,
                       num_frames_per_video=config.seq_len,
                       clip_duration_seconds=config.clip_duration_seconds,
                       decode_retries=args.decode_retries)
    dataset = SpatialVidDataset(args.csv, args.video_root, **common_data)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if use_ddp else None
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=sampler is None,
                        sampler=sampler, num_workers=args.num_workers,
                        collate_fn=collate_fn, drop_last=True,
                        pin_memory=device_type == "cuda",
                        **multiprocessing_loader_kwargs(args.num_workers))
    eval_loader = None
    if main_process:
        eval_dataset = SpatialVidDataset(
            args.eval_csv, args.video_root, max_videos=args.eval_clips,
            temporal_jitter=False, **common_data)
        eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False,
                                 num_workers=0, collate_fn=collate_fn)

    step = 0
    best = {"metric": "quality_composite", "value": float("inf"), "step": -1,
            "metrics": {}, "passing": False}
    early = {"bad_evals": 0, "stopped": False}
    if args.resume:
        required = ("optimizer", "scheduler", "global_step", "rng")
        missing = [key for key in required if key not in artifact]
        if missing:
            raise RuntimeError(f"resume checkpoint lacks full state: {missing}")
        optimizer.load_state_dict(artifact["optimizer"])
        step = int(artifact["global_step"])
        saved_max = int(checkpoint_args(artifact).get("max_steps", args.max_steps))
        extending = (step >= saved_max and args.max_steps > saved_max
                     and args.extension_lr > 0)
        if extending:
            ratio = args.pretrained_lr / max(args.lr, 1e-12)
            scheduler = build_extension_scheduler(
                optimizer, args.extension_lr, args.extension_lr * ratio,
                args.extension_warmup_steps, args.max_steps - step)
        else:
            if args.max_steps != saved_max:
                raise RuntimeError(
                    "max_steps may change only for a completed run with "
                    "--extension_lr > 0")
            scheduler.load_state_dict(artifact["scheduler"])
        best.update(artifact.get("best_state", {}))
        early.update(artifact.get("early_stop", {})); early["stopped"] = False
        restore_rng_state(artifact["rng"])

    writer = SummaryWriter(os.path.join(args.output_dir, "tb")) if main_process else None
    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    chunk = args.frames_chunk_size if args.frames_chunk_size > 0 else None
    if config.decoder_temporal_blocks > 0 and chunk is not None \
            and chunk < config.seq_len:
        raise ValueError(
            "frames_chunk_size would split the pretrained decoder's temporal "
            "attention; use 0 or at least seq_len")

    @torch.no_grad()
    def frozen_encode(frames):
        return encode_dual(encoder, compressor, tex_encoder, frames, encoder_dtype)

    def payload():
        snapshot_best = {"metric": best["metric"], "value": best["value"],
                         "step": best["step"],
                         "metrics": dict(best.get("metrics", {})),
                         "passing": bool(best.get("passing", False))}
        snapshot_early = dict(early)
        return {"schema": "r7-training-checkpoint-v1",
                "model": merged_representation_state(
                    compressor, tex_encoder, core.tokenizer, core.decoder),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "global_step": step,
                "args": {**checkpoint_args(artifact or {}), **vars(args),
                         **contract["config"], "phase": args.phase,
                         "has_temporal_codec": True, "latent_dim": config.latent_dim},
                "representation_contract": contract,
                "resume_objective": resume_objective(args),
                "best_state": snapshot_best,
                "early_stop": snapshot_early, "rng": capture_rng_state()}

    def save(kind):
        name = (f"checkpoint_step{step:07d}.pt" if kind == "periodic"
                else f"checkpoint_{kind}.pt")
        current_payload = payload()
        atomic_torch_save(current_payload, os.path.join(args.output_dir, name))
        if kind == "periodic":
            atomic_torch_save(current_payload, os.path.join(
                args.output_dir, "checkpoint_latest.pt"))

    @torch.no_grad()
    def evaluate():
        from PIL import Image
        core.eval(); lp = get_lpips(device)
        psnr, l1s, perceptual, geo_rt, tex_rt, geo_cos = [], [], [], [], [], []
        transitions = [[] for _ in range(config.seq_len - 1)]
        latent_means = [[] for _ in range(config.latent_seq_len)]
        latent_stds = [[] for _ in range(config.latent_seq_len)]
        grid_saved = False
        for batch in eval_loader:
            frames = batch["frames"].to(device)
            geo, tex = frozen_encode(frames)
            pred, geo_rec, tex_rec, latent = core(geo, tex, chunk)
            pred = pred[..., :3].float().clamp(0, 1)
            target = frames.float().clamp(0, 1).permute(0, 1, 3, 4, 2)
            mse = (pred - target).square().mean((1, 2, 3, 4))
            psnr.extend((-10 * torch.log10(mse.clamp_min(1e-10))).cpu().tolist())
            l1s.append(F.l1_loss(pred, target).item())
            perceptual.append(lpips_chunked(lp, pred, target,
                args.lpips_chunk_size, args.lpips_resize).item())
            delta_error = ((pred[:, 1:] - pred[:, :-1])
                           - (target[:, 1:] - target[:, :-1])).abs().mean((0, 2, 3, 4))
            for index, value in enumerate(delta_error.cpu().tolist()):
                transitions[index].append(value)
            geo_rt.append(F.l1_loss(geo_rec, geo).item())
            tex_rt.append(F.l1_loss(tex_rec, tex).item())
            geo_cos.append(motion_cosine(geo_rec, geo).item())
            for index in range(config.latent_seq_len):
                latent_means[index].append(latent[:, index].float().mean().item())
                latent_stds[index].append(latent[:, index].float().std(unbiased=False).item())
            if not grid_saved:
                image = torch.cat([torch.cat([target[0, i], pred[0, i]], 1)
                                   for i in range(config.seq_len)], 0)
                Image.fromarray((image.cpu().numpy() * 255).astype(np.uint8)).save(
                    os.path.join(args.output_dir, "samples", f"step{step:07d}_grid.png"))
                grid_saved = True
        if not psnr:
            raise RuntimeError("evaluation dataset yielded no clips")
        trans = [float(np.mean(values)) for values in transitions]
        boundaries = [i for i in range(len(trans))
                      if i % config.temporal_factor == 0]
        within = [i for i in range(len(trans)) if i not in boundaries]
        boundary_error = float(np.mean([trans[i] for i in boundaries])) if boundaries else 0.
        within_error = float(np.mean([trans[i] for i in within])) if within else 0.
        row = {"step": step, "eval/psnr": float(np.mean(psnr)),
               "eval/psnr_std": float(np.std(psnr)), "eval/l1": float(np.mean(l1s)),
               "eval/lpips": float(np.mean(perceptual)),
               "eval/transition_within": within_error,
               "eval/transition_boundary": boundary_error,
               "eval/boundary_ratio": boundary_error / max(within_error, 1e-12),
               "eval/latent_roundtrip_geo_l1": float(np.mean(geo_rt)),
               "eval/latent_roundtrip_tex_l1": float(np.mean(tex_rt)),
               "eval/geo_motion_cosine": float(np.mean(geo_cos)),
               "eval/clips": len(psnr)}
        for i, value in enumerate(trans): row[f"eval/transition_{i}_{i+1}"] = value
        for i in range(config.latent_seq_len):
            pos = "anchor" if i == 0 else f"future_{i}"
            row[f"eval/latent_{pos}_mean"] = float(np.mean(latent_means[i]))
            row[f"eval/latent_{pos}_std"] = float(np.mean(latent_stds[i]))
        row["eval/anchor_boundary_error"] = trans[0]
        row.update({"gate/psnr": row["eval/psnr"] >= args.gate_psnr,
                    "gate/lpips": row["eval/lpips"] <= args.gate_lpips,
                    "gate/boundary": row["eval/boundary_ratio"] <= args.gate_boundary_ratio,
                    "gate/geo_motion": row["eval/geo_motion_cosine"] >= args.gate_geo_motion_cosine})
        row["gate/passed"] = all(row[key] for key in
            ("gate/psnr", "gate/lpips", "gate/boundary", "gate/geo_motion"))
        # LPIPS is primary; soft penalties prevent selecting a perceptually sharp
        # checkpoint that regresses the R7 temporal/geometry contract.
        row["eval/quality_composite"] = (
            row["eval/lpips"]
            + 0.10 * max(row["eval/boundary_ratio"] - 1.0, 0.0)
            + 0.10 * max(args.gate_geo_motion_cosine
                         - row["eval/geo_motion_cosine"], 0.0))
        append_metrics(metrics_path, row); log_scalars(writer, row, step)
        print("[eval]", row); core.train()
        return row

    def convergence(row):
        value = row["eval/quality_composite"]
        passing = bool(row["gate/passed"])
        # Any fully passing checkpoint outranks diagnostics. Within the same
        # class, minimize the reconstruction composite.
        improved = ((passing and not best.get("passing", False))
                    or (passing == best.get("passing", False)
                        and value < best["value"] - args.early_stop_min_delta))
        if improved:
            best.update(value=value, step=step, metrics=dict(row),
                        passing=passing)
            early["bad_evals"] = 0
        elif step >= args.min_steps:
            early["bad_evals"] += 1
        early["stopped"] = (args.early_stop_patience > 0 and step >= args.min_steps
                            and early["bad_evals"] >= args.early_stop_patience)
        return improved, early["stopped"]

    if main_process:
        print(f"R7 {args.phase}: factor={config.temporal_factor} "
              f"split={config.geo_latent_dim}+{config.tex_latent_dim} world={world_size}")
    optimizer.zero_grad(set_to_none=True)
    epoch = micro = 0; stopped = step >= args.max_steps; last_eval = -1
    throughput_meter = ThroughputMeter()
    while not stopped:
        if sampler: sampler.set_epoch(epoch)
        progress = tqdm(loader, disable=not main_process, desc=f"ep{epoch}")
        for batch in progress:
            frames = batch["frames"].to(device)
            if frames.shape[1] != config.seq_len:
                raise RuntimeError(f"expected {config.seq_len} frames")
            geo, tex = frozen_encode(frames)
            sync = (micro + 1) % args.accum_steps == 0
            sync_context = contextlib.nullcontext() if sync or not use_ddp else model.no_sync()
            with sync_context:
                pred, geo_rec, tex_rec, latent = model(geo, tex, chunk)
                target = frames.float().clamp(0, 1).permute(0, 1, 3, 4, 2)
                losses = {"l1": F.l1_loss(pred, target),
                          "latent": F.l1_loss(geo_rec, geo) + F.l1_loss(tex_rec, tex),
                          "temporal": temporal_loss(pred, target),
                          "accel": acceleration_loss(pred, target),
                          "geo_motion": temporal_loss(geo_rec, geo),
                          "geo_motion_cosine": motion_cosine_loss(geo_rec, geo),
                          "comp_reg": latent_regularization(latent)}
                losses["lpips"] = (lpips_chunked(get_lpips(device), pred, target,
                    args.lpips_chunk_size, args.lpips_resize)
                    if args.lambda_lpips > 0 else pred.new_zeros(()))
                loss = pred.new_zeros(())
                for name, value in losses.items():
                    loss = loss + getattr(args, "lambda_" + name) * value
                (loss / args.accum_steps).backward()
            throughput_meter.update(count_latent_tokens(latent)); micro += 1
            if not sync: continue
            grad = torch.nn.utils.clip_grad_norm_(
                [p for p in core.parameters() if p.requires_grad], args.max_grad_norm)
            optimizer.step(); optimizer.zero_grad(set_to_none=True); scheduler.step(); step += 1
            if main_process and step % args.log_every == 0:
                raw_throughput = throughput_meter.rate()
                throughput = raw_throughput
                row = {"step": step, "train/loss": loss.item(),
                       **{f"train/{k}": v.item() for k, v in losses.items()},
                       "train/grad_norm": float(grad), "train/lr": scheduler.get_last_lr()[0],
                       "DI_throughput": throughput}
                append_metrics(metrics_path, row); log_scalars(writer, row, step)
                progress.set_postfix(DI_throughput=f"{throughput:.2f} tokens/s/npu")
                progress.write(f"{datetime.now()}: {row}")
            should_eval = step % args.eval_every == 0
            should_save = step % args.save_every == 0
            if should_eval or should_save:
                if use_ddp:
                    dist.barrier()
                should_stop = False
                if main_process:
                    if should_eval:
                        row = evaluate(); last_eval = step
                        improved, should_stop = convergence(row)
                        if improved: save("best")
                        save("latest")
                    if should_save:
                        save("periodic")
                if use_ddp:
                    flag = torch.tensor(int(should_stop), device=device,
                                        dtype=torch.int32)
                    dist.broadcast(flag, src=0)
                    should_stop = bool(flag.item())
                    dist.barrier()
                stopped = should_stop
            if step >= args.max_steps: stopped = True
            if stopped: break
        epoch += 1
    if use_ddp: dist.barrier()
    if main_process:
        if last_eval != step:
            row = evaluate(); improved, _ = convergence(row)
            if improved: save("best")
        save("final"); save("latest")
        if writer: writer.close()
    if use_ddp: dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
