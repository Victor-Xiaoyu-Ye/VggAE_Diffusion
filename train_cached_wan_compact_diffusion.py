#!/usr/bin/env python3
"""Train Wan-initialized compact latent flow matching from cached shards."""

import argparse
import contextlib
import io
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.latent_shard_dataset import LatentShardDataset, latent_collate_fn
from data.loader_utils import multiprocessing_loader_kwargs
from models.flow_matching import OTCFM
from models.wan_compact_adapter import WanCompactAdapter
from train_cached_compact_diffusion import (
    compare_cache_representations,
    evaluate_preview,
    load_preview_sample,
    load_rgb_preview,
)
from utils.distributed import is_main_process, setup_ddp
from utils.device import (
    configure_backend_compatibility,
    create_grad_scaler,
    get_device,
    get_device_name,
    manual_seed_all,
    resolve_dtype,
)
from utils.latent_stats import normalize_latent, validate_latent_stats
from utils.moxing_io import is_remote_path, read_bytes
from utils.training import (
    ThroughputMeter,
    append_metrics,
    atomic_torch_save,
    build_optimizer,
    build_scheduler,
    capture_rng_state,
    count_latent_tokens,
    restore_rng_state,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train WanCompactAdapter from cached compact latents")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--eval_manifest", default="")
    parser.add_argument("--eval_stats", default="")
    parser.add_argument("--eval_i0_path", default="")
    parser.add_argument("--i0_decoder_ckpt", default="")
    parser.add_argument("--wan_ckpt_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument(
        "--resume_mode",
        choices=["full", "weights"],
        default="weights",
        help=(
            "full restores optimizer/scheduler/scaler/RNG; weights loads "
            "trainable Wan adapter weights and EMA, then starts a fresh "
            "optimizer schedule"))

    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--latent_grid", type=int, default=18)
    parser.add_argument("--seq_len", type=int, default=7)
    parser.add_argument("--train_text_adapter", action="store_true")
    parser.add_argument(
        "--freeze_wan_qkv",
        action="store_true",
        help=(
            "Keep Wan self-attention QKV frozen. Use this only for adapter-only "
            "warmup or memory debugging."))
    parser.add_argument(
        "--train_qkv_last_n",
        type=int,
        default=0,
        help=(
            "When QKV is trainable, unfreeze only the last N Wan blocks. "
            "0 means all blocks. Use a small value such as 4 for 14B DDP."))
    parser.add_argument("--ddp_bucket_cap_mb", type=int, default=64)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--wd", type=float, default=1e-2)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--shuffle_buffer", type=int, default=512)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--eval_every", type=int, default=2000)
    parser.add_argument("--sample_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=0)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    manual_seed_all(seed)


def trainable_state_dict(model):
    state = model.state_dict()
    trainable = {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    return {
        name: value.detach().cpu()
        for name, value in state.items()
        if name in trainable
    }


def load_trainable_state_dict(model, state_dict, label):
    parameters = dict(model.named_parameters())
    missing = []
    for name, parameter in parameters.items():
        if not parameter.requires_grad:
            continue
        if name not in state_dict:
            missing.append(name)
            continue
        parameter.data.copy_(state_dict[name].to(
            device=parameter.device, dtype=parameter.dtype))
    unexpected = sorted(set(state_dict) - set(parameters))
    if missing or unexpected:
        raise ValueError(
            f"{label} trainable state mismatch: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}")


class TrainableEMA:
    """EMA only for trainable Wan adapter parameters.

    Full Wan weights are loaded from ``--wan_ckpt_dir`` on every run and are
    intentionally excluded from checkpoints.
    """

    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {
            name: parameter.detach().float().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    def update(self, model):
        for name, parameter in model.named_parameters():
            if name not in self.shadow:
                continue
            self.shadow[name] = self.shadow[name].to(parameter.device)
            value = parameter.detach().float()
            self.shadow[name].mul_(self.decay).add_(value, alpha=1 - self.decay)

    def state_dict(self):
        return {
            name: value.detach().cpu()
            for name, value in self.shadow.items()
        }

    def load_state_dict(self, state_dict):
        self.shadow = {
            name: value.detach().float().clone()
            for name, value in state_dict.items()
        }

    def copy_to(self, model):
        load_trainable_state_dict(model, self.shadow, "EMA")


def wan_config(model):
    return {
        "dim": int(model.wan_dim),
        "freq_dim": int(model.freq_dim),
        "num_heads": int(model.num_heads),
        "num_layers": int(getattr(model.wan, "num_layers")),
        "model_type": str(getattr(model.wan, "model_type")),
        "text_dim": int(getattr(model.wan, "text_dim")),
    }


def cast_frozen_parameters(model, dtype):
    for parameter in model.parameters():
        if parameter.requires_grad or not parameter.is_floating_point():
            continue
        parameter.data = parameter.data.to(dtype=dtype)


def save_checkpoint(path, model, ema, optimizer, scheduler, scaler, step,
                    stats, args, config):
    atomic_torch_save({
        "checkpoint_version": 1,
        "architecture": "wan_compact_adapter",
        "model_trainable": trainable_state_dict(model),
        "ema_trainable": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "global_step": step,
        "normalization": stats,
        "wan_config": config,
        "rng_state": capture_rng_state(),
        "args": vars(args),
    }, path)


def validate_resume(checkpoint, cache_stats, args, config):
    if checkpoint.get("architecture") != "wan_compact_adapter":
        raise ValueError(
            f"Expected wan_compact_adapter checkpoint, got "
            f"{checkpoint.get('architecture')!r}")
    saved_args = checkpoint.get("args", {})
    for key in (
            "latent_dim", "latent_grid", "seq_len", "train_text_adapter",
            "freeze_wan_qkv", "train_qkv_last_n"):
        if saved_args.get(key, getattr(args, key)) != getattr(args, key):
            raise ValueError(
                f"Resume mismatch for {key}: "
                f"checkpoint={saved_args.get(key)}, current={getattr(args, key)}")
    if checkpoint.get("wan_config") != config:
        raise ValueError(
            f"Wan checkpoint config mismatch: "
            f"checkpoint={checkpoint.get('wan_config')}, current={config}")

    saved_stats = checkpoint.get("normalization")
    if saved_stats is None:
        raise ValueError("Resume checkpoint has no cached-latent normalization")
    compare_cache_representations(
        saved_stats.get("representation"),
        cache_stats.get("representation"),
        label="resume/cache")
    for group in ("target", "cond"):
        for key in ("mean", "std"):
            if not torch.equal(
                    saved_stats[group][key].cpu(),
                    cache_stats[group][key].cpu()):
                raise ValueError(
                    f"Resume normalization mismatch for {group}.{key}")


def main():
    args = parse_args()
    if args.warmup_steps >= args.max_steps:
        adjusted = max(1, args.max_steps - 1)
        print(
            f"[WARN] --warmup_steps={args.warmup_steps} must be smaller "
            f"than --max_steps={args.max_steps}; using {adjusted}")
        args.warmup_steps = adjusted
    if args.sample_steps < 1:
        raise ValueError("--sample_steps must be positive")

    use_ddp, rank, local_rank, world_size = setup_ddp()
    main_process = is_main_process()
    device_type = get_device_name()
    configure_backend_compatibility(device_type)
    if device_type == "cpu":
        raise RuntimeError("Wan cached diffusion training requires accelerator")
    device = get_device(local_rank)
    model_dtype = resolve_dtype(args.dtype)
    use_scaler = model_dtype == torch.float16
    set_seed(args.seed + rank)
    os.makedirs(args.output_dir, exist_ok=True)

    stats_path = (
        io.BytesIO(read_bytes(args.stats))
        if is_remote_path(args.stats) else args.stats)
    cache_stats = torch.load(
        stats_path, map_location="cpu", weights_only=False)
    validate_latent_stats(
        cache_stats["target"], args.seq_len, args.latent_dim, name="target")
    validate_latent_stats(
        cache_stats["cond"], 1, args.latent_dim, name="condition")
    target_mean = cache_stats["target"]["mean"].to(
        device=device, dtype=torch.float32)
    target_std = cache_stats["target"]["std"].to(
        device=device, dtype=torch.float32).clamp_min(1e-6)
    cond_mean = cache_stats["cond"]["mean"].to(
        device=device, dtype=torch.float32)
    cond_std = cache_stats["cond"]["std"].to(
        device=device, dtype=torch.float32).clamp_min(1e-6)
    representation = cache_stats.get("representation", {})
    if representation:
        if int(representation["latent_grid"]) != args.latent_grid:
            raise ValueError(
                f"Stats latent_grid={representation['latent_grid']} "
                f"!= trainer {args.latent_grid}")
        expected_future_frames = int(representation["seq_len"]) - 1
        if expected_future_frames != args.seq_len:
            raise ValueError(
                f"Cache contains {expected_future_frames} future frames, "
                f"trainer expects {args.seq_len}")

    eval_manifest = args.eval_manifest or args.manifest
    preview_batch = None
    rgb_preview = None
    if main_process:
        if args.eval_stats:
            eval_stats_path = (
                io.BytesIO(read_bytes(args.eval_stats))
                if is_remote_path(args.eval_stats) else args.eval_stats)
            eval_stats = torch.load(
                eval_stats_path, map_location="cpu", weights_only=False)
            compare_cache_representations(
                representation,
                eval_stats.get("representation"),
                label="train/eval cache")
        preview_batch = load_preview_sample(eval_manifest)
        print(
            "Preview cache sample: "
            f"{preview_batch['video_id'][0] or '<unknown>'}")
        rgb_preview = load_rgb_preview(args, representation, preview_batch)

    dataset = LatentShardDataset(
        args.manifest,
        shuffle_buffer=args.shuffle_buffer,
        seed=args.seed,
        repeat=True,
        rank=rank,
        world_size=world_size,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=latent_collate_fn,
        pin_memory=device_type == "cuda",
        drop_last=True,
        **multiprocessing_loader_kwargs(args.num_workers),
    )

    model = WanCompactAdapter(
        args.wan_ckpt_dir,
        latent_dim=args.latent_dim,
        latent_grid=args.latent_grid,
        seq_len=args.seq_len,
        i0_condition=True,
        train_text_adapter=args.train_text_adapter,
        train_qkv=not args.freeze_wan_qkv,
        train_qkv_last_n=args.train_qkv_last_n,
    )
    cast_frozen_parameters(model, model_dtype)
    model = model.to(device=device)
    config = wan_config(model)
    flow = OTCFM(model)
    ema = TrainableEMA(model, decay=args.ema_decay)
    optimizer = build_optimizer(model, lr=args.lr, wd=args.wd)
    scheduler = build_scheduler(
        optimizer, warmup_steps=args.warmup_steps, total_steps=args.max_steps)
    scaler = create_grad_scaler(enabled=use_scaler)
    global_step = 0

    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(
                f"Resume checkpoint not found: {args.resume}")
        checkpoint = torch.load(
            args.resume, map_location="cpu", weights_only=False)
        validate_resume(checkpoint, cache_stats, args, config)
        load_trainable_state_dict(
            model, checkpoint["model_trainable"], "model")
        ema.load_state_dict(checkpoint["ema_trainable"])
        global_step = int(checkpoint["global_step"])
        if args.resume_mode == "full":
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            if "scaler" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler"])
            if rank == 0:
                restore_rng_state(checkpoint.get("rng_state"))
            else:
                set_seed(args.seed + rank + global_step * 1009)
            if main_process:
                print(
                    f"Resumed full Wan adapter state from {args.resume} "
                    f"at step {global_step}")
        else:
            set_seed(args.seed + rank + global_step * 1009)
            if main_process:
                print(
                    f"Resumed Wan adapter weights from {args.resume} at "
                    f"step {global_step}; optimizer schedule reset")
        if global_step >= args.max_steps:
            raise ValueError(
                f"Resume checkpoint is already at step {global_step}, but "
                f"--max_steps is {args.max_steps}")

    if use_ddp:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            bucket_cap_mb=args.ddp_bucket_cap_mb)
        flow.model = model

    writer = (
        SummaryWriter(
            log_dir=os.path.join(args.output_dir, "tb"),
            purge_step=global_step if global_step > 0 else None)
        if main_process else None
    )
    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    base_model = model.module if use_ddp else model
    if main_process:
        trainable_params = sum(
            parameter.numel()
            for parameter in base_model.parameters()
            if parameter.requires_grad)
        total_params = sum(parameter.numel() for parameter in base_model.parameters())
        print(
            f"Training cached WanCompactAdapter: "
            f"{trainable_params / 1e6:.1f}M trainable / "
            f"{total_params / 1e9:.2f}B total, "
            f"{cache_stats.get('num_samples', 'unknown')} cached videos, "
            f"{world_size} {device_type.upper()} devices")
        append_metrics(metrics_path, {
            "run/start": True,
            "architecture": "wan_compact_adapter",
            "resume_step": global_step,
            "max_steps": args.max_steps,
            "world_size": world_size,
            "batch_size": args.batch_size,
            "accum_steps": args.accum_steps,
            "effective_batch": world_size * args.batch_size * args.accum_steps,
            "lr": args.lr,
            "resume_mode": args.resume_mode,
            "freeze_wan_qkv": args.freeze_wan_qkv,
            "train_qkv_last_n": args.train_qkv_last_n,
            "ddp_bucket_cap_mb": args.ddp_bucket_cap_mb,
            **{f"wan/{key}": value for key, value in config.items()},
        })

    data_iterator = iter(dataloader)
    optimizer.zero_grad(set_to_none=True)
    last_preview_step = -1
    throughput_meter = ThroughputMeter()
    pbar = tqdm(
        total=args.max_steps, initial=global_step,
        desc="Training cached WanCompactAdapter",
        disable=not main_process, dynamic_ncols=True)

    while global_step < args.max_steps:
        accumulated_loss = torch.zeros((), device=device)
        for micro_step in range(args.accum_steps):
            batch = next(data_iterator)
            target_raw = batch["target"].to(
                device=device, dtype=torch.float32, non_blocking=True)
            cond_raw = batch["cond"].to(
                device=device, dtype=torch.float32, non_blocking=True)
            throughput_meter.update(count_latent_tokens(target_raw))
            expected_shape = (
                args.seq_len, args.latent_grid ** 2, args.latent_dim)
            if target_raw.shape[1:] != expected_shape:
                raise ValueError(
                    f"Unexpected target shape {tuple(target_raw.shape)}, "
                    f"expected batch x {expected_shape}")

            target = normalize_latent(
                target_raw, {"mean": target_mean, "std": target_std}
            ).to(model_dtype)
            cond = normalize_latent(
                cond_raw, {"mean": cond_mean, "std": cond_std}
            ).to(model_dtype)

            sync_context = contextlib.nullcontext()
            if use_ddp and micro_step < args.accum_steps - 1:
                sync_context = model.no_sync()
            with sync_context:
                with torch.autocast(
                        device_type=device_type, dtype=model_dtype,
                        enabled=model_dtype != torch.float32):
                    loss = flow.compute_loss(target, cond=cond)
                scaled_loss = loss / args.accum_steps
                if use_scaler:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
            accumulated_loss += loss.detach()

        if use_scaler:
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.max_grad_norm)
        if use_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        base_model = model.module if use_ddp else model
        ema.update(base_model)
        global_step += 1
        pbar.update(1)

        mean_loss = accumulated_loss / args.accum_steps
        if use_ddp:
            dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
            mean_loss /= world_size

        if main_process and global_step % args.log_every == 0:
            loss_value = mean_loss.item()
            lr = optimizer.param_groups[0]["lr"]
            raw_throughput = throughput_meter.rate()
            throughput = raw_throughput
            print(
                f"step={global_step} loss={loss_value:.6f} "
                f"lr={lr:.3e} grad_norm={float(grad_norm):.3f} "
                f"DI_throughput: {throughput:.2f} tokens/s/npu")
            pbar.set_postfix(
                loss=f"{loss_value:.4f}",
                DI_throughput=f"{throughput:.2f} tokens/s/npu")
            writer.add_scalar("train/loss", loss_value, global_step)
            writer.add_scalar("train/lr", lr, global_step)
            writer.add_scalar(
                "train/grad_norm", float(grad_norm), global_step)
            writer.add_scalar(
                "DI_throughput", throughput, global_step)
            append_metrics(metrics_path, {
                "step": global_step,
                "train/loss": loss_value,
                "train/lr": lr,
                "train/grad_norm": float(grad_norm),
                "DI_throughput": throughput,
            })

        save_due = global_step % args.save_every == 0
        eval_due = global_step % args.eval_every == 0
        if main_process and save_due:
            step_checkpoint = os.path.join(
                args.output_dir, f"checkpoint_step{global_step:08d}.pt")
            save_checkpoint(
                step_checkpoint, base_model, ema, optimizer, scheduler,
                scaler, global_step, cache_stats, args, config)
            save_checkpoint(
                os.path.join(args.output_dir, "checkpoint_latest.pt"),
                base_model, ema, optimizer, scheduler, scaler,
                global_step, cache_stats, args, config)
            print(f"Saved checkpoint at step {global_step}")

        should_eval = save_due or eval_due
        if use_ddp and should_eval:
            dist.barrier()
        if main_process and should_eval:
            training_state = trainable_state_dict(base_model)
            try:
                ema.copy_to(base_model)
                eval_metrics = evaluate_preview(
                    base_model, preview_batch,
                    target_mean, target_std, cond_mean, cond_std,
                    device, model_dtype, args, global_step,
                    device_type,
                    rgb_preview=rgb_preview)
                for name, value in eval_metrics.items():
                    writer.add_scalar(f"eval/{name}", value, global_step)
                writer.flush()
                append_metrics(metrics_path, {
                    "step": global_step,
                    **{
                        f"eval/{name}": value
                        for name, value in eval_metrics.items()
                    },
                })
                print(
                    f"preview step={global_step} "
                    f"velocity_mse={eval_metrics['velocity_mse']:.6f} "
                    f"generated_std={eval_metrics['generated_std']:.4f}")
                last_preview_step = global_step
            finally:
                load_trainable_state_dict(
                    base_model, training_state, "training restore")
                del training_state
        if use_ddp and should_eval:
            dist.barrier()

    pbar.close()

    if main_process:
        save_checkpoint(
            os.path.join(args.output_dir, "checkpoint_final.pt"),
            base_model, ema, optimizer, scheduler, scaler, global_step,
            cache_stats, args, config)
        if last_preview_step != global_step:
            training_state = trainable_state_dict(base_model)
            try:
                ema.copy_to(base_model)
                eval_metrics = evaluate_preview(
                    base_model, preview_batch,
                    target_mean, target_std, cond_mean, cond_std,
                    device, model_dtype, args, global_step,
                    device_type,
                    rgb_preview=rgb_preview)
                for name, value in eval_metrics.items():
                    writer.add_scalar(f"eval/{name}", value, global_step)
                append_metrics(metrics_path, {
                    "step": global_step,
                    "final": True,
                    **{
                        f"eval/{name}": value
                        for name, value in eval_metrics.items()
                    },
                })
            finally:
                load_trainable_state_dict(
                    base_model, training_state, "training restore")
        writer.close()

    if use_ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
