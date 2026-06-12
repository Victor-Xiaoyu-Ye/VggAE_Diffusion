#!/usr/bin/env python3
"""Train I0-conditioned flow matching from cached compact latent shards."""

import argparse
import contextlib
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.latent_shard_dataset import LatentShardDataset, latent_collate_fn
from models.compact_dit import CompactLatentDiT
from models.flow_matching import OTCFM
from utils.distributed import is_main_process, setup_ddp
from utils.device import (
    create_grad_scaler,
    empty_cache,
    get_device,
    get_device_name,
    manual_seed_all,
    resolve_dtype,
)
from utils.training import (
    EMA,
    append_metrics,
    atomic_torch_save,
    build_optimizer,
    build_scheduler,
    capture_rng_state,
    restore_rng_state,
)
from utils.latent_stats import (
    denormalize_latent,
    normalize_latent,
    validate_latent_stats,
)
from utils.moxing_io import stage_remote_file


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CompactLatentDiT from cached latent tar shards")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--eval_manifest", default="")
    parser.add_argument("--eval_stats", default="")
    parser.add_argument(
        "--eval_i0_path", default="",
        help="Legacy fallback; eval cache i0_rgb is preferred")
    parser.add_argument("--i0_decoder_ckpt", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resume", default="")

    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--latent_grid", type=int, default=18)
    parser.add_argument("--seq_len", type=int, default=7,
                        help="Number of generated future frames")
    parser.add_argument("--model_dim", type=int, default=768)
    parser.add_argument("--spatial_depth", type=int, default=8)
    parser.add_argument("--temporal_depth", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--time_scale", type=float, default=1000.0)

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accum_steps", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=500000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-2)
    parser.add_argument("--warmup_steps", type=int, default=5000)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--shuffle_buffer", type=int, default=256)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=10000)
    parser.add_argument("--eval_every", type=int, default=10000)
    parser.add_argument("--sample_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=0)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    manual_seed_all(seed)


def save_checkpoint(path, model, ema, optimizer, scheduler, scaler, step,
                    stats, args):
    atomic_torch_save({
        "checkpoint_version": 2,
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "global_step": step,
        "normalization": stats,
        "rng_state": capture_rng_state(),
        "args": vars(args),
    }, path)


def validate_resume(checkpoint, cache_stats, args):
    saved_args = checkpoint.get("args", {})
    for key in (
            "latent_dim", "latent_grid", "seq_len", "model_dim",
            "spatial_depth", "temporal_depth", "num_heads", "time_scale"):
        default = 1.0 if key == "time_scale" else getattr(args, key)
        if float(saved_args.get(key, default)) != float(getattr(args, key)):
            raise ValueError(
                f"Resume mismatch for {key}: "
                f"checkpoint={saved_args.get(key)}, current={getattr(args, key)}")

    saved_stats = checkpoint.get("normalization")
    if saved_stats is None:
        raise ValueError("Resume checkpoint has no cached-latent normalization")
    if saved_stats.get("representation") != cache_stats.get("representation"):
        raise ValueError(
            "Resume checkpoint was trained with a different representation")
    for group in ("target", "cond"):
        for key in ("mean", "std"):
            if not torch.equal(
                    saved_stats[group][key].cpu(),
                    cache_stats[group][key].cpu()):
                raise ValueError(
                    f"Resume normalization mismatch for {group}.{key}")


def load_preview_sample(manifest_path):
    dataset = LatentShardDataset(
        manifest_path,
        shuffle_buffer=1,
        seed=0,
        repeat=False,
        rank=0,
        world_size=1,
    )
    try:
        return latent_collate_fn([next(iter(dataset))])
    except StopIteration as exc:
        raise RuntimeError(
            f"Evaluation manifest is empty: {manifest_path}") from exc


def _colorize_map(value):
    value = value.clamp(0, 1)
    return torch.stack([
        value,
        1 - (2 * value - 1).abs(),
        1 - value,
    ], dim=-1)


def save_latent_preview(target_raw, generated_raw, path, latent_grid):
    target_map = target_raw.square().mean(dim=-1).sqrt()
    generated_map = generated_raw.square().mean(dim=-1).sqrt()
    combined = torch.cat([
        target_map.flatten(), generated_map.flatten()
    ]).float().cpu()
    low = torch.quantile(combined, 0.02).to(target_map.device)
    high = torch.quantile(combined, 0.98).to(target_map.device)
    scale = (high - low).clamp_min(1e-6)

    rows = []
    for maps in (target_map, generated_map):
        frames = []
        for frame in maps[0]:
            normalized = ((frame - low) / scale).clamp(0, 1)
            frame_rgb = _colorize_map(
                normalized.reshape(latent_grid, latent_grid))
            frames.append(frame_rgb)
        rows.append(torch.cat(frames, dim=1))
    image = torch.cat(rows, dim=0)
    image = (image.cpu().numpy() * 255).round().astype(np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(image).resize(
        (image.shape[1] * 8, image.shape[0] * 8),
        Image.Resampling.NEAREST,
    ).save(path)


def load_rgb_preview(args, representation, preview_batch):
    if not args.i0_decoder_ckpt:
        return None

    from models.appearance_cnn import AppearanceCNN
    from models.i0_decoder import (
        I0ConditionalDecoder, load_i0_decoder_state_dict)

    checkpoint = torch.load(
        args.i0_decoder_ckpt, map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    for key, expected in (
            ("latent_dim", args.latent_dim),
            ("latent_grid", args.latent_grid)):
        if key in checkpoint_args and int(checkpoint_args[key]) != expected:
            raise ValueError(
                f"I0 decoder {key}={checkpoint_args[key]} "
                f"does not match cached trainer {expected}")
    target_size = int(
        checkpoint_args.get(
            "target_size", representation.get("target_size", 518)))
    if "i0_rgb" in preview_batch:
        reference = preview_batch["i0_rgb"].float() / 255.0
        if reference.shape[-2:] != (target_size, target_size):
            reference = torch.nn.functional.interpolate(
                reference, size=(target_size, target_size),
                mode="bilinear", align_corners=False)
    elif args.eval_i0_path:
        image = Image.open(args.eval_i0_path).convert("RGB").resize(
            (target_size, target_size), Image.Resampling.BILINEAR)
        reference = torch.from_numpy(
            np.asarray(image, dtype=np.float32) / 255.0
        ).permute(2, 0, 1).unsqueeze(0)
    else:
        print(
            "[WARN] Evaluation cache has no i0_rgb; skipping RGB preview. "
            "Rebuild it with cache_compact_latents.py --store_i0_rgb.")
        return None

    app_cnn = AppearanceCNN().eval()
    decoder = I0ConditionalDecoder(
        latent_dim=args.latent_dim,
        base_dim=int(checkpoint_args.get("decoder_base_dim", 384)),
        img_size=target_size,
        latent_grid=args.latent_grid,
        num_resblocks=int(
            checkpoint_args.get("decoder_num_resblocks", 2)),
        use_checkpoint=False,
    ).eval()
    app_cnn.load_state_dict(checkpoint["app_cnn"])
    load_i0_decoder_state_dict(decoder, checkpoint["decoder"])
    return {
        "reference": reference,
        "app_cnn": app_cnn,
        "decoder": decoder,
        "target_size": target_size,
    }


@torch.no_grad()
def save_rgb_preview(target_raw, generated_raw, cond_raw, rgb_preview,
                     device, model_dtype, device_type, path, latent_grid,
                     latent_dim):
    app_cnn = rgb_preview["app_cnn"].to(
        device=device, dtype=model_dtype)
    decoder = rgb_preview["decoder"].to(device=device)
    reference = rgb_preview["reference"].to(
        device=device, dtype=model_dtype)
    cond_grid = cond_raw.reshape(
        1, 1, latent_grid, latent_grid, latent_dim)
    try:
        appearance = app_cnn(reference)

        def decode(residual):
            future = residual + cond_raw.expand_as(residual)
            future_grid = future.reshape(
                1, residual.shape[1], latent_grid, latent_grid, latent_dim)
            video_latent = torch.cat([cond_grid, future_grid], dim=1).float()
            with torch.autocast(
                    device_type=device_type, dtype=model_dtype,
                    enabled=model_dtype != torch.float32):
                predictions, _ = decoder(video_latent, appearance)
            return predictions[0].clamp(0, 1)

        target_rgb = decode(target_raw)
        generated_rgb = decode(generated_raw)
        rows = []
        for video in (target_rgb, generated_rgb):
            rows.append(torch.cat([frame for frame in video], dim=1))
        comparison = torch.cat(rows, dim=0)
        comparison = (
            comparison.float().cpu().numpy() * 255).round().astype(np.uint8)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Image.fromarray(comparison).save(path)
    finally:
        rgb_preview["app_cnn"] = app_cnn.cpu()
        rgb_preview["decoder"] = decoder.cpu()
        empty_cache()


@torch.no_grad()
def evaluate_preview(model, preview_batch, target_mean, target_std,
                     cond_mean, cond_std, device, model_dtype, args, step,
                     device_type, rgb_preview=None):
    was_training = model.training
    model.eval()
    target_raw = preview_batch["target"].to(
        device=device, dtype=torch.float32)
    cond_raw = preview_batch["cond"].to(
        device=device, dtype=torch.float32)
    target_stats = {"mean": target_mean, "std": target_std}
    cond_stats = {"mean": cond_mean, "std": cond_std}
    target = normalize_latent(target_raw, target_stats).to(model_dtype)
    cond = normalize_latent(cond_raw, cond_stats).to(model_dtype)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + 17)
    x0 = torch.randn(
        target.shape, device="cpu", dtype=torch.float32,
        generator=generator).to(device=device, dtype=model_dtype)
    velocity_target = target - x0

    velocity_losses = []
    for t_value in (0.1, 0.5, 0.9):
        t = torch.full(
            (target.shape[0],), t_value,
            device=device, dtype=model_dtype)
        t_view = t.view(-1, 1, 1, 1)
        xt = (1 - t_view) * x0 + t_view * target
        with torch.autocast(
                device_type=device_type, dtype=model_dtype,
                enabled=model_dtype != torch.float32):
            velocity = model(xt, t, cond=cond)
        velocity_losses.append(
            torch.nn.functional.mse_loss(
                velocity.float(), velocity_target.float()))

    z = x0.clone()
    dt = 1.0 / args.sample_steps
    for index in range(args.sample_steps):
        t = torch.full(
            (target.shape[0],), index / args.sample_steps,
            device=device, dtype=model_dtype)
        with torch.autocast(
                device_type=device_type, dtype=model_dtype,
                enabled=model_dtype != torch.float32):
            velocity = model(z, t, cond=cond)
        z_mid = z + 0.5 * dt * velocity
        t_mid = torch.full(
            (target.shape[0],), (index + 0.5) / args.sample_steps,
            device=device, dtype=model_dtype)
        with torch.autocast(
                device_type=device_type, dtype=model_dtype,
                enabled=model_dtype != torch.float32):
            midpoint_velocity = model(z_mid, t_mid, cond=cond)
        z = z + dt * midpoint_velocity

    generated_raw = denormalize_latent(z, target_stats)
    save_latent_preview(
        target_raw, generated_raw,
        os.path.join(
            args.output_dir, "samples",
            f"latent_preview_step{step:08d}.png"),
        args.latent_grid,
    )
    if rgb_preview is not None:
        try:
            save_rgb_preview(
                target_raw, generated_raw, cond_raw, rgb_preview,
                device, model_dtype, device_type,
                os.path.join(
                    args.output_dir, "samples",
                    f"rgb_preview_step{step:08d}.png"),
                args.latent_grid, args.latent_dim,
            )
        except Exception as error:
            print(f"[WARN] RGB preview failed: {error}")
    if was_training:
        model.train()
    return {
        "velocity_mse": torch.stack(velocity_losses).mean().item(),
        "target_mean": target_raw.mean().item(),
        "target_std": target_raw.std().item(),
        "generated_mean": generated_raw.mean().item(),
        "generated_std": generated_raw.std().item(),
    }


def main():
    args = parse_args()
    if args.warmup_steps >= args.max_steps:
        raise ValueError("--warmup_steps must be smaller than --max_steps")
    if args.eval_every < 1 or args.save_every < 1 or args.log_every < 1:
        raise ValueError("log/save/eval intervals must be positive")
    if args.sample_steps < 1:
        raise ValueError("--sample_steps must be positive")

    use_ddp, rank, local_rank, world_size = setup_ddp()
    device_type = get_device_name()
    if device_type == "cpu":
        raise RuntimeError(
            "Cached diffusion training requires an accelerator")
    device = get_device(local_rank)
    model_dtype = resolve_dtype(args.dtype)
    use_scaler = model_dtype == torch.float16
    set_seed(args.seed + rank)
    os.makedirs(args.output_dir, exist_ok=True)

    metadata_cache = os.environ.get(
        "MOX_METADATA_CACHE_DIR",
        "/cache/yexiaoyu/vggae_runtime/cache/metadata")
    stats_path = stage_remote_file(
        args.stats, metadata_cache, max_cache_bytes=10 * 1024 ** 3)
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
                f"Stats latent grid {representation['latent_grid']} "
                f"!= {args.latent_grid}")
        expected_future_frames = int(representation["seq_len"]) - 1
        if expected_future_frames != args.seq_len:
            raise ValueError(
                f"Cache contains {expected_future_frames} future frames, "
                f"trainer expects {args.seq_len}")

    eval_manifest = args.eval_manifest or args.manifest
    preview_batch = None
    rgb_preview = None
    if is_main_process():
        if not args.eval_manifest:
            print(
                "[WARN] --eval_manifest not set; preview metrics use "
                "a fixed training-cache sample")
        if args.eval_stats:
            eval_stats_path = stage_remote_file(
                args.eval_stats, metadata_cache,
                max_cache_bytes=10 * 1024 ** 3)
            eval_stats = torch.load(
                eval_stats_path, map_location="cpu", weights_only=False)
            if eval_stats.get("representation") != representation:
                raise ValueError(
                    "Evaluation cache uses a different representation")
        preview_batch = load_preview_sample(eval_manifest)
        print(
            "Preview cache sample: "
            f"{preview_batch['video_id'][0] or '<unknown>'}")
        rgb_preview = load_rgb_preview(
            args, representation, preview_batch)

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
        persistent_workers=args.num_workers > 0,
    )

    model = CompactLatentDiT(
        latent_dim=args.latent_dim,
        num_tokens=args.latent_grid ** 2,
        model_dim=args.model_dim,
        spatial_depth=args.spatial_depth,
        temporal_depth=args.temporal_depth,
        num_heads=args.num_heads,
        seq_len=args.seq_len,
        text_cond=False,
        i0_condition=True,
        time_scale=args.time_scale,
    ).to(device=device)
    model.i0_residual = True
    flow = OTCFM(model)
    ema = EMA(model, decay=args.ema_decay, dtype=torch.float32).to(device)
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
        validate_resume(checkpoint, cache_stats, args)
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        ema = ema.to(device)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        global_step = checkpoint["global_step"]
        if rank == 0:
            restore_rng_state(checkpoint.get("rng_state"))
        else:
            set_seed(args.seed + rank + global_step * 1009)

    if use_ddp:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank)
        flow.model = model

    writer = (
        SummaryWriter(
            log_dir=os.path.join(args.output_dir, "tb"),
            purge_step=global_step if global_step > 0 else None,
        )
        if is_main_process() else None
    )
    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    if is_main_process():
        total_params = sum(
            parameter.numel()
            for parameter in (model.module if use_ddp else model).parameters())
        print(
            f"Training cached CompactLatentDiT: {total_params / 1e6:.1f}M params, "
            f"{cache_stats.get('num_samples', 'unknown')} cached videos, "
            f"{world_size} {device_type.upper()} devices")

    data_iterator = iter(dataloader)
    optimizer.zero_grad(set_to_none=True)
    last_preview_step = -1
    while global_step < args.max_steps:
        accumulated_loss = torch.zeros((), device=device)
        for micro_step in range(args.accum_steps):
            batch = next(data_iterator)
            target_raw = batch["target"].to(
                device=device, dtype=torch.float32, non_blocking=True)
            cond_raw = batch["cond"].to(
                device=device, dtype=torch.float32, non_blocking=True)
            if target_raw.shape[1:] != (
                    args.seq_len, args.latent_grid ** 2, args.latent_dim):
                raise ValueError(
                    f"Unexpected target shape {tuple(target_raw.shape)}")

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

        mean_loss = accumulated_loss / args.accum_steps
        if use_ddp:
            dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
            mean_loss /= world_size

        if is_main_process() and global_step % args.log_every == 0:
            loss_value = mean_loss.item()
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"step={global_step} loss={loss_value:.6f} "
                f"lr={lr:.3e} grad_norm={float(grad_norm):.3f}")
            writer.add_scalar("train/loss", loss_value, global_step)
            writer.add_scalar("train/lr", lr, global_step)
            writer.add_scalar(
                "train/grad_norm", float(grad_norm), global_step)
            append_metrics(metrics_path, {
                "step": global_step,
                "train/loss": loss_value,
                "train/lr": lr,
                "train/grad_norm": float(grad_norm),
            })

        save_due = global_step % args.save_every == 0
        eval_due = global_step % args.eval_every == 0
        if is_main_process() and save_due:
            save_checkpoint(
                os.path.join(
                    args.output_dir, f"checkpoint_step{global_step:08d}.pt"),
                base_model, ema, optimizer, scheduler, scaler, global_step,
                cache_stats, args)
            print(f"Saved checkpoint at step {global_step}")

        should_eval = save_due or eval_due
        if use_ddp and should_eval:
            dist.barrier()
        if is_main_process() and should_eval:
            training_state = {
                key: value.clone()
                for key, value in base_model.state_dict().items()}
            try:
                base_model.load_state_dict(ema.state_dict())
                eval_metrics = evaluate_preview(
                    base_model, preview_batch,
                    target_mean, target_std, cond_mean, cond_std,
                    device, model_dtype, args, global_step,
                    device_type,
                    rgb_preview=rgb_preview)
                for name, value in eval_metrics.items():
                    writer.add_scalar(
                        f"eval/{name}", value, global_step)
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
                base_model.load_state_dict(training_state)
                del training_state
        if use_ddp and should_eval:
            dist.barrier()

    if is_main_process():
        save_checkpoint(
            os.path.join(args.output_dir, "checkpoint_final.pt"),
            model.module if use_ddp else model,
            ema, optimizer, scheduler, scaler, global_step,
            cache_stats, args)
        if last_preview_step != global_step:
            base_model = model.module if use_ddp else model
            training_state = {
                key: value.clone()
                for key, value in base_model.state_dict().items()}
            try:
                base_model.load_state_dict(ema.state_dict())
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
                base_model.load_state_dict(training_state)
        writer.close()

    if use_ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
