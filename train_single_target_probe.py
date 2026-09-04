#!/usr/bin/env python3
"""Quickly test whether one R7 target frame is learnable from a clean anchor."""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.loader_utils import multiprocessing_loader_kwargs
from data.video_dataset import SpatialVidDataset, collate_fn
from models.single_target_generator import (
    SingleTargetFlowGenerator, SingleTargetGenerator, euclidean_flow_sample)
from streamvggt.models.streamvggt import StreamVGGT
from train_causal_dual_tokenizer import get_lpips, lpips_chunked
from utils.device import (configure_backend_compatibility, empty_cache,
                          get_device, get_device_name, manual_seed_all,
                          resolve_dtype)
from utils.encoder_loader import load_encoder_checkpoint
from utils.file_signature import validate_file_signature
from utils.r7_representation import (
    build_contract, encode_dual, load_checkpoint, load_r7_modules,
    validate_contract)
from utils.training import append_metrics, atomic_torch_save, build_optimizer
from utils.video_preview import save_video_preview


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--eval_csv", required=True)
    parser.add_argument("--video_root", required=True)
    parser.add_argument("--encoder_ckpt", required=True)
    parser.add_argument("--r7_ckpt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", choices=("deterministic", "flow"),
                        default="deterministic")
    parser.add_argument("--target_index", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=1)
    parser.add_argument("--eval_samples", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--wd", type=float, default=1e-2)
    parser.add_argument("--lambda_latent", type=float, default=1.0)
    parser.add_argument("--lambda_rgb", type=float, default=0.0)
    parser.add_argument("--lambda_lpips", type=float, default=0.0)
    parser.add_argument("--eval_lpips", action="store_true",
                        help="compute LPIPS during eval without training on it")
    parser.add_argument("--sample_steps", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    manual_seed_all(seed)


def psnr(prediction, target):
    error = F.mse_loss(prediction.float(), target.float()).clamp_min(1e-12)
    return float(-10 * torch.log10(error))


def cosine_delta(anchor, prediction, target):
    predicted_delta = (prediction.float() - anchor.float()).flatten(1)
    target_delta = (target.float() - anchor.float()).flatten(1)
    return float(F.cosine_similarity(predicted_delta, target_delta, dim=1).mean())


class FrozenR7:
    def __init__(self, encoder_ckpt, r7_ckpt, device, dtype):
        checkpoint = load_checkpoint(r7_ckpt)
        config, compressor, tex_encoder, tokenizer, decoder, matched = \
            load_r7_modules(checkpoint)
        contract = checkpoint.get("representation_contract")
        if not contract:
            raise ValueError("R7 checkpoint has no representation_contract")
        validate_contract(
            contract, build_contract(config, include_signatures=False))
        expected_encoder = contract.get("signatures", {}).get("streamvggt")
        if expected_encoder:
            validate_file_signature(
                encoder_ckpt, expected_encoder, "StreamVGGT checkpoint")
        if config.temporal_factor != 1 or config.latent_seq_len != config.seq_len:
            raise ValueError(
                "single-target quick probe requires a frame-aligned factor-1 R7")
        if config.decoder_temporal_blocks != 0:
            raise ValueError(
                "single-target v1 requires a framewise RGB decoder; temporal "
                "decoder attention would make a two-frame decode differ from "
                "the full-sequence reconstruction contract")
        encoder = StreamVGGT(
            img_size=config.target_size, patch_size=14, embed_dim=1024)
        load_encoder_checkpoint(encoder, encoder_ckpt, verbose=True)
        self.config = config
        self.representation_contract = contract
        del checkpoint
        self.matched = matched
        self.encoder_dtype = dtype
        self.encoder = encoder.to(device=device, dtype=dtype).eval()
        self.compressor = compressor.to(device).eval()
        self.tex_encoder = tex_encoder.to(device).eval()
        self.tokenizer = tokenizer.to(device).eval()
        self.decoder = decoder.to(device).eval()
        for module in (self.encoder, self.compressor, self.tex_encoder,
                       self.tokenizer, self.decoder):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    @torch.no_grad()
    def encode(self, frames):
        geo, tex = encode_dual(
            self.encoder, self.compressor, self.tex_encoder,
            frames, self.encoder_dtype)
        latent = self.tokenizer.encode(geo, tex)
        return latent.reshape(
            frames.shape[0], self.config.latent_seq_len,
            self.config.latent_grid ** 2, self.config.latent_dim)

    def decode(self, anchor, target):
        """Decode target with its clean anchor as the causal prefix."""
        if anchor.shape != target.shape or target.ndim != 3:
            raise ValueError("anchor/target must share [B,N,D] shape")
        b, _, d = target.shape
        grid = self.config.latent_grid
        sequence = torch.stack((anchor, target), dim=1).reshape(
            b, 2, grid, grid, d)
        # Optional RGB losses backpropagate through these frozen modules into the
        # generator. The anchor prefix prevents zero-context temporal decoding.
        geo, tex = self.tokenizer.decode(sequence)
        return self.decoder(geo, tex)[:, 1:2, ..., :3].float().clamp(0, 1)


def build_loader(csv, video_root, config, samples, batch_size, workers,
                 shuffle):
    dataset = SpatialVidDataset(
        csv, video_root, seq_len=config.seq_len,
        num_frames_per_video=config.seq_len, target_size=config.target_size,
        clip_duration_seconds=config.clip_duration_seconds,
        temporal_jitter=False, max_videos=samples)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=workers, collate_fn=collate_fn,
        drop_last=False, **multiprocessing_loader_kwargs(workers))


@torch.no_grad()
def materialize(loader, frozen, target_index, keep_rgb):
    """Encode each selected clip once; training never reruns StreamVGGT."""
    result = []
    for batch in loader:
        frames = batch["frames"].to(next(frozen.decoder.parameters()).device)
        latent = frozen.encode(frames)
        for index in range(frames.shape[0]):
            item = {
                "anchor": latent[index, 0].float().cpu(),
                "target": latent[index, target_index].float().cpu(),
                "video_id": batch["video_id"][index],
            }
            if keep_rgb:
                item["anchor_rgb"] = frames[index, 0].float().clamp(0, 1) \
                    .mul(255).round().to(device="cpu", dtype=torch.uint8)
                item["raw_target"] = frames[index, target_index].float() \
                    .clamp(0, 1).mul(255).round().to(
                        device="cpu", dtype=torch.uint8)
            result.append(item)
    if not result:
        raise RuntimeError("selected dataset yielded no samples")
    return result


def stack_items(items, device):
    return (torch.stack([item["anchor"] for item in items]).to(device),
            torch.stack([item["target"] for item in items]).to(device))


def main(argv=None):
    args = parse_args(argv)
    if args.target_index != 1:
        raise ValueError(
            "quick-probe v1 supports only frame 0 -> frame 1; farther targets "
            "require an explicit intermediate-context contract")
    if args.max_samples not in (1, 16, 256):
        raise ValueError("quick-probe max_samples must be 1, 16, or 256")
    if min(args.max_samples, args.eval_samples, args.max_steps,
           args.batch_size, args.eval_every, args.log_every) < 1:
        raise ValueError("sample/step arguments must be positive")
    seed_all(args.seed)
    device_type = get_device_name()
    configure_backend_compatibility(device_type)
    if device_type == "cpu":
        raise RuntimeError("single-target probe requires an accelerator")
    device = get_device(0)
    compute_dtype = resolve_dtype(args.dtype)
    frozen = FrozenR7(
        args.encoder_ckpt, args.r7_ckpt, device, compute_dtype)
    if not 0 < args.target_index < frozen.config.latent_seq_len:
        raise ValueError(
            f"target_index must be in [1,{frozen.config.latent_seq_len - 1}]")
    train_csv = args.eval_csv if args.max_samples == 1 else args.csv
    train_loader = build_loader(
        train_csv, args.video_root, frozen.config, args.max_samples,
        args.batch_size, args.num_workers, shuffle=args.max_samples > 1)
    eval_loader = build_loader(
        args.eval_csv, args.video_root, frozen.config, args.eval_samples,
        1, 0, shuffle=False)
    num_tokens = frozen.config.latent_grid ** 2
    common = dict(latent_dim=frozen.config.latent_dim,
                  num_tokens=num_tokens, hidden_dim=args.hidden_dim,
                  depth=args.depth,
                  max_target_index=frozen.config.latent_seq_len - 1)
    train_items = materialize(
        train_loader, frozen, args.target_index, keep_rgb=True)
    # One-pair validation must test exact overfit. Larger arms use the held-out
    # eval split and therefore measure generalization rather than memorization.
    eval_items = (train_items if args.max_samples == 1 else materialize(
        eval_loader, frozen, args.target_index, keep_rgb=True))
    del train_loader, eval_loader
    # The quick generator is judged in R7 latent/RGB space. Release the much
    # larger frozen encoder stack before optimization; geometry re-encoding is
    # an offline follow-up after this MVP passes held-out copy-anchor baselines.
    del frozen.encoder, frozen.compressor, frozen.tex_encoder
    empty_cache()

    model = (SingleTargetGenerator(**common) if args.mode == "deterministic"
             else SingleTargetFlowGenerator(**common)).to(device)
    optimizer = build_optimizer(model, args.lr, args.wd)
    os.makedirs(os.path.join(args.output_dir, "samples"), exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    lpips_model = get_lpips(device) \
        if args.lambda_lpips > 0 or args.eval_lpips else None
    order = list(range(len(train_items)))
    rng = random.Random(args.seed)
    cursor = 0
    step = 0
    best = float("inf")

    @torch.no_grad()
    def evaluate():
        model.eval()
        rows = []
        for index, item in enumerate(eval_items):
            anchor, target = stack_items([item], device)
            if args.mode == "deterministic":
                generated = model(anchor, args.target_index)
            else:
                generated = euclidean_flow_sample(
                    model, anchor, args.target_index, target.shape,
                    steps=args.sample_steps,
                    seed=args.seed + 1009 * index)
            copy_rgb = frozen.decode(anchor, anchor)
            target_rgb = frozen.decode(anchor, target)
            generated_rgb = frozen.decode(anchor, generated)
            raw_target = item["raw_target"].permute(1, 2, 0)[None, None] \
                .to(device=device, dtype=torch.float32).div(255)
            anchor_rgb = item["anchor_rgb"].permute(1, 2, 0)[None, None] \
                .to(device=device, dtype=torch.float32).div(255)
            row = {
                "latent_mse": float(F.mse_loss(generated.float(), target.float())),
                "latent_norm_ratio": float(generated.float().norm() /
                                           target.float().norm().clamp_min(1e-8)),
                "motion_cosine": cosine_delta(anchor, generated, target),
                "copy_psnr_raw": psnr(copy_rgb, raw_target),
                "generated_psnr_raw": psnr(generated_rgb, raw_target),
                "ae_psnr_raw": psnr(target_rgb, raw_target),
                "generated_psnr_ae": psnr(generated_rgb, target_rgb),
            }
            if lpips_model is not None:
                row["generated_lpips_raw"] = float(lpips_chunked(
                    lpips_model, generated_rgb, raw_target, 1, 256))
                row["copy_lpips_raw"] = float(lpips_chunked(
                    lpips_model, copy_rgb, raw_target, 1, 256))
                row["ae_lpips_raw"] = float(lpips_chunked(
                    lpips_model, target_rgb, raw_target, 1, 256))
            rows.append(row)
            if index < 4:
                save_video_preview(
                    os.path.join(args.output_dir, "samples"),
                    f"step{step:07d}_eval{index:02d}",
                    {"ANCHOR": anchor_rgb[0], "RAW_TARGET": raw_target[0],
                     "AE_TARGET": target_rgb[0], "COPY_ANCHOR": copy_rgb[0],
                     "GENERATED": generated_rgb[0]}, fps=1,
                    metadata={"step": step, "mode": args.mode,
                              "target_index": args.target_index,
                              "video_id": item["video_id"]},
                    save_frames=index == 0, save_mp4=False)
        model.train()
        keys = set.intersection(*(set(row) for row in rows))
        result = {f"eval/{key}": float(np.mean([row[key] for row in rows]))
                  for key in keys}
        result.update({"step": step, "eval/samples": len(rows),
                       "eval/split": "train-overfit" if args.max_samples == 1
                                     else "held-out"})
        return result

    while step < args.max_steps:
        if cursor == 0:
            rng.shuffle(order)
        selected = [train_items[i] for i in order[cursor:cursor + args.batch_size]]
        cursor += len(selected)
        if cursor >= len(order):
            cursor = 0
        anchor, target = stack_items(selected, device)
        if args.mode == "deterministic":
            prediction = model(anchor, args.target_index)
        else:
            noise = torch.randn_like(target)
            t = torch.rand(target.shape[0], device=device)
            te = t[:, None, None]
            noisy = (1 - te) * noise + te * target
            prediction = model(noisy, anchor, args.target_index, t)
        latent_loss = F.mse_loss(prediction.float(), target.float())
        # RGB losses are intentionally optional and only practical for 1/16-pair
        # probes. They use raw RGB, not an autoencoder target.
        rgb_loss = latent_loss.new_zeros(())
        perceptual = latent_loss.new_zeros(())
        if args.lambda_rgb > 0 or args.lambda_lpips > 0:
            prediction_rgb = frozen.decode(anchor, prediction)
            raw_target = torch.stack([
                item["raw_target"].permute(1, 2, 0) for item in selected
            ]).to(device=device, dtype=torch.float32).div(255)[:, None]
            if args.lambda_rgb > 0:
                rgb_loss = F.l1_loss(prediction_rgb, raw_target)
            if args.lambda_lpips > 0:
                perceptual = lpips_chunked(
                    lpips_model, prediction_rgb, raw_target, 1, 256)
        loss = (args.lambda_latent * latent_loss
                + args.lambda_rgb * rgb_loss
                + args.lambda_lpips * perceptual)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1
        if step % args.log_every == 0 or step == 1:
            append_metrics(metrics_path, {
                "step": step, "train/loss": float(loss.detach()),
                "train/latent": float(latent_loss.detach()),
                "train/rgb_l1": float(rgb_loss.detach()),
                "train/lpips": float(perceptual.detach()),
                "train/grad_norm": float(grad)})
        if step % args.eval_every == 0 or step == args.max_steps:
            row = evaluate()
            append_metrics(metrics_path, row)
            score = row["eval/latent_mse"]
            payload = {
                "schema": "r7-single-target-probe-v1",
                "model": model.state_dict(), "step": step,
                "mode": args.mode, "args": vars(args),
                "representation": frozen.representation_contract,
                "metrics": row,
            }
            atomic_torch_save(
                payload, os.path.join(args.output_dir, "checkpoint_latest.pt"))
            if score < best:
                best = score
                atomic_torch_save(
                    payload, os.path.join(args.output_dir, "checkpoint_best.pt"))
            print(row)
    print(f"single-target quick probe finished: {args.output_dir}")


if __name__ == "__main__":
    main()
