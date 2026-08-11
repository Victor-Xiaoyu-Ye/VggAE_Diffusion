#!/usr/bin/env python3
"""R7 Wan2.1-T2V-1.3B full-parameter diffusion training from cached tar shards.

Replaces the from-scratch CompactLatentDiT of ``train_causal_video_diffusion``
with a fully finetuned Wan2.1-T2V-1.3B trunk (``WanCompactAdapter`` in
``full_finetune`` + ``anchor_frame`` mode) and switches the objective to
x0/clean-target prediction with flow-matching sampling (VGGT-World: velocity
prediction collapses in high-dimensional frozen-feature spaces).

Text conditioning uses precomputed UMT5-xxl embeddings keyed by ``video_id``
(``precompute_wan_text_embeddings.py``); missing captions and CFG dropout fall
back to the empty-prompt embedding, matching Wan's own null-prompt convention.

The cache/normalization/eval/checkpoint surface deliberately mirrors
``train_causal_video_diffusion.py``; shared helpers are imported from it.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.latent_shard_dataset import LatentShardDataset, latent_collate_fn
from data.loader_utils import multiprocessing_loader_kwargs
from models.wan_compact_adapter import WanCompactAdapter
from train_causal_video_diffusion import (
    CONTEXT_CHUNKS, FUTURE_CHUNKS, LATENT_DIM, assert_same_representation,
    auxiliary_losses, aux_scale, build_extension_scheduler,
    checked_decode_chunk_size, composite_score, decode_rgb_metrics, ema_weights,
    exact_equal, expanded_geo_motion_metrics, gather_rng, latent_metrics,
    load_eval_batches, load_r7_stack, load_torch_artifact, _mean,
    normalize_fp32, set_seed, shift_time, stats_tensors,
    validate_batch, validate_r7_artifact_representation, validate_stats)
from utils.device import (configure_backend_compatibility, create_grad_scaler,
                          get_device, get_device_name, resolve_dtype)
from utils.distributed import is_main_process, setup_ddp
from utils.training import (EMA, ThroughputMeter, append_metrics,
                            atomic_torch_save, build_scheduler,
                            count_latent_tokens, restore_rng_state)

X0_MSE_BUCKETS = (0.1, 0.3, 0.5, 0.7, 0.9)
STRICT_MODEL_ARGS = ("latent_dim", "latent_grid", "time_shift_alpha",
                     "normalization_mode")
STRICT_TRAIN_ARGS = ("lambda_motion", "lambda_accel", "lambda_geo_motion",
                     "aux_warmup_steps", "aux_ramp_steps", "aux_t_min",
                     "batch_size", "accum_steps", "adapter_lr", "wan_lr", "wd",
                     "warmup_steps", "wan_freeze_steps", "ema_decay",
                     "max_grad_norm", "dtype", "require_rgb_lpips",
                     "text_drop_prob")
OBJECTIVE_SCHEMA = "r7-wan13b-t2v-x0-fm-v1"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="R7 Wan2.1-T2V-1.3B full-finetune diffusion")
    p.add_argument("--manifest", required=True)
    p.add_argument("--stats", required=True)
    p.add_argument("--eval_manifest", required=True)
    p.add_argument("--eval_stats", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--resume", default="")
    p.add_argument("--wan_ckpt_dir", required=True,
                   help="Wan2.1-T2V-1.3B checkpoint directory")
    p.add_argument("--r7_ckpt", default="",
                   help="accepted R7 checkpoint (must match cache signatures.r7)")
    p.add_argument("--decoder_ckpt", default="",
                   help="decoder_robust checkpoint; only its decoder.* weights "
                        "override the R7 decoder for eval-time RGB decode")
    p.add_argument("--text_embedding_dir", required=True,
                   help="precomputed UMT5-xxl sidecar (index.json + shards)")
    p.add_argument("--latent_dim", type=int, default=LATENT_DIM)
    p.add_argument("--latent_grid", type=int, default=18)
    p.add_argument("--context_chunks", type=int, default=CONTEXT_CHUNKS)
    p.add_argument("--future_chunks", type=int, default=FUTURE_CHUNKS)
    p.add_argument("--time_shift_alpha", type=float, default=3.0,
                   help="RAE high-dimensional time shift (identity at 1.0)")
    p.add_argument("--normalization_mode", choices=("zscore", "none"),
                   default="zscore")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--accum_steps", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=30000)
    p.add_argument("--adapter_lr", type=float, default=1e-4)
    p.add_argument("--wan_lr", type=float, default=1e-5)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--wan_freeze_steps", type=int, default=500,
                   help="drop trunk grads until the fresh adapters settle")
    p.add_argument("--extension_lr", type=float, default=0.0,
                   help="required peak LR when extending a completed run")
    p.add_argument("--extension_warmup_steps", type=int, default=200)
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="bf16")
    p.add_argument("--text_drop_prob", type=float, default=0.1,
                   help="CFG dropout to the empty-prompt embedding")
    p.add_argument("--cfg_scale", type=float, default=3.0,
                   help="classifier-free guidance scale for eval sampling")
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
    p.add_argument("--decode_chunk_size", type=int, default=0)
    p.add_argument("--require_rgb_lpips", action="store_true")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--early_stop_min_steps", type=int, default=10000)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--local_rank", type=int, default=0)
    return p.parse_args(argv)


class TextEmbeddingBank:
    """In-RAM lookup of precomputed UMT5-xxl embeddings by video id.

    Embeddings are stored fp16 and trimmed to true token length, so the whole
    10k-caption bank stays in the low-GB range per rank. The bank lives in the
    training process only (lookup happens after collate), so spawn DataLoader
    workers never copy it.
    """

    def __init__(self, sidecar_dir):
        self.dir = sidecar_dir

        def load(name):
            path = self._join(name)
            if name.endswith(".json") or name == "_SUCCESS":
                from utils.moxing_io import is_remote_path, read_text
                if is_remote_path(path):
                    return json.loads(read_text(path))
                with open(path) as handle:
                    return json.load(handle)
            return load_torch_artifact(path)

        success = load("_SUCCESS")
        if success.get("schema") != "wan-umt5xxl-text-embeddings-v1":
            raise ValueError(
                f"unsupported text-embedding sidecar schema in {sidecar_dir}")
        self.text_len = int(success["text_len"])
        index = load("index.json")
        self.empty = load("empty_prompt.pt").float()
        self.embeddings = {}
        for shard_name in sorted(set(index.values())):
            shard = load(shard_name)
            for video_id, tensor in shard.items():
                self.embeddings[video_id] = tensor
        missing = set(index) - set(self.embeddings)
        if missing:
            raise ValueError(
                f"text-embedding index references missing ids: "
                f"{sorted(missing)[:8]}")
        self.hits = self.misses = 0

    def _join(self, name):
        from utils.moxing_io import is_remote_path, join_remote
        if is_remote_path(self.dir):
            return join_remote(self.dir, name)
        return os.path.join(self.dir, name)

    def lookup(self, video_id):
        tensor = self.embeddings.get(video_id)
        if tensor is None:
            self.misses += 1
            return self.empty
        self.hits += 1
        return tensor.float()

    def batch(self, video_ids, device, drop_mask=None):
        """[B, L_max, 4096] zero-padded batch (Wan pads context with zeros)."""
        chosen = []
        for index, video_id in enumerate(video_ids):
            if drop_mask is not None and bool(drop_mask[index]):
                chosen.append(self.empty)
            else:
                chosen.append(self.lookup(video_id))
        longest = max(tensor.shape[0] for tensor in chosen)
        batch = torch.zeros(len(chosen), longest, chosen[0].shape[-1],
                            dtype=torch.float32)
        for index, tensor in enumerate(chosen):
            batch[index, :tensor.shape[0]] = tensor
        return batch.to(device)

    def empty_batch(self, count, device):
        return self.batch([""] * count, device,
                          drop_mask=[True] * count)


def wan_config_snapshot(model):
    return {"dim": int(model.wan_dim), "freq_dim": int(model.freq_dim),
            "num_heads": int(model.num_heads),
            "num_layers": len(model.wan.blocks),
            "ffn_dim": int(getattr(model.wan, "ffn_dim")),
            "text_dim": int(getattr(model.wan, "text_dim")),
            "model_type": str(getattr(model.wan, "model_type"))}


def architecture_contract(args, wan_config):
    return {"class": "WanCompactAdapter", "full_finetune": True,
            "anchor_frame": True, "i0_condition": False,
            "latent_dim": int(args.latent_dim),
            "latent_grid": int(args.latent_grid),
            "seq_len": FUTURE_CHUNKS, "text_cond": True,
            "wan_config": dict(wan_config)}


def objective_contract(args):
    return {
        "schema": OBJECTIVE_SCHEMA,
        "prediction": "x0",
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
        "text_drop_prob": float(args.text_drop_prob),
        "optimizer": {
            "name": "AdamW", "adapter_lr": float(args.adapter_lr),
            "wan_lr": float(args.wan_lr), "weight_decay": float(args.wd),
            "betas": [0.9, 0.95], "eps": 1e-8,
            "wan_freeze_steps": int(args.wan_freeze_steps),
        },
    }


def build_model(args):
    return WanCompactAdapter(
        args.wan_ckpt_dir, latent_dim=args.latent_dim,
        latent_grid=args.latent_grid, seq_len=FUTURE_CHUNKS,
        full_finetune=True, anchor_frame=True)


def build_wan_optimizer(model, args):
    """Two-speed AdamW (train_dual_diffusion precedent): fresh adapter layers
    at --adapter_lr, pretrained trunk at --wan_lr."""
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        family = "wan" if name.startswith("wan.") else "adapter"
        decay = not (parameter.ndim < 2 or "norm" in name or "bias" in name
                     or "modulation" in name)
        groups.setdefault((family, decay), []).append(parameter)
    param_groups = []
    for (family, decay), params in sorted(groups.items()):
        param_groups.append({
            "params": params,
            "lr": args.adapter_lr if family == "adapter" else args.wan_lr,
            "weight_decay": args.wd if decay else 0.0,
            "name": family})
    return torch.optim.AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8)


def flow_forward_x0(model, x1, cond, text_emb, alpha):
    x0 = torch.randn_like(x1)
    t = shift_time(torch.rand(x1.shape[0], device=x1.device), alpha).to(x1.dtype)
    te = t.view(-1, 1, 1, 1)
    xt = (1 - te) * x0 + te * x1
    x1_hat = model(xt, t, cond=cond, text_emb=text_emb)
    return F.mse_loss(x1_hat.float(), x1.float()), {"t": t, "x1_pred": x1_hat}


def sample_x0_shifted(model, cond, shape, steps, alpha, noise, text_emb,
                      uncond_emb=None, cfg_scale=1.0, device_type="cpu",
                      compute_dtype=torch.float32):
    """Euler flow-matching sampling from an x0-predicting model.

    The velocity is recovered as v = (x1_hat - z_t) / (1 - t); the final step
    lands exactly on x1_hat because dt equals the remaining 1 - t.
    """
    model_dtype = cond.dtype
    z = (noise if noise is not None else torch.randn(
        shape, device=cond.device, dtype=model_dtype)).float()
    grid = torch.linspace(0, 1, steps + 1, device=cond.device,
                          dtype=torch.float32)
    shifted = shift_time(grid, alpha)
    use_cfg = uncond_emb is not None and cfg_scale != 1.0
    for i in range(steps):
        t_val = float(shifted[i])
        t = torch.full((shape[0],), t_val, device=cond.device,
                       dtype=model_dtype)
        with torch.autocast(
                device_type=device_type, dtype=compute_dtype,
                enabled=compute_dtype != torch.float32):
            x1_hat = model(z.to(model_dtype), t, cond=cond,
                           text_emb=text_emb).float()
            if use_cfg:
                x1_uncond = model(z.to(model_dtype), t, cond=cond,
                                  text_emb=uncond_emb).float()
                x1_hat = x1_uncond + cfg_scale * (x1_hat - x1_uncond)
        velocity = (x1_hat - z) / max(1.0 - t_val, 1e-6)
        z = z + velocity * (float(shifted[i + 1]) - t_val)
    return z


def load_robust_decoder(decoder_ckpt, stats_representation, decoder):
    """Override only decoder.* weights from a decoder_robust checkpoint.

    The cache binds signatures.r7 to the accepted joint checkpoint bytes, so a
    retrained decoder cannot be passed as --r7_ckpt; it rides a separate flag
    and must share the exact representation config/layout and frozen-source
    signatures with the cache.
    """
    from utils.r7_representation import (
        load_checkpoint, model_state, validate_contract)
    artifact = load_checkpoint(decoder_ckpt)
    contract = artifact.get("representation_contract")
    if not isinstance(contract, dict):
        raise ValueError("--decoder_ckpt must contain representation_contract")
    validate_contract(stats_representation, contract)
    phase = str((artifact.get("args") or {}).get("phase", ""))
    if phase != "decoder_robust":
        raise ValueError(
            f"--decoder_ckpt must come from PHASE=decoder_robust, got "
            f"{phase or 'unknown'!r}")
    from utils.r7_representation import load_prefixed_strict
    matched = load_prefixed_strict(decoder, model_state(artifact), "decoder.")
    print(f"robust decoder override: {matched} decoder.* tensors from "
          f"{os.path.basename(decoder_ckpt)}")


def strict_resume(checkpoint, args, stats, signature, world_size, wan_config):
    saved = checkpoint.get("args", {})
    saved_max = int(saved.get("max_steps", 1 << 60))
    saved_step = int(checkpoint.get("global_step", -1))
    if args.max_steps != saved_max:
        if saved_step < saved_max:
            raise ValueError(
                "max_steps may change only after the saved run completed")
        if (args.max_steps <= saved_step or args.max_steps < saved_max
                or args.extension_lr <= 0):
            raise ValueError(
                "extension requires a completed saved run, a larger target, "
                "and positive --extension_lr")
    for key in STRICT_MODEL_ARGS + STRICT_TRAIN_ARGS:
        if saved.get(key) != getattr(args, key):
            raise ValueError(
                f"resume mismatch {key}: {saved.get(key)!r} != "
                f"{getattr(args, key)!r}")
    saved_effective = int(saved.get("batch_size", -1)) * \
        int(saved.get("accum_steps", -1)) * \
        int(checkpoint.get("world_size", world_size))
    if saved_effective != args.batch_size * args.accum_steps * world_size:
        raise ValueError("resume effective batch mismatch")
    for key, expected in (("context_chunks", 1), ("future_chunks", 4)):
        if int(saved.get(key, -1)) != expected:
            raise ValueError(f"resume {key} mismatch")
    if not exact_equal(checkpoint.get("architecture"),
                       architecture_contract(args, wan_config)):
        raise ValueError("resume architecture contract is not exactly identical")
    if not exact_equal(checkpoint.get("objective"), objective_contract(args)):
        raise ValueError("resume objective contract is not exactly identical")
    if checkpoint.get("normalization_signature") != signature or not exact_equal(
            checkpoint.get("normalization"), stats):
        raise ValueError("resume normalization/representation is not identical")


def checkpoint_payload(core, ema, optimizer, scheduler, scaler, step, args,
                       stats, signature, rng_by_rank, best, world_size,
                       wan_config):
    return {"checkpoint_version": 4, "model": core.state_dict(),
            "ema": ema.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "global_step": step, "world_size": world_size,
            "rng_by_rank": rng_by_rank, "args": vars(args),
            "architecture": architecture_contract(args, wan_config),
            "objective": objective_contract(args),
            "wan_config": dict(wan_config),
            "representation": stats["representation"],
            "normalization": stats, "normalization_signature": signature,
            "best": best,
            "production_mode": args.normalization_mode == "zscore"}


def main(argv=None):
    args = parse_args(argv)
    if (args.context_chunks, args.future_chunks, args.latent_dim) != (1, 4, 192):
        raise ValueError("production R7 contract is strictly context=1, future=4, D=192")
    if args.require_rgb_lpips and not args.r7_ckpt:
        raise ValueError("--require_rgb_lpips requires --r7_ckpt")
    if args.normalization_mode == "none":
        print("[WARN] normalization=none is diagnostic only; checkpoint is non-production")
    for value in (args.accum_steps, args.eval_every, args.save_every,
                  args.sample_steps, args.eval_clips, args.early_stop_min_steps):
        if value < 1:
            raise ValueError("step/count arguments must be positive")
    if args.warmup_steps >= args.max_steps:
        raise ValueError("warmup must be < max_steps")
    if not 0.0 <= args.text_drop_prob < 1.0:
        raise ValueError("--text_drop_prob must be in [0, 1)")
    if any(v < 0 for v in (args.lambda_motion, args.lambda_accel,
                            args.lambda_geo_motion, args.extension_lr)):
        raise ValueError("loss weights and extension_lr must be nonnegative")
    if "14B" in os.path.basename(os.path.normpath(args.wan_ckpt_dir)):
        raise ValueError(
            "this stage is restricted to Wan2.1-T2V-1.3B; 14B trunks OOM "
            "under replicated DDP")

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

    text_bank = TextEmbeddingBank(args.text_embedding_dir)

    dataset = LatentShardDataset(args.manifest, args.shuffle_buffer, args.seed,
                                 True, rank, world_size)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=latent_collate_fn, pin_memory=device_type == "cuda",
        drop_last=True, **multiprocessing_loader_kwargs(args.num_workers))
    eval_batches = load_eval_batches(args.eval_manifest, args.eval_clips, args) \
        if main_process else None

    core = build_model(args).to(device=device, dtype=torch.float32)
    wan_config = wan_config_snapshot(core)
    # FP32 master weights/Adam/EMA; autocast (fp16 on NPU) for the trunk
    # forward. Normalization, integration state, and losses remain FP32.
    compute_dtype = resolve_dtype(args.dtype)
    model_dtype = torch.float32
    use_scaler = compute_dtype == torch.float16
    ema = EMA(core, args.ema_decay, dtype=torch.float32).to(device)
    optimizer = build_wan_optimizer(core, args)
    scheduler = build_scheduler(optimizer, args.warmup_steps, args.max_steps)
    scaler = create_grad_scaler(enabled=use_scaler)
    step = 0; best = {"score": float("inf"), "step": 0, "bad_evals": 0,
                      "source": "none", "guarded": True}
    if args.resume:
        checkpoint = load_torch_artifact(args.resume)
        strict_resume(checkpoint, args, stats, signature, world_size, wan_config)
        core.load_state_dict(checkpoint["model"], strict=True)
        ema.load_state_dict(checkpoint["ema"]); ema.to(device)
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint: scaler.load_state_dict(checkpoint["scaler"])
        step = int(checkpoint["global_step"]); best = checkpoint.get("best", best)
        saved_max = int(checkpoint["args"]["max_steps"])
        if args.max_steps != saved_max:
            scheduler = build_extension_scheduler(
                optimizer, [g["lr"] for g in optimizer.param_groups],
                args.extension_lr, args.extension_warmup_steps,
                args.max_steps - step)
        else:
            scheduler.load_state_dict(checkpoint["scheduler"])
        states = checkpoint.get("rng_by_rank") or []
        if rank < len(states) and states[rank] is not None:
            restore_rng_state(states[rank])
        else:
            set_seed(args.seed + rank + step * 1009)

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
    if args.decoder_ckpt and decoder is not None and main_process:
        load_robust_decoder(args.decoder_ckpt, stats["representation"], decoder)
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
    writer = SummaryWriter(os.path.join(args.output_dir, "tb"),
                           purge_step=step or None) if main_process else None
    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")

    @torch.no_grad()
    def evaluate():
        nonlocal decode_enabled
        core.eval(); aggregate = []; packs = []
        for index, cpu_batch in enumerate(eval_batches):
            cond_raw, target_raw = validate_batch(cpu_batch, args)
            cond_raw = cond_raw.to(device).float()
            target_raw = target_raw.to(device).float()
            cond = normalize_fp32(cond_raw, st["cond"],
                                  args.normalization_mode).to(model_dtype)
            target = normalize_fp32(target_raw, st["target"],
                                    args.normalization_mode).to(model_dtype)
            video_ids = cpu_batch.get("video_id", [""])
            text_emb = text_bank.batch(video_ids, device)
            uncond_emb = text_bank.empty_batch(len(video_ids), device)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(args.seed + 1009 * index)
            noise = torch.randn(target.shape, generator=generator).to(
                device, model_dtype)
            row = {}
            for bucket in X0_MSE_BUCKETS:
                t = torch.full((target.shape[0],), bucket, device=device,
                               dtype=model_dtype)
                te = t.view(-1, 1, 1, 1)
                with torch.autocast(
                        device_type=device_type, dtype=compute_dtype,
                        enabled=compute_dtype != torch.float32):
                    x1_hat = core((1-te)*noise + te*target, t, cond=cond,
                                  text_emb=text_emb)
                row[f"eval/x0_mse_t{bucket:.1f}"] = F.mse_loss(
                    x1_hat.float(), target.float()).item()
            row["eval/x0_mse"] = _mean([row[k] for k in row])
            generated = sample_x0_shifted(
                core, cond, target.shape, args.sample_steps,
                args.time_shift_alpha, noise, text_emb, uncond_emb,
                args.cfg_scale, device_type, compute_dtype)
            latent_row, gen_raw = latent_metrics(
                generated, target_raw, cond_raw, st["target"],
                args.normalization_mode)
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
                    print(f"[WARN] RGB decode disabled after failure: {exc}")
            aggregate.append(row)
            if index < args.sample_clips:
                packs.append({"cond_anchor": cond_raw.cpu(),
                              "target_future": target_raw.cpu(),
                              "sampled_future": gen_raw.cpu(),
                              "sampled_normalized": generated.float().cpu(),
                              "rgb_generated_9f": rgb_generated,
                              "rgb_ae_target_9f": rgb_target,
                              "video_id": video_ids[0]})
        keys = set.intersection(*(set(row) for row in aggregate))
        result = {key: _mean([row[key] for row in aggregate]) for key in keys}
        if args.require_rgb_lpips and (
                len(aggregate) != len(eval_batches)
                or "eval/rgb_lpips_future_vs_ae_target" not in result):
            raise RuntimeError(
                "production eval did not produce RGB LPIPS for every clip")
        result.update({"step": step, "eval/weights": "ema",
                       "eval/normalization_signature": signature,
                       "eval/decode_available": decode_enabled,
                       "eval/cfg_scale": args.cfg_scale})
        score, source, guarded = composite_score(result)
        result.update({"eval/composite": score, "eval/composite_source": source,
                       "eval/composite_guarded": guarded})
        atomic_torch_save({"schema": "r7-wan-diffusion-samples-v1", "step": step,
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
                                     best, world_size, wan_config)
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

    if main_process:
        print(f"Wan T2V full-finetune: world={world_size} "
              f"text_bank={len(text_bank.embeddings)} captions "
              f"max_steps={args.max_steps}")
    iterator = iter(loader); optimizer.zero_grad(set_to_none=True)
    stop = False; throughput_meter = ThroughputMeter()
    drop_generator = torch.Generator()
    drop_generator.manual_seed(args.seed * 7919 + rank + step)
    while step < args.max_steps and not stop:
        totals = torch.zeros(4, device=device)
        for micro in range(args.accum_steps):
            batch = next(iterator)
            cond_raw, target_raw = validate_batch(batch, args)
            cond_raw = cond_raw.to(device, non_blocking=True).float()
            target_raw = target_raw.to(device, non_blocking=True).float()
            throughput_meter.update(count_latent_tokens(target_raw))
            cond = normalize_fp32(cond_raw, st["cond"],
                                  args.normalization_mode).to(model_dtype)
            target = normalize_fp32(target_raw, st["target"],
                                    args.normalization_mode).to(model_dtype)
            video_ids = batch.get("video_id", [""] * target.shape[0])
            drop_mask = (torch.rand(len(video_ids), generator=drop_generator)
                         < args.text_drop_prob).tolist()
            text_emb = text_bank.batch(video_ids, device, drop_mask)
            sync = model.no_sync() if use_ddp and micro < args.accum_steps-1 \
                else contextlib.nullcontext()
            with sync:
                with torch.autocast(
                        device_type=device_type, dtype=compute_dtype,
                        enabled=compute_dtype != torch.float32):
                    flow_loss, out = flow_forward_x0(
                        model, target, cond, text_emb, args.time_shift_alpha)
                scale = aux_scale(step, args)
                zero = flow_loss.new_zeros(())
                motion = accel = geo = zero
                if scale and (args.lambda_motion or args.lambda_accel
                              or args.lambda_geo_motion):
                    # x0 prediction: x1_pred IS the model output, no velocity
                    # integration needed before the raw-space aux terms.
                    motion, accel, geo = auxiliary_losses(
                        out["x1_pred"], target_raw, cond_raw, st["target"],
                        args.normalization_mode, out["t"], args, tokenizer)
                total = flow_loss + scale * (args.lambda_motion*motion +
                    args.lambda_accel*accel + args.lambda_geo_motion*geo)
                if use_scaler:
                    scaler.scale(total / args.accum_steps).backward()
                else:
                    (total / args.accum_steps).backward()
            totals += torch.stack((total.detach(), motion.detach(),
                                   accel.detach(), geo.detach()))
        if use_scaler: scaler.unscale_(optimizer)
        if step < args.wan_freeze_steps:
            # Adapter warm-up (train_dual_diffusion precedent): grads are
            # DDP-reduced normally, then dropped for the pretrained trunk so
            # the random projections settle before the prior moves.
            for name, parameter in core.named_parameters():
                if name.startswith("wan.") and parameter.grad is not None:
                    parameter.grad = None
        grad = torch.nn.utils.clip_grad_norm_(
            [p for p in core.parameters() if p.requires_grad],
            args.max_grad_norm)
        if use_scaler:
            scaler.step(optimizer); scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True); scheduler.step(); ema.update(core)
        step += 1; totals /= args.accum_steps
        if use_ddp: dist.all_reduce(totals); totals /= world_size
        if main_process and step % args.log_every == 0:
            throughput = throughput_meter.rate()
            row = {"step": step, "train/loss": totals[0].item(),
                   "train/motion_loss": totals[1].item(),
                   "train/accel_loss": totals[2].item(),
                   "train/geo_motion_loss": totals[3].item(),
                   "train/grad_norm": float(grad),
                   "train/lr": optimizer.param_groups[0]["lr"],
                   "DI_throughput": throughput}
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
                                     best, world_size, wan_config)
        atomic_torch_save(payload, os.path.join(args.output_dir,
                                                "checkpoint_latest.pt"))
        writer.close()
    if use_ddp:
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
