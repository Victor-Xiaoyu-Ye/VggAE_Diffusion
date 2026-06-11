#!/usr/bin/env python3
"""Cache I0-conditioned compact latents into streaming tar shards.

Run this as a job array over ``--index_num_shards``. Each array job may itself
use torchrun so StreamVGGT encoding is distributed across local GPUs.
"""

import argparse
import glob
import io
import json
import os
import tarfile

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from data.token_utils import strip_special_tokens
from data.video_dataset import SpatialVidDataset, collate_fn
from models.generative_tokenizer import GenerativeTokenizer
from streamvggt.models.streamvggt import StreamVGGT
from utils.device import get_device, get_device_name, resolve_dtype
from utils.distributed import is_main_process, setup_ddp


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cache compact residual latents for large-scale training")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--video_root", required=True)
    parser.add_argument("--annotation_index", default="")
    parser.add_argument("--encoder_ckpt", required=True)
    parser.add_argument("--autoencoder_ckpt", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--index_shard_id", type=int, default=0)
    parser.add_argument("--index_num_shards", type=int, default=1)
    parser.add_argument("--partition_id", type=int, default=-1)
    parser.add_argument("--num_partitions", type=int, default=0)
    parser.add_argument("--max_videos", type=int, default=0)
    parser.add_argument("--check_files", action="store_true")
    parser.add_argument("--samples_per_tar", type=int, default=512)

    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--latent_grid", type=int, default=18)
    parser.add_argument("--token_dim", type=int, default=2048)
    parser.add_argument("--levels", type=int, nargs="+", default=[4, 11, 17, 23])
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--target_size", type=int, default=518)
    parser.add_argument("--num_frames_per_video", type=int, default=8)
    parser.add_argument("--max_frame_span", type=int, default=32)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--local_rank", type=int, default=0)
    return parser.parse_args()


class TarShardWriter:
    def __init__(self, output_dir, prefix, samples_per_tar):
        self.output_dir = output_dir
        self.prefix = prefix
        self.samples_per_tar = samples_per_tar
        self.shard_index = 0
        self.samples_in_shard = 0
        self.archive = None
        self.paths = []

    def _open_next(self):
        path = os.path.join(
            self.output_dir, f"{self.prefix}-{self.shard_index:06d}.tar")
        if os.path.exists(path):
            raise FileExistsError(
                f"Refusing to overwrite existing cache shard: {path}")
        self.archive = tarfile.open(path, mode="w")
        self.paths.append(path)
        self.samples_in_shard = 0
        self.shard_index += 1

    def write(self, key, sample):
        if self.archive is None or self.samples_in_shard >= self.samples_per_tar:
            self.close_current()
            self._open_next()

        buffer = io.BytesIO()
        torch.save(sample, buffer)
        payload = buffer.getvalue()
        info = tarfile.TarInfo(name=f"{key}.pt")
        info.size = len(payload)
        self.archive.addfile(info, io.BytesIO(payload))
        self.samples_in_shard += 1

    def close_current(self):
        if self.archive is not None:
            self.archive.close()
            self.archive = None

    def close(self):
        self.close_current()


class SafeDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        try:
            return self.dataset[index]
        except Exception as error:
            entry = self.dataset.index[index]
            return {
                "_error": repr(error),
                "video_id": entry.get("video_id", ""),
                "video_path": entry.get("video_path", ""),
            }


def safe_collate_fn(batch):
    errors = [sample for sample in batch if "_error" in sample]
    valid = [sample for sample in batch if "_error" not in sample]
    if not valid:
        return {"frames": None, "errors": errors}
    result = collate_fn(valid)
    result["errors"] = errors
    return result


def load_models(args, device, compute_dtype):
    encoder = StreamVGGT(
        img_size=args.target_size, patch_size=14, embed_dim=1024)
    encoder_state = torch.load(args.encoder_ckpt, map_location="cpu")
    encoder.load_state_dict(encoder_state, strict=False)
    encoder = encoder.to(device=device, dtype=compute_dtype).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    tokenizer = GenerativeTokenizer(
        token_dim=args.token_dim,
        latent_dim=args.latent_dim,
        latent_grid=args.latent_grid,
        levels=args.levels,
        seq_len=args.seq_len,
        input_grid=args.target_size // 14,
    ).to(device=device)
    checkpoint = torch.load(
        args.autoencoder_ckpt, map_location="cpu", weights_only=False)
    tokenizer.load_state_dict(checkpoint["tokenizer"])
    tokenizer.eval()
    for parameter in tokenizer.parameters():
        parameter.requires_grad_(False)
    return encoder, tokenizer


def update_stats(stats, tensor):
    values = tensor.float().reshape(-1, tensor.shape[-1])
    stats["sum"] += values.sum(dim=0)
    stats["sum_sq"] += values.square().sum(dim=0)
    stats["count"] += values.shape[0]


def finalize_stats(stats):
    count = stats["count"].clamp_min(1)
    mean = stats["sum"] / count
    variance = stats["sum_sq"] / count - mean.square()
    return {
        "mean": mean.float().cpu(),
        "std": variance.clamp_min(1e-12).sqrt().float().cpu(),
        "count": int(stats["count"].item()),
    }


def count_failure_records(paths):
    count = 0
    for path in paths:
        with open(path) as failure_file:
            count += sum(1 for line in failure_file if line.strip())
    return count


def file_signature(path):
    stat = os.stat(path)
    return {
        "path": os.path.abspath(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def main():
    args = parse_args()
    use_ddp, rank, local_rank, world_size = setup_ddp()
    device_type = get_device_name()
    if device_type == "cpu":
        raise RuntimeError("Compact latent caching requires an accelerator")
    device = get_device(local_rank)
    compute_dtype = resolve_dtype(args.dtype)

    partition_id = (
        args.partition_id
        if args.partition_id >= 0 else args.index_shard_id)
    num_partitions = (
        args.num_partitions
        if args.num_partitions > 0 else args.index_num_shards)
    if not 0 <= partition_id < num_partitions:
        raise ValueError(
            f"partition_id must be in [0, {num_partitions}), "
            f"got {partition_id}")
    partition_dir = os.path.join(
        args.output_dir,
        f"part-{partition_id:05d}-of-{num_partitions:05d}",
    )
    os.makedirs(partition_dir, exist_ok=True)

    dataset = SpatialVidDataset(
        csv_path=args.csv,
        video_root=args.video_root,
        seq_len=args.seq_len,
        target_size=args.target_size,
        annotation_index_path=args.annotation_index,
        max_videos=args.max_videos,
        num_frames_per_video=args.num_frames_per_video,
        temporal_jitter=False,
        index_shard_id=args.index_shard_id,
        index_num_shards=args.index_num_shards,
        check_files=args.check_files,
        max_frame_span=args.max_frame_span,
    )
    rank_indices = range(rank, len(dataset), world_size)
    rank_dataset = Subset(SafeDataset(dataset), rank_indices)
    dataloader = DataLoader(
        rank_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=safe_collate_fn,
        pin_memory=device_type == "cuda",
        drop_last=False,
    )

    encoder, tokenizer = load_models(args, device, compute_dtype)
    writer = TarShardWriter(
        partition_dir, f"latents-r{rank:04d}", args.samples_per_tar)

    target_stats = {
        "sum": torch.zeros(args.latent_dim, device=device, dtype=torch.float32),
        "sum_sq": torch.zeros(args.latent_dim, device=device, dtype=torch.float32),
        "count": torch.zeros((), device=device, dtype=torch.float32),
    }
    cond_stats = {
        "sum": torch.zeros(args.latent_dim, device=device, dtype=torch.float32),
        "sum_sq": torch.zeros(args.latent_dim, device=device, dtype=torch.float32),
        "count": torch.zeros((), device=device, dtype=torch.float32),
    }

    sample_index = 0
    failed_samples = []
    successful_samples = torch.zeros(
        (), device=device, dtype=torch.float32)
    progress = tqdm(
        dataloader, disable=not is_main_process(), desc="Caching compact latents")
    try:
        with torch.inference_mode():
            for batch in progress:
                failed_samples.extend(batch["errors"])
                if batch["frames"] is None:
                    continue
                frames = batch["frames"].to(
                    device=device, dtype=compute_dtype, non_blocking=True)

                with torch.autocast(
                        device_type=device_type, dtype=compute_dtype,
                        enabled=compute_dtype != torch.float32):
                    video_tokens, video_psi = encoder(frames)
                    video_tokens = strip_special_tokens(
                        video_tokens, video_psi)
                    _, video_flat = tokenizer(video_tokens)

                    i0_tokens, i0_psi = encoder(frames[:, :1])
                    i0_tokens = strip_special_tokens(i0_tokens, i0_psi)
                    _, i0_flat = tokenizer(i0_tokens)

                # Frame 0 is observed, so diffusion only models future residuals.
                target = (
                    video_flat[:, 1:]
                    - i0_flat.expand(-1, video_flat.shape[1] - 1, -1, -1)
                )

                update_stats(target_stats, target)
                update_stats(cond_stats, i0_flat)

                for batch_index in range(target.shape[0]):
                    key = (
                        f"p{partition_id:05d}-r{rank:04d}-"
                        f"{sample_index:09d}")
                    writer.write(key, {
                        "target": target[batch_index].to(
                            device="cpu", dtype=torch.float16),
                        "cond": i0_flat[batch_index].to(
                            device="cpu", dtype=torch.float16),
                        "caption": batch["caption"][batch_index],
                        "video_id": batch["video_id"][batch_index],
                    })
                    sample_index += 1
                    successful_samples += 1
    finally:
        writer.close()

    if use_ddp:
        for stats in (target_stats, cond_stats):
            dist.all_reduce(stats["sum"], op=dist.ReduceOp.SUM)
            dist.all_reduce(stats["sum_sq"], op=dist.ReduceOp.SUM)
            dist.all_reduce(stats["count"], op=dist.ReduceOp.SUM)
        dist.all_reduce(successful_samples, op=dist.ReduceOp.SUM)
        dist.barrier()

    failure_path = os.path.join(
        partition_dir, f"failures-r{rank:04d}.jsonl")
    with open(failure_path, "w") as failure_file:
        for failure in failed_samples:
            failure_file.write(json.dumps(failure) + "\n")
    if use_ddp:
        dist.barrier()

    if is_main_process():
        shard_paths = sorted(glob.glob(os.path.join(partition_dir, "*.tar")))
        manifest_path = os.path.join(partition_dir, "manifest.txt")
        with open(manifest_path, "w") as manifest:
            for path in shard_paths:
                manifest.write(os.path.basename(path) + "\n")

        stats_path = os.path.join(partition_dir, "stats.pt")
        torch.save({
            "target": finalize_stats(target_stats),
            "cond": finalize_stats(cond_stats),
            "num_samples": int(successful_samples.item()),
            "num_failed": count_failure_records(glob.glob(
                os.path.join(partition_dir, "failures-r*.jsonl"))),
            "representation": {
                "encoder": file_signature(args.encoder_ckpt),
                "autoencoder": file_signature(args.autoencoder_ckpt),
                "latent_dim": args.latent_dim,
                "latent_grid": args.latent_grid,
                "levels": args.levels,
                "seq_len": args.seq_len,
                "target_size": args.target_size,
                "max_frame_span": args.max_frame_span,
            },
            "config": vars(args),
        }, stats_path)
        with open(os.path.join(partition_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        print(f"Wrote {len(shard_paths)} tar shards to {partition_dir}")

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
