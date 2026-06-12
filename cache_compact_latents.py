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
from utils.distributed import is_main_process, setup_ddp
from utils.device import get_device, get_device_name, resolve_dtype
from utils.file_signature import sampled_file_signature
from utils.latent_stats import (
    create_moments,
    finalize_moments,
    merge_raw_moments,
    reduce_moments,
    update_moments,
)
from utils.moxing_io import (
    copy_file,
    is_remote_path,
    join_remote,
    read_text,
    remote_exists,
    stage_remote_file,
    write_text,
)


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
    parser.add_argument(
        "--store_i0_rgb", action="store_true",
        help="Store the observed first RGB frame for automatic previews")

    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--latent_grid", type=int, default=18)
    parser.add_argument("--token_dim", type=int, default=2048)
    parser.add_argument("--levels", type=int, nargs="+", default=[4, 11, 17, 23])
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--target_size", type=int, default=518)
    parser.add_argument("--num_frames_per_video", type=int, default=8)
    parser.add_argument("--max_frame_span", type=int, default=32)
    parser.add_argument("--clip_duration_seconds", type=float, default=0.0)
    parser.add_argument("--disable_temporal_mixer", action="store_true")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--local_rank", type=int, default=0)
    return parser.parse_args()


class TarShardWriter:
    def __init__(self, output_dir, prefix, samples_per_tar,
                 remote_output_dir=""):
        self.output_dir = output_dir
        self.remote_output_dir = remote_output_dir
        self.prefix = prefix
        self.samples_per_tar = samples_per_tar
        self.shard_index = 0
        self.samples_in_shard = 0
        self.archive = None
        self.paths = []

    def _open_next(self):
        path = os.path.join(
            self.output_dir, f"{self.prefix}-{self.shard_index:06d}.tar")
        remote_path = (
            join_remote(self.remote_output_dir, os.path.basename(path))
            if self.remote_output_dir else "")
        if os.path.exists(path) or (
                remote_path and remote_exists(remote_path)):
            raise FileExistsError(
                f"Refusing to overwrite existing cache shard: "
                f"{remote_path or path}")
        self.archive = tarfile.open(path, mode="w")
        self.paths.append(remote_path or path)
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
            local_path = self.archive.name
            self.archive.close()
            self.archive = None
            if self.remote_output_dir:
                remote_path = join_remote(
                    self.remote_output_dir,
                    os.path.basename(local_path))
                copy_file(local_path, remote_path)
                os.unlink(local_path)

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
    tokenizer.disable_temporal_mixer = (
        args.disable_temporal_mixer
        or bool(checkpoint.get("args", {}).get(
            "disable_temporal_mixer", False))
    )
    tokenizer.eval()
    for parameter in tokenizer.parameters():
        parameter.requires_grad_(False)
    return encoder, tokenizer


def count_failure_records(paths):
    count = 0
    for path in paths:
        with open(path) as failure_file:
            count += sum(1 for line in failure_file if line.strip())
    return count


def moments_to_cpu(moments):
    return {
        key: value.detach().cpu()
        for key, value in moments.items()
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
    remote_partition_dir = ""
    if is_remote_path(args.output_dir):
        remote_partition_dir = join_remote(
            args.output_dir,
            f"part-{partition_id:05d}-of-{num_partitions:05d}")
        local_root = os.environ.get(
            "MOX_CACHE_WRITER_DIR", "/cache/vggae/cache_writer")
    else:
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
        if world_size > local_world_size:
            raise ValueError(
                "Multi-node caching requires an OBS --output_dir so every "
                "rank's moments and manifest are visible to global rank 0")
        local_root = args.output_dir
    local_partition_root = os.path.join(
        local_root,
        f"part-{partition_id:05d}-of-{num_partitions:05d}",
    )
    partition_dir = os.path.join(
        local_partition_root,
        f"rank-{rank:05d}",
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
        clip_duration_seconds=args.clip_duration_seconds,
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
        partition_dir, f"latents-r{rank:05d}", args.samples_per_tar,
        remote_output_dir=remote_partition_dir)

    # Long 10M runs cannot accumulate sum/sum_sq accurately in NPU FP32.
    # Each batch is reduced on-device, then the tiny [S,D] arrays are added
    # into CPU FP64 moments.
    stats_device = torch.device("cpu") if remote_partition_dir else device
    target_stats = create_moments(
        args.seq_len - 1, args.latent_dim, stats_device)
    cond_stats = create_moments(1, args.latent_dim, stats_device)

    sample_index = 0
    failed_samples = []
    successful_samples = torch.zeros(
        (), device=device, dtype=torch.long)
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

                update_moments(target_stats, target)
                update_moments(cond_stats, i0_flat)

                for batch_index in range(target.shape[0]):
                    key = (
                        f"p{partition_id:05d}-r{rank:04d}-"
                        f"{sample_index:09d}")
                    sample = {
                        "target": target[batch_index].to(
                            device="cpu", dtype=torch.float16),
                        "cond": i0_flat[batch_index].to(
                            device="cpu", dtype=torch.float16),
                        "caption": batch["caption"][batch_index],
                        "video_id": batch["video_id"][batch_index],
                    }
                    if args.store_i0_rgb:
                        sample["i0_rgb"] = (
                            frames[batch_index, 0].float()
                            .clamp(0, 1).mul(255).round()
                            .to(device="cpu", dtype=torch.uint8)
                        )
                    writer.write(key, sample)
                    sample_index += 1
                    successful_samples += 1
    finally:
        writer.close()

    failed_count = torch.tensor(
        len(failed_samples), device=device, dtype=torch.long)
    if use_ddp:
        if not remote_partition_dir:
            for stats in (target_stats, cond_stats):
                reduce_moments(stats)
        dist.all_reduce(successful_samples, op=dist.ReduceOp.SUM)
        dist.all_reduce(failed_count, op=dist.ReduceOp.SUM)

    failure_path = os.path.join(
        partition_dir, f"failures-r{rank:04d}.jsonl")
    with open(failure_path, "w") as failure_file:
        for failure in failed_samples:
            failure_file.write(json.dumps(failure) + "\n")
    rank_manifest = "\n".join(writer.paths)
    if rank_manifest:
        rank_manifest += "\n"
    if remote_partition_dir:
        rank_moments_path = os.path.join(
            partition_dir, f"moments-r{rank:05d}.pt")
        torch.save({
            "target": moments_to_cpu(target_stats),
            "cond": moments_to_cpu(cond_stats),
        }, rank_moments_path)
        copy_file(
            failure_path,
            join_remote(
                remote_partition_dir,
                f"failures-r{rank:05d}.jsonl"))
        write_text(
            join_remote(
                remote_partition_dir,
                f"manifest-r{rank:05d}.txt"),
            rank_manifest)
        copy_file(
            rank_moments_path,
            join_remote(
                remote_partition_dir,
                f"moments-r{rank:05d}.pt"))
    if use_ddp:
        dist.barrier()

    if is_main_process():
        if remote_partition_dir:
            shard_paths = []
            target_rank_moments = []
            cond_rank_moments = []
            for source_rank in range(world_size):
                rank_manifest_path = join_remote(
                    remote_partition_dir,
                    f"manifest-r{source_rank:05d}.txt")
                shard_paths.extend(
                    path.strip()
                    for path in read_text(rank_manifest_path).splitlines()
                    if path.strip())
                remote_moments_path = join_remote(
                    remote_partition_dir,
                    f"moments-r{source_rank:05d}.pt")
                local_moments_path = stage_remote_file(
                    remote_moments_path,
                    os.environ.get(
                        "MOX_METADATA_CACHE_DIR",
                        "/cache/vggae/metadata_cache"),
                    max_cache_bytes=10 * 1024 ** 3,
                )
                rank_moments = torch.load(
                    local_moments_path, map_location="cpu",
                    weights_only=False)
                target_rank_moments.append(rank_moments["target"])
                cond_rank_moments.append(rank_moments["cond"])
            target_stats = merge_raw_moments(target_rank_moments)
            cond_stats = merge_raw_moments(cond_rank_moments)
        else:
            shard_paths = sorted(
                glob.glob(os.path.join(local_partition_root, "*.tar"))
                + glob.glob(os.path.join(
                    local_partition_root, "rank-*", "*.tar")))
        manifest_path = os.path.join(partition_dir, "manifest.txt")
        with open(manifest_path, "w") as manifest:
            for path in shard_paths:
                manifest.write(path + "\n")

        stats_path = os.path.join(partition_dir, "stats.pt")
        torch.save({
            "normalization_version": 2,
            "target": finalize_moments(target_stats),
            "cond": finalize_moments(cond_stats),
            "moments": {
                "target": moments_to_cpu(target_stats),
                "cond": moments_to_cpu(cond_stats),
            },
            "num_samples": int(successful_samples.item()),
            "num_failed": int(failed_count.item()),
            "representation": {
                "encoder": sampled_file_signature(args.encoder_ckpt),
                "autoencoder": sampled_file_signature(args.autoencoder_ckpt),
                "latent_dim": args.latent_dim,
                "latent_grid": args.latent_grid,
                "levels": args.levels,
                "seq_len": args.seq_len,
                "target_size": args.target_size,
                "max_frame_span": args.max_frame_span,
                "clip_duration_seconds": args.clip_duration_seconds,
                "disable_temporal_mixer": (
                    tokenizer.disable_temporal_mixer),
            },
            "config": vars(args),
        }, stats_path)
        with open(os.path.join(partition_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        if remote_partition_dir:
            copy_file(
                manifest_path,
                join_remote(remote_partition_dir, "manifest.txt"))
            copy_file(
                stats_path,
                join_remote(remote_partition_dir, "stats.pt"))
            copy_file(
                os.path.join(partition_dir, "config.json"),
                join_remote(remote_partition_dir, "config.json"))
        print(
            f"Wrote {len(shard_paths)} tar shards to "
            f"{remote_partition_dir or partition_dir}")

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
