#!/usr/bin/env python3
"""Cache durable absolute R7 latents into streaming tar shards."""

from __future__ import annotations

import argparse
import io
import json
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from cache_compact_latents import SafeDataset, TarShardWriter, safe_collate_fn
from data.loader_utils import multiprocessing_loader_kwargs
from data.video_dataset import SpatialVidDataset
from streamvggt.models.streamvggt import StreamVGGT
from utils.device import (
    configure_backend_compatibility,
    get_device,
    get_device_name,
    resolve_dtype,
)
from utils.distributed import is_main_process, setup_ddp
from utils.encoder_loader import load_encoder_checkpoint
from utils.latent_stats import finalize_moments, merge_raw_moments
from utils.moxing_io import (
    copy_file,
    is_remote_path,
    join_remote,
    read_bytes,
    read_text,
    remote_exists,
    write_text,
)
from utils.r7_representation import (
    build_contract,
    encode_dual,
    load_checkpoint,
    load_r7_modules,
    validate_contract,
)
from utils.training import ThroughputMeter, count_latent_tokens


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cache R7 absolute anchor/future latents")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--video_root", required=True)
    parser.add_argument("--annotation_index", default="")
    parser.add_argument("--encoder_ckpt", required=True)
    parser.add_argument("--r7_ckpt", required=True)
    parser.add_argument(
        "--dual_ae_ckpt", default="",
        help="Optional source dual-AE path; otherwise use the R7 contract")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", choices=("train", "eval"), required=True)

    parser.add_argument("--index_shard_id", type=int, default=0)
    parser.add_argument("--index_num_shards", type=int, default=1)
    parser.add_argument("--partition_id", type=int, default=-1)
    parser.add_argument("--num_partitions", type=int, default=0)
    parser.add_argument("--max_videos", type=int, default=0)
    parser.add_argument("--check_files", action="store_true")
    parser.add_argument("--samples_per_tar", type=int, default=512)
    parser.add_argument("--clips_per_video", type=int, default=1)
    parser.add_argument("--resume_cache", action="store_true")
    parser.add_argument("--store_i0_rgb", action="store_true")
    parser.add_argument("--store_rgb", action="store_true", help="Keep full raw clip for held-out RGB evaluation")
    parser.add_argument("--independent_anchor", action="store_true", help="Encode condition from the first frame alone")
    parser.add_argument("--window_ae_norm", choices=('legacy','framewise'), default=None,
                        help="Explicit historical codec semantics for versioned window caches")
    parser.add_argument(
        "--allow_legacy_checkpoint", action="store_true",
        help="Allow a contract-less R7 checkpoint; strict four-prefix load remains")

    parser.add_argument("--seq_len", type=int, default=9)
    parser.add_argument("--target_size", type=int, default=518)
    parser.add_argument("--latent_grid", type=int, default=18)
    parser.add_argument("--max_frame_span", type=int, default=0)
    parser.add_argument("--clip_duration_seconds", type=float, default=1.0)
    parser.add_argument("--decode_retries", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--dtype", choices=("fp16", "bf16", "fp32"), default="bf16")
    parser.add_argument("--local_rank", type=int, default=0)
    return parser.parse_args()


def cpu_moments(num_chunks, latent_dim):
    return {
        "sum": torch.zeros(num_chunks, latent_dim, dtype=torch.float64),
        "sum_sq": torch.zeros(num_chunks, latent_dim, dtype=torch.float64),
        "count": torch.zeros(num_chunks, 1, dtype=torch.float64),
    }


def update_cpu_moments(moments, tensor):
    """Accumulate [B,S,N,D] values with all reductions on CPU in FP64."""
    values = tensor.detach().to(device="cpu", dtype=torch.float64)
    if values.ndim != 4:
        raise ValueError(f"Expected [B,S,N,D], got {tuple(values.shape)}")
    expected = (moments["sum"].shape[0], moments["sum"].shape[1])
    if (values.shape[1], values.shape[-1]) != expected:
        raise ValueError(
            f"Moment shape mismatch {tuple(values.shape)}; expected "
            f"[B,{expected[0]},N,{expected[1]}]")
    moments["sum"] += values.sum(dim=(0, 2))
    moments["sum_sq"] += values.square().sum(dim=(0, 2))
    moments["count"] += values.shape[0] * values.shape[2]


def moments_to_cpu(moments):
    return {key: value.detach().double().cpu() for key, value in moments.items()}


def artifact_path(partition_dir, name):
    return (join_remote(partition_dir, name)
            if is_remote_path(partition_dir)
            else os.path.join(partition_dir, name))


def load_torch_artifact(path):
    source = io.BytesIO(read_bytes(path)) if is_remote_path(path) else path
    return torch.load(source, map_location="cpu", weights_only=False)


def publish_file(local_path, destination):
    if is_remote_path(destination):
        copy_file(local_path, destination)
    elif os.path.abspath(local_path) != os.path.abspath(destination):
        os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
        os.replace(local_path, destination)


def publish_torch(payload, destination, staging_dir, filename):
    local_path = os.path.join(staging_dir, filename)
    torch.save(payload, local_path)
    publish_file(local_path, destination)


def build_cache_representation(args, checkpoint, config):
    """Validate the accepted checkpoint contract and add cache signatures."""
    saved = checkpoint.get("representation_contract")
    structural = build_contract(config, include_signatures=False)
    if saved is None:
        if not args.allow_legacy_checkpoint:
            raise RuntimeError(
                "R7 checkpoint has no representation_contract; pass "
                "--allow_legacy_checkpoint only for a known legacy artifact")
        if not args.dual_ae_ckpt:
            raise RuntimeError(
                "Legacy R7 caching requires --dual_ae_ckpt so the source "
                "representation can be signed")
    else:
        validate_contract(saved, structural)

    representation = build_contract(
        config,
        encoder_ckpt=args.encoder_ckpt,
        dual_ae_ckpt=args.dual_ae_ckpt,
        r7_ckpt=args.r7_ckpt,
    )
    saved_signatures = saved.get("signatures", {}) if saved else {}
    current_signatures = representation["signatures"]
    saved_encoder = saved_signatures.get("streamvggt")
    if (saved_encoder is not None
            and saved_encoder != current_signatures["streamvggt"]):
        raise RuntimeError(
            "--encoder_ckpt does not match the accepted R7 checkpoint contract")
    saved_source = saved_signatures.get("source_dual_ae")
    if args.dual_ae_ckpt:
        if (saved_source is not None
                and saved_source != current_signatures["source_dual_ae"]):
            raise RuntimeError(
                "--dual_ae_ckpt does not match the accepted R7 checkpoint contract")
    elif saved_source is not None:
        current_signatures["source_dual_ae"] = saved_source
    else:
        raise RuntimeError(
            "R7 checkpoint contract lacks source_dual_ae signature; provide "
            "--dual_ae_ckpt")
    if args.window_ae_norm:
        from utils.window_codec import runtime_contract
        if not args.independent_anchor: raise ValueError('window cache requires independent anchor')
        representation['window_codec_runtime'] = runtime_contract(args.window_ae_norm)
    return representation


def cache_run_config(
        args, partition_id, num_partitions, world_size):
    """Return cursor-sensitive settings kept separate from representation."""
    return {
        "split": args.split,
        "csv": args.csv,
        "video_root": args.video_root,
        "annotation_index": args.annotation_index,
        "index_shard_id": args.index_shard_id,
        "index_num_shards": args.index_num_shards,
        "partition_id": partition_id,
        "num_partitions": num_partitions,
        "world_size": world_size,
        "max_videos": args.max_videos,
        "check_files": args.check_files,
        "samples_per_tar": args.samples_per_tar,
        "clips_per_video": args.clips_per_video,
        "store_i0_rgb": args.store_i0_rgb,
        **({"store_rgb": True} if args.store_rgb else {}),
        **({"independent_anchor": True} if args.independent_anchor else {}),
        **({"window_ae_norm": args.window_ae_norm} if args.window_ae_norm else {}),
        "seq_len": args.seq_len,
        "target_size": args.target_size,
        "latent_grid": args.latent_grid,
        "max_frame_span": args.max_frame_span,
        "clip_duration_seconds": args.clip_duration_seconds,
        "decode_retries": args.decode_retries,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
    }


def validate_requested_config(args, config):
    mismatches = {}
    requested = {
        "seq_len": args.seq_len,
        "target_size": args.target_size,
        "latent_grid": args.latent_grid,
        "clip_duration_seconds": args.clip_duration_seconds,
    }
    for name, value in requested.items():
        if getattr(config, name) != value:
            mismatches[name] = (getattr(config, name), value)
    if mismatches:
        raise RuntimeError(
            f"Cache arguments differ from accepted R7 config: {mismatches}")
    if config.seq_len != 9 or config.temporal_factor not in (1, 2):
        raise RuntimeError(
            "R7 probe cache requires nine RGB frames and temporal_factor 1 or 2")
    if config.latent_dim != 192:
        raise RuntimeError(
            "R7 probe cache requires latent channel dimension 192, got "
            f"D={config.latent_dim}")
    expected_latent_frames = 1 + (config.seq_len - 1) // config.temporal_factor
    if config.latent_seq_len != expected_latent_frames:
        raise RuntimeError(
            "R7 latent sequence contract is inconsistent: "
            f"T={config.latent_seq_len}, expected={expected_latent_frames}")
    if config.latent_grid != 18 or config.latent_dim != 192:
        raise RuntimeError(
            "R7 durable cache requires grid=18 and total latent width 192, got "
            f"grid={config.latent_grid}, split="
            f"{config.geo_latent_dim}|{config.tex_latent_dim}")


def main():
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError(
            "Durable resume requires --batch_size 1 for an exact cursor")
    if args.samples_per_tar < 1 or args.clips_per_video < 1:
        raise ValueError("samples_per_tar and clips_per_video must be positive")

    use_ddp, rank, local_rank, world_size = setup_ddp()
    device_type = get_device_name()
    if device_type == "cpu":
        raise RuntimeError("R7 latent caching requires an accelerator")
    configure_backend_compatibility(device_type)
    device = get_device(local_rank)
    encoder_dtype = resolve_dtype(args.dtype)

    partition_id = (args.partition_id if args.partition_id >= 0
                    else args.index_shard_id)
    num_partitions = (args.num_partitions if args.num_partitions > 0
                      else args.index_num_shards)
    if not 0 <= partition_id < num_partitions:
        raise ValueError(
            f"partition_id must be in [0,{num_partitions}), got {partition_id}")
    partition_name = f"part-{partition_id:05d}-of-{num_partitions:05d}"

    remote_partition_dir = ""
    if is_remote_path(args.output_dir):
        remote_partition_dir = join_remote(args.output_dir, partition_name)
        local_root = os.environ.get(
            "MOX_CACHE_WRITER_DIR",
            "/cache/yexiaoyu/vggae_runtime/cache/latent_writer")
    else:
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
        if world_size > local_world_size:
            raise ValueError(
                "Multi-node caching requires an OBS --output_dir")
        local_root = args.output_dir
    local_partition_root = os.path.join(local_root, partition_name)
    rank_dir = os.path.join(local_partition_root, f"rank-{rank:05d}")
    os.makedirs(rank_dir, exist_ok=True)
    shared_partition = remote_partition_dir or local_partition_root

    success_path = artifact_path(shared_partition, "_SUCCESS")
    if args.resume_cache and remote_exists(success_path):
        if is_main_process():
            print(f"R7 cache partition already complete: {success_path}")
        if use_ddp:
            dist.barrier()
            dist.destroy_process_group()
        return

    checkpoint = load_checkpoint(args.r7_ckpt)
    config, compressor, tex_encoder, tokenizer, decoder, matched = \
        load_r7_modules(checkpoint)
    validate_requested_config(args, config)
    representation = build_cache_representation(args, checkpoint, config)
    if args.window_ae_norm:
        from utils.window_codec import configure_codec
        configure_codec(tokenizer,args.window_ae_norm)
    del decoder

    encoder = StreamVGGT(
        img_size=config.target_size, patch_size=14, embed_dim=1024)
    load_encoder_checkpoint(
        encoder, args.encoder_ckpt, verbose=is_main_process())
    encoder = encoder.to(device=device, dtype=encoder_dtype).eval()
    compressor = compressor.to(device).eval()
    tex_encoder = tex_encoder.to(device).eval()
    tokenizer = tokenizer.to(device).eval()
    for module in (encoder, compressor, tex_encoder, tokenizer):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    if is_main_process():
        print(f"[R7 strict load] {matched}")

    dataset = SpatialVidDataset(
        csv_path=args.csv,
        video_root=args.video_root,
        seq_len=config.seq_len,
        target_size=config.target_size,
        annotation_index_path=args.annotation_index,
        max_videos=args.max_videos,
        num_frames_per_video=config.seq_len,
        temporal_jitter=False,
        index_shard_id=args.index_shard_id,
        index_num_shards=args.index_num_shards,
        check_files=args.check_files,
        max_frame_span=args.max_frame_span,
        clip_duration_seconds=config.clip_duration_seconds,
        decode_retries=args.decode_retries,
        clips_per_video=args.clips_per_video,
    )
    rank_indices = range(rank, len(dataset), world_size)
    total_rank_items = len(rank_indices)

    run_config = cache_run_config(
        args, partition_id, num_partitions, world_size)
    progress_path = artifact_path(
        shared_partition, f"progress-r{rank:05d}.pt")
    progress_state = None
    if args.resume_cache and remote_exists(progress_path):
        progress_state = load_torch_artifact(progress_path)
        if progress_state.get("representation") != representation:
            raise RuntimeError(
                f"Cache representation changed while resuming rank {rank}")
        if progress_state.get("config") != run_config:
            raise RuntimeError(
                f"Cache run config changed while resuming rank {rank}")
        if progress_state.get("world_size") != world_size:
            raise RuntimeError(
                "Cache resume requires the original DDP world size")

    processed_items = int(
        progress_state.get("processed_items", 0) if progress_state else 0)
    sample_index = int(
        progress_state.get("sample_index", 0) if progress_state else 0)
    failed_samples = list(
        progress_state.get("failed_samples", []) if progress_state else [])
    if not 0 <= processed_items <= total_rank_items:
        raise RuntimeError(
            f"Invalid rank cursor {processed_items}/{total_rank_items}")
    future_chunks = config.latent_seq_len - 1
    target_moments = (progress_state["moments"]["target"]
                      if progress_state else cpu_moments(
                          future_chunks, config.latent_dim))
    cond_moments = (progress_state["moments"]["cond"]
                    if progress_state else cpu_moments(1, config.latent_dim))

    dataloader = DataLoader(
        Subset(SafeDataset(dataset), rank_indices[processed_items:]),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=safe_collate_fn,
        pin_memory=device_type == "cuda",
        drop_last=False,
        **multiprocessing_loader_kwargs(args.num_workers),
    )
    writer = TarShardWriter(
        rank_dir,
        f"r7-r{rank:05d}",
        args.samples_per_tar,
        remote_output_dir=remote_partition_dir,
        shard_index=int(
            progress_state.get("shard_index", 0) if progress_state else 0),
        paths=progress_state.get("paths", []) if progress_state else None,
    )

    def write_artifact_text(name, content):
        path = artifact_path(shared_partition, name)
        if is_remote_path(path):
            write_text(path, content)
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as handle:
                handle.write(content)

    def status(phase, error=""):
        payload = {
            "phase": phase,
            "rank": rank,
            "world_size": world_size,
            "partition_id": partition_id,
            "num_partitions": num_partitions,
            "processed_items": processed_items,
            "total_rank_items": total_rank_items,
            "successful_samples": sample_index,
            "failed_samples": len(failed_samples),
            "completed_shards": len(writer.paths),
            "last_shard": writer.paths[-1] if writer.paths else "",
        }
        if error:
            payload["error"] = error
        return payload

    def save_rank_progress(phase="running"):
        state = {
            "version": 1,
            "rank": rank,
            "world_size": world_size,
            "processed_items": processed_items,
            "sample_index": sample_index,
            "shard_index": writer.shard_index,
            "paths": list(writer.paths),
            "moments": {
                "target": moments_to_cpu(target_moments),
                "cond": moments_to_cpu(cond_moments),
            },
            "failed_samples": list(failed_samples),
            "representation": representation,
            "config": run_config,
        }
        publish_torch(
            state, progress_path, rank_dir, f"progress-r{rank:05d}.pt")
        write_artifact_text(
            f"status-r{rank:05d}.json",
            json.dumps(status(phase), indent=2, sort_keys=True))

    throughput_meter = ThroughputMeter()
    progress = tqdm(
        dataloader, disable=not is_main_process(),
        desc=f"Caching R7 {args.split} latents")
    try:
        with torch.inference_mode():
            for batch in progress:
                failed_samples.extend(batch["errors"])
                replacement_count = int(batch.get("decode_replacements", 0))
                if replacement_count:
                    failed_samples.extend({
                        "error": "decode_replacement",
                        "requested_video_id": requested,
                        "replacement_video_id": actual,
                    } for requested, actual in zip(
                        batch.get("requested_video_id", []),
                        batch.get("video_id", [])))
                if batch["frames"] is None:
                    processed_items += batch["_batch_size"]
                    if writer.archive is None:
                        save_rank_progress()
                    continue

                frames = batch["frames"].to(device=device, non_blocking=True)
                geo, texture = encode_dual(
                    encoder, compressor, tex_encoder, frames, encoder_dtype)
                latent = tokenizer.encode(geo, texture)
                expected = (
                    1, config.latent_seq_len, config.latent_grid,
                    config.latent_grid, config.latent_dim)
                if tuple(latent.shape) != expected:
                    raise RuntimeError(
                        f"R7 encode shape {tuple(latent.shape)} != {expected}")
                throughput_meter.update(count_latent_tokens(latent))
                flat = latent.reshape(
                    1, config.latent_seq_len,
                    config.latent_grid ** 2, config.latent_dim)
                cond = flat[:, :1]
                target = flat[:, 1:]
                anchor_relative_l2 = None
                if args.independent_anchor:
                    geo0, tex0 = encode_dual(encoder, compressor, tex_encoder, frames[:, :1], encoder_dtype)
                    independent = tokenizer.encode(geo0, tex0).reshape_as(cond)
                    anchor_relative_l2 = float((independent-cond).float().norm()/cond.float().norm().clamp_min(1e-8))
                    cond = independent
                if tuple(cond.shape[1:]) != (
                        1, config.latent_grid ** 2, config.latent_dim):
                    raise RuntimeError(
                        "R7 cond shape differs from representation contract")
                if tuple(target.shape[1:]) != (
                        future_chunks, config.latent_grid ** 2,
                        config.latent_dim):
                    raise RuntimeError(
                        "R7 target shape differs from representation contract")

                window_index = int(batch["window_index"][0])
                key = (f"p{partition_id:05d}-r{rank:05d}-"
                       f"s{sample_index:010d}-w{window_index:04d}")
                cached = {
                    "target": target[0].to(
                        device="cpu", dtype=torch.float16),
                    "cond": cond[0].to(
                        device="cpu", dtype=torch.float16),
                    "caption": batch["caption"][0],
                    "video_id": batch["video_id"][0],
                    "requested_video_id": batch.get(
                        "requested_video_id", batch["video_id"])[0],
                    "window_index": window_index,
                    "clips_per_video": args.clips_per_video,
                }
                if args.store_i0_rgb:
                    cached["i0_rgb"] = (
                        frames[0, 0].float().clamp(0, 1).mul(255).round()
                        .to(device="cpu", dtype=torch.uint8))
                if args.store_rgb:
                    cached["rgb"] = (frames[0].float().clamp(0, 1).mul(255).round()
                                     .to(device="cpu", dtype=torch.uint8))
                if anchor_relative_l2 is not None:
                    cached["anchor_relative_l2"] = anchor_relative_l2
                writer.write(key, cached)
                update_cpu_moments(target_moments, target)
                update_cpu_moments(cond_moments, cond)
                sample_index += 1
                processed_items += 1
                if writer.samples_in_shard >= writer.samples_per_tar:
                    writer.close_current()
                    save_rank_progress()
                progress.set_postfix(
                    samples=sample_index,
                    failed=len(failed_samples),
                    shards=len(writer.paths),
                    DI_throughput=throughput_meter.format(),
                )
    except BaseException as error:
        writer.abort_current()
        try:
            write_artifact_text(
                f"status-r{rank:05d}.json",
                json.dumps(
                    status("aborted", repr(error)), indent=2, sort_keys=True))
        except Exception:
            pass
        raise

    writer.close()
    save_rank_progress("complete")
    failure_local = os.path.join(rank_dir, f"failures-r{rank:05d}.jsonl")
    with open(failure_local, "w") as handle:
        for failure in failed_samples:
            handle.write(json.dumps(failure, sort_keys=True) + "\n")
    failure_target = artifact_path(
        shared_partition, f"failures-r{rank:05d}.jsonl")
    publish_file(failure_local, failure_target)
    write_artifact_text(
        f"manifest-r{rank:05d}.txt",
        "".join(path + "\n" for path in writer.paths))
    publish_torch(
        {"target": moments_to_cpu(target_moments),
         "cond": moments_to_cpu(cond_moments)},
        artifact_path(shared_partition, f"moments-r{rank:05d}.pt"),
        rank_dir,
        f"moments-r{rank:05d}.pt",
    )

    if use_ddp:
        dist.barrier()

    if is_main_process():
        shard_paths = []
        target_entries = []
        cond_entries = []
        total_samples = 0
        total_failed = 0
        for source_rank in range(world_size):
            rank_status = json.loads(read_text(artifact_path(
                shared_partition, f"status-r{source_rank:05d}.json")))
            if rank_status.get("phase") != "complete":
                raise RuntimeError(
                    f"Rank {source_rank} did not complete its cache")
            total_samples += int(rank_status["successful_samples"])
            total_failed += int(rank_status["failed_samples"])
            shard_paths.extend(line.strip() for line in read_text(artifact_path(
                shared_partition, f"manifest-r{source_rank:05d}.txt"
            )).splitlines() if line.strip())
            rank_moments = load_torch_artifact(artifact_path(
                shared_partition, f"moments-r{source_rank:05d}.pt"))
            target_entries.append(rank_moments["target"])
            cond_entries.append(rank_moments["cond"])
        if len(shard_paths) != len(set(shard_paths)):
            raise RuntimeError("Duplicate tar shard paths in R7 cache")
        target_raw = merge_raw_moments(target_entries)
        cond_raw = merge_raw_moments(cond_entries)
        expected_count = total_samples * config.latent_grid ** 2
        if not torch.all(target_raw["count"] == expected_count):
            raise RuntimeError("Target moment count does not match sample count")
        if not torch.all(cond_raw["count"] == expected_count):
            raise RuntimeError("Cond moment count does not match sample count")

        manifest_local = os.path.join(rank_dir, "manifest.txt")
        with open(manifest_local, "w") as handle:
            handle.writelines(path + "\n" for path in shard_paths)
        publish_file(
            manifest_local, artifact_path(shared_partition, "manifest.txt"))
        cache_config = run_config
        config_local = os.path.join(rank_dir, "config.json")
        with open(config_local, "w") as handle:
            json.dump(cache_config, handle, indent=2, sort_keys=True)
        publish_file(
            config_local, artifact_path(shared_partition, "config.json"))
        representation_local = os.path.join(rank_dir, "representation.json")
        with open(representation_local, "w") as handle:
            json.dump(representation, handle, indent=2, sort_keys=True)
        publish_file(
            representation_local,
            artifact_path(shared_partition, "representation.json"))

        stats = {
            "normalization_version": 2,
            "target": finalize_moments(target_raw),
            "cond": finalize_moments(cond_raw),
            "moments": {"target": target_raw, "cond": cond_raw},
            "num_samples": total_samples,
            "num_failed": total_failed,
            "num_shards": len(shard_paths),
            "representation": representation,
            "config": cache_config,
        }
        publish_torch(
            stats, artifact_path(shared_partition, "stats.pt"),
            rank_dir, "stats.pt")
        write_artifact_text("_SUCCESS", json.dumps({
            "partition_id": partition_id,
            "num_partitions": num_partitions,
            "num_samples": total_samples,
            "num_failed": total_failed,
            "num_shards": len(shard_paths),
            "representation": representation,
            "config": cache_config,
        }, sort_keys=True))
        print(
            f"Completed R7 {args.split} partition {partition_id}: "
            f"samples={total_samples}, failed={total_failed}, "
            f"shards={len(shard_paths)}")

    if use_ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
