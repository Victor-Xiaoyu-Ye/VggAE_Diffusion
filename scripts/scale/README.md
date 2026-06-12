# Large-scale training path

The scale path freezes the representation before the full SpatialVID run.
It does not decode MP4 or run StreamVGGT inside the diffusion training loop.

Edit local checkpoint/environment paths once in `scripts/spatialvid_config.sh`.
The SpatialVID-HQ root is already set to:

```text
obs://yw-pixelgeek-training-data-gy1/01.USERS/z00546255/data/yexiaoyu/dataset/SpatialVID-HQ
```

Stage 02 automatically
creates deterministic full-train and held-out metadata from the original
SpatialVID CSV. No manually prepared evaluation CSV is required.

1. `00_train_geometry_autoencoder.sh`
   - Train tokenizer and decoder on a diverse representation subset.
   - Set `ENABLE_DEPTH=1` only after confirming the OBS depth layout.
2. `01_train_i0_decoder.sh`
   - Train the appearance-conditioned RGB decoder with the tokenizer frozen.
3. `02_shard_metadata.sh`
   - Optional helper for multiple independent cache jobs.
4. `03_cache_latents.sh`
   - By default, one 24x8 job partitions the full CSV across 192 ranks.
   - For multiple cache jobs, edit `CACHE_PARTITION_ID` and
     `CACHE_NUM_PARTITIONS`.
5. `03_cache_eval_latents.sh`
   - Run once to cache the automatically selected held-out split.
6. `04_merge_latent_cache.sh`
   - Merge tar manifests and exact per-channel normalization statistics.
   - Training and evaluation caches are merged separately.
7. `05_train_compact_dit.sh`
   - Train by optimizer step from streaming tar shards.
   - Set `EVAL_CACHE_DIR` to a held-out cache for fixed validation metrics.
   - Every checkpoint writes TensorBoard/JSONL metrics and a target/generated
     latent preview.
   - The eval cache stores its first frame, so RGB previews automatically use
     the I0 aligned with the first latent sample.
8. `06_sample_compact_dit.sh`
   - Generate seven future frames from one observed RGB frame.

At 512 channels, an eight-frame fp16 cache is about 2.65 MB/video. The current
365,362-row metadata CSV therefore needs roughly 1 TB for compact latents,
before tar overhead and temporary files. The configured `/cache` budget is
used only as a bounded staging area; persistent shards remain under
`$OUTPUT_URL`.

Distributed launch uses HCCL. ModelArts variables are read automatically:
`VC_WORKER_NUM`, `VC_TASK_INDEX`, and `VC_WORKER_HOSTS`. The scale scripts
expect 24 workers and 8 NPUs per worker. `MASTER_PORT` remains editable at the
top of each script.

Video files are downloaded individually through MoXing into
`MOX_VIDEO_CACHE_DIR`. Compact latent tar shards stay on OBS and are staged
through `MOX_LATENT_CACHE_DIR`.

The representation checkpoint is a data contract. Do not continue changing the
tokenizer after latent caching starts. If the tokenizer changes, rebuild the
cache and its statistics.

All stage 00, 01, and 05 training checkpoints are resumable. `RESUME` accepts
either a local path or an `obs://` checkpoint. Stage 05 restores model, FP32 EMA, optimizer,
schedule, global step, normalization contract, and RNG state. The streaming
shard iterator restarts rather than resuming at an exact tar byte offset, so
sample order after recovery is not bit-identical.

Wan initialization is deliberately not the default scale script. The current
compact adapter bypasses Wan's native VAE patch interface, and the legacy CLIP
text path does not match Wan's pretrained UMT5 context. Establish the compact
DiT baseline first, then compare Wan initialization on the same cached latents.

Outputs are written under local `RUN_ROOT`, then global rank 0 mirrors them
every `OUTPUT_SYNC_SECONDS` to
`$OUTPUT_URL/scale/<stage>`.
