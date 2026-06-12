# Large-scale training path

The scale path freezes the representation before the full SpatialVID run.
It does not decode MP4 or run StreamVGGT inside the diffusion training loop.

## Default 6x8 configuration

- Source videos: 365,362.
- Cached windows per video: 4 deterministic one-second windows.
- Total diffusion clips: 1,461,448.
- Compact latent: `256 x 18 x 18`.
- Compact DiT: 640 hidden, 8 spatial blocks, 4 temporal blocks, 10 heads
  (about 68M parameters).
- Global diffusion batch: `48 NPUs x batch 1 x accumulation 4 = 192`.
- 40,000 optimizer steps: about 5.25 passes over the cached clips.

The representation autoencoder trains for 8 epochs and the I0 decoder for 6.
Those online stages choose a new random one-second window on every dataset
access. The cache stage instead uses four deterministic windows distributed
from the beginning to the end of each raw video.

The launch command copies external dependencies to
`/cache/yexiaoyu/vggae_ref`. The active scale path expects the frozen encoder
at:

```text
/cache/yexiaoyu/vggae_ref/StreamVGGT/checkpoints.pth
```

There is no `veggie_ref` directory. The geometry autoencoder, I0 decoder,
latent statistics, and Compact DiT are produced by this pipeline.

Persistent OBS layout:

```text
obs://yw-ads-training-gy1/data/external/personal/g00833899/y50046448/
├── cache_latents/
│   └── vggae_streamvggt_256x18_v1/
│       ├── train/
│       └── eval/
└── output/                         # fallback when OUTPUT_URL is absent
```

ModelArts training outputs use `$OUTPUT_URL` when it is provided. The latent
cache always uses the fixed `cache_latents` path above so a new training job can
reuse it independently of its output directory.

The SpatialVID-HQ root is already set to:

```text
obs://yw-ads-training-gy1/data/external/personal/g00833899/y50046448/dataset/SpatialVID-HQ
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
   - By default, one 6x8 job partitions the full CSV across 48 ranks.
   - For multiple cache jobs, edit `CACHE_PARTITION_ID` and
     `CACHE_NUM_PARTITIONS`.
   - Each completed tar shard is uploaded immediately to the persistent
     `cache_latents/.../train` directory, then removed from local staging.
   - `progress-rXXXXX.pt` is the exact resume cursor and raw-moment state.
     `status-rXXXXX.json` is the human-readable progress report.
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

At 256 channels, each eight-frame fp16 cache sample is about 1.27 MiB. The
1,461,448-clip cache therefore needs roughly 1.8 TiB before tar overhead and
metadata. The configured `/cache` budget is used only for source MP4 files
that OpenCV must access through a seekable local path, checkpoint staging, and
local outputs. Manifest, statistics, progress metadata, and latent tar shards
are read directly through `moxing.file.read/File`; latent tar files are
streamed and are not downloaded into the local cache. Persistent latent shards
remain under the fixed `cache_latents` OBS directory. Training checkpoints,
metrics, samples, TensorBoard events, stdout, and NPU process logs remain under
`$OUTPUT_URL`.

Distributed launch uses HCCL. ModelArts variables are read automatically:
`VC_WORKER_NUM`, `VC_TASK_INDEX`, and `VC_WORKER_HOSTS`. The scale scripts
expect 6 workers and 8 NPUs per worker. `MASTER_PORT` remains editable at the
top of each script.

Video files are copied individually through MoXing into
`MOX_VIDEO_CACHE_DIR` because OpenCV requires a local seekable path. Compact
latent tar shards stay on OBS and are streamed with `moxing.file.File`.

The representation checkpoint is a data contract. Do not continue changing the
tokenizer after latent caching starts. If the tokenizer changes, rebuild the
cache and its statistics.

Cache generation uses shard-level transactional resume. Re-running the same
`03_cache_latents.sh` with the same 6x8 topology loads each rank's progress
checkpoint and continues after its last uploaded tar. A partition writes
`_SUCCESS`, final moments, and its manifest only after all ranks finish.
`04_merge_latent_cache.sh` rejects partitions without `_SUCCESS`. Diffusion
training restarts never depend on local latent files and always read the
persistent OBS manifest.

All stage 00, 01, and 05 training checkpoints are resumable. Every periodic
save also updates `checkpoint_latest.pt`, and the scale scripts automatically
resume it when rerun. `RESUME` can still override this with a local or `obs://`
checkpoint. Stage 05 restores model, FP32 EMA, optimizer,
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

Each stage contains:

```text
checkpoint_latest.pt
checkpoint_*.pt
metrics.jsonl
tb/
samples/
logs/train_node*.log
logs/node-*/npu/
```
