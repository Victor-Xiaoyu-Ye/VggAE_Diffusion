# Large-scale training path

The scale path freezes the representation before the full SpatialVID run.
It does not decode MP4 or run StreamVGGT inside the diffusion training loop.

## Scale configuration

- Source videos: 365,362.
- Cached windows per video: 4 deterministic one-second windows.
- Total diffusion clips: 1,461,448.
- Compact latent: `256 x 18 x 18`.
- Compact DiT: 640 hidden, 8 spatial blocks, 4 temporal blocks, 10 heads
  (about 68M parameters).
- The existing latent cache was generated with one 6x8 job, but merged cache
  training is independent of cache-generation world size.
- Compact DiT continuation supports larger ModelArts jobs such as 30-44 nodes.
  The script uses all assigned nodes by default.
- Default continuation batch on 30-44 nodes:
  `nodes x 8 NPUs x batch 1 x accumulation 1 = 240-352`.
- Default continuation target: `MAX_STEPS=100000`, resuming model/EMA weights
  from the latest checkpoint and resetting the optimizer schedule with
  `RESUME_MODE=weights`.

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
   - After it writes `checkpoint_latest.pt`, run
     `preflight_next_stage.sh after_i0` to validate the checkpoint contract.
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
   - Check completion with
     `bash scripts/scale/check_cache_status.sh train`. It must print
     `COMPLETE` before merge.
   - If all ranks report complete progress but `_SUCCESS` is missing because
     the torchrun job failed at finalization, recover with
     `bash scripts/scale/finalize_cache_partition.sh train`.
5. `03_cache_eval_latents.sh`
   - Run once to cache the automatically selected held-out split.
   - Check completion with
     `bash scripts/scale/check_cache_status.sh eval`. It must print
     `COMPLETE` before merge.
   - The same finalization recovery is available with
     `bash scripts/scale/finalize_cache_partition.sh eval`.
   - Run this before the full cache, then run `smoke_compact_dit.sh`. The smoke
     job uses only the held-out cache and performs two optimizer steps to test
     OBS streaming, NPU forward/backward, EMA, checkpointing, and RGB preview.
6. `04_merge_latent_cache.sh`
   - Merge tar manifests and exact per-channel normalization statistics.
   - Training and evaluation caches are merged separately.
   - First run `preflight_next_stage.sh before_merge`; it requires every
     partition to contain `_SUCCESS`, `manifest.txt`, `stats.pt`, and
     `config.json`.
7. `05_train_compact_dit.sh`
   - Train by optimizer step from streaming tar shards.
   - Set `EVAL_CACHE_DIR` to a held-out cache for fixed validation metrics.
   - Every checkpoint writes TensorBoard/JSONL metrics and a target/generated
     latent preview.
   - The eval cache stores its first frame, so RGB previews automatically use
     the I0 aligned with the first latent sample.
   - By default `AUTO_RESUME=1` tries the current output URL and then the
     persistent personal OBS mirror for `checkpoint_latest.pt`.
   - For large continuation jobs, the default `RESUME_MODE=weights` loads
     model/EMA/global step but starts a fresh optimizer and LR schedule. Use
     `RESUME_MODE=full` only when continuing the same run with the same
     optimizer schedule.
   - Run `preflight_next_stage.sh before_diffusion` before launching the full
     DiT job. It validates both merged caches, normalization tensor
     shapes, sampled tar objects, and the I0 checkpoint.
   - The default script is the 640-dim baseline/continuation path.
8. `05_train_compact_dit_large.sh`
   - Train a fresh larger Compact DiT from the same merged latent cache.
   - Default experiment name:
     `compact_dit_768d10s6t_h12_v1`.
   - Default model:
     `model_dim=768`, `spatial_depth=10`, `temporal_depth=6`,
     `num_heads=12`.
   - Default launch expectation: 32 nodes x 8 NPUs. The script still uses the
     actual nodes assigned by ModelArts, but prints a warning if it differs
     from `EXPECTED_NNODES=32`.
   - This script sets `AUTO_RESUME=0` by default so it does not accidentally
     continue the 640-dim baseline. Set `RESUME=...` only when explicitly
     continuing the same large experiment.
   - Output and checkpoints are versioned under:

```text
$OUTPUT_URL/scale/compact_dit_768d10s6t_h12_v1
obs://yw-ads-training-gy1/data/external/personal/g00833899/y50046448/output/scale/compact_dit_768d10s6t_h12_v1
```

   - `DI_throughput` is reported after dividing the raw token rate by 20, and
     `train/raw_DI_throughput` is also written to JSONL for audit.
9. `06_sample_compact_dit.sh`
   - Generate seven future frames from one observed RGB frame.
   - Run `preflight_next_stage.sh before_sample` to validate the generator and
     all upstream representation artifacts.
10. `06_sample_compact_dit_large.sh`
   - Sample the versioned large DiT checkpoint from
     `compact_dit_768d10s6t_h12_v1`.
   - Outputs are written to `scale/samples_compact_dit_768d10s6t_h12_v1`.
11. `smoke_wan_compact.sh`
   - Run two Wan-initialized optimizer steps and preview generation.
   - Use this before the full 14B run to verify weight loading, NPU memory,
     DDP communication, checkpoint writing, RGB preview, and output mirroring.
12. `05_train_wan_compact.sh`
   - Train a Wan-initialized generator on the same cached compact residual
     latents.
   - Default experiment name: `wan_compact_i2v14b480p_v1`.
   - Default checkpoint selection prefers `Wan2.1-I2V-14B-480P` from
     `WAN_CKPT_DIR`, then falls back to `Wan2.1-T2V-1.3B`.
   - The checkpoint stores only trainable Wan adapter/QKV/time/modulation
     deltas plus EMA, not the frozen Wan backbone. Resume and sampling
     therefore require the same `WAN_CKPT_DIR`.
   - `TRAIN_QKV=1` and `TRAIN_QKV_LAST_N=4` by default. This finetunes the
     last four Wan self-attention QKV blocks while preserving lower/middle
     pretrained video priors.
   - Full-QKV finetuning requires `TRAIN_QKV_LAST_N=40` for the 14B checkpoint,
     but the replicated DDP path has already shown OOM risk on 60 GiB NPUs.
   - This is the current A/B test against from-scratch Compact DiT. It does
     not require retraining the geometry AE, I0 decoder, or latent cache.
   - Override `WAN_CKPT_DIR` to test a larger Wan checkpoint after confirming
     it fits memory; the adapter reads hidden dimension, frequency dimension,
     heads, and layers from the checkpoint config.
13. `06_sample_wan_compact.sh`
   - Sample `wan_compact_i2v14b480p_v1` using the same I0-conditioned decoder.
   - Outputs are written to `scale/samples_wan_compact_i2v14b480p_v1`.

Geometry-autoencoder inference can be run independently with:

```bash
bash scripts/scale/inference_geometry_autoencoder.sh
```

It reads held-out SpatialVID videos from OBS on demand and uploads
reconstruction grids plus `metrics.json` to the configured output URL.

I0-decoder inference can be run independently with:

```bash
bash scripts/scale/inference_i0_decoder.sh
```

It uploads grids and `temporal_metrics.json` under
`$OUTPUT_URL/scale/inference/i0_decoder`. Each grid compares target frames,
AE reconstruction, I0 reconstruction, and repeated-z0 controls for checking
whether temporal changes come from geometry latents or from copying the first
frame.

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
All active DataLoaders use the `spawn` multiprocessing context so workers do
not inherit a MemArts gRPC handle initialized by the parent process. Missing
or corrupt MP4 objects use deterministic replacement samples and are counted
as `data/decode_replacements` in `metrics.jsonl` and TensorBoard. Treat a
replacement rate above 1% as a dataset-copy/layout failure rather than normal
training noise.

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

Checkpoint staging and resume first try the current ModelArts `$OUTPUT_URL`,
then fall back to the persistent owner mirror:

```text
obs://yw-ads-training-gy1/data/external/personal/g00833899/y50046448/output/scale/
```

This fallback applies to the geometry autoencoder, I0 decoder, Compact DiT,
preflight checks, inference, and sampling, so a new ModelArts job can continue
artifacts written by an earlier job.

Wan initialization is available as `05_train_wan_compact.sh`. It still bypasses
Wan's native VAE patch interface, so the fair comparison is not "Wan video VAE
vs our latent"; it is "from-scratch transformer vs Wan-pretrained temporal/
spatial attention on the same StreamVGGT compact latent contract." Text
conditioning remains off by default.

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
