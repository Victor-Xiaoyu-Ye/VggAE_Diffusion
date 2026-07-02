# StreamVGGT Geometry-Aware Generation: Scale Plan

## Objective

Use frozen StreamVGGT features as a geometry-sensitive teacher, compress them
into a tractable video latent, and generate future geometry conditioned on the
observed first RGB frame.

The scale target is:

```text
I0 RGB -> StreamVGGT + tokenizer -> z0
future frames -> StreamVGGT + tokenizer -> z1...z7
diffusion target: rt = zt - z0, t=1...7
decoder input: [z0, z0+r1, ..., z0+r7] + I0 appearance features
```

Frame 0 is observed and must not be diffused.

## Non-Negotiable Stage Boundaries

1. Train and validate the representation.
2. Freeze the tokenizer.
3. Cache compact latents and exact normalization statistics.
4. Train the generator from cached latents.
5. Decode generated latents with the frozen I0 decoder.

Changing the tokenizer invalidates all cached latents and diffusion checkpoints.
On ModelArts, videos remain on OBS and are staged per sample into a bounded
node-local cache. Raw StreamVGGT features are never persisted. Compact latent
shards and exact raw normalization moments are written under the persistent
`obs://yw-ads-training-gy1/data/external/personal/g00833899/y50046448/cache_latents`
directory.
The `/cache/yexiaoyu/vggae_runtime` tree is only staging and may disappear
between jobs. Diffusion reads manifest/statistics through `moxing.file.read`
and streams tar shards through `moxing.file.File`, so resume never relies on
local latent files. Source MP4 is the exception: OpenCV needs a seekable local
path, so each requested video is copied into a bounded disposable cache.

Cache progress is transactional per rank. Every closed tar is uploaded before
the rank progress checkpoint advances. The progress checkpoint contains the
dataset cursor, shard list, failure records, and exact raw normalization
moments. Re-running the same cache script with the same world size resumes from
that cursor. Final merge requires a partition `_SUCCESS` marker.

The active six-node configuration uses a 256-channel, 18x18 latent. Four
deterministic one-second windows per source video produce 1,461,448 cached
clips from the 365,362-row SpatialVID CSV. The generator is a roughly 68M
parameter Compact DiT (`640/8/4/10`) trained with global batch 192 for 40K
steps.

## Checkpoint and Monitoring Contract

Every active trainer writes atomic periodic and final checkpoints. A checkpoint
is resumable only when it contains model weights, FP32 EMA, optimizer,
scheduler, global step, RNG state, arguments, and the exact latent
normalization contract where applicable. FP16 training paths also store the
gradient scaler.

Metrics are written to both TensorBoard and `metrics.jsonl`. Representation
training records RGB loss, PSNR, LPIPS, temporal error, latent statistics,
gradient norm, and held-out depth L1 when depth is available. Generator
training records flow loss, target statistics, gradient norm, learning rate,
fixed-sample velocity MSE, and generated latent statistics.

Every checkpoint save triggers a visual preview. Online training saves decoded
RGB comparisons. Cached training always saves target/generated residual-energy
maps; with an aligned held-out I0 image and I0 decoder checkpoint it also saves
decoded target-latent versus generated-latent RGB grids.

Every node writes stdout and Ascend process logs under `logs/`; rank-0 output
sync and per-node log sync upload them every 60 seconds. Periodic saves update
`checkpoint_latest.pt`, which scale scripts detect and resume automatically.

## Representation Gate

RGB reconstruction alone is not evidence that the compact latent preserves
StreamVGGT geometry. The representation run must include aligned geometry
supervision or teacher probes.

Required validation:

- RGB: PSNR, LPIPS, temporal error, and train/validation gap.
- Geometry: scale-invariant depth error plus edge/normal consistency.
- Robustness: reconstruction after adding the same latent noise used by flow
  training.
- Temporal correspondence: static points should remain stable across frames.
- I0 ablation: geometry should change with the latent, appearance should change
  primarily with the RGB condition.

Do not start the full cache if depth metrics do not improve over an RGB-only
tokenizer or if the decoder collapses to copying I0.

## Generator Bring-Up

Use this sequence before a full run:

1. Overfit 8-32 clips. Samples must reproduce recognizable future frames.
2. Train on 10K cached clips. Verify loss, latent mean/std, and decoded motion.
3. Train on 100K-1M clips. Check diversity and geometry metrics.
4. Start the full SpatialVID run only after the same sampler works at each
   smaller scale.

The first production baseline is the compact DiT, I0-conditioned, generating
seven residual latent frames. Keep text conditioning off until video generation
works; then add classifier-free text conditioning as a separate experiment.

## Wan Initialization

Wan initialization is now a formal scale A/B path:

```bash
bash scripts/scale/05_train_wan_compact.sh
bash scripts/scale/06_sample_wan_compact.sh
```

It consumes the same cached seven-frame residual latents and normalization as
CompactLatentDiT, conditions on I0, and writes a versioned output directory
under `scale/wan_compact_i2v14b480p_v1` by default. Checkpoints store only
trainable adapter, time, modulation, and optional QKV deltas plus EMA; the
frozen Wan backbone is reloaded from `WAN_CKPT_DIR` on resume and sampling.

This does not preserve Wan's native VAE patch input/output semantics. The fair
question is whether pretrained Wan temporal/spatial attention is a better
initialization for the StreamVGGT compact latent field than a from-scratch DiT.
Text conditioning remains off by default so the geometry-aware latent contract
is the only variable.

The current default prefers `Wan2.1-I2V-14B-480P` and finetunes only the last
four self-attention QKV blocks with `TRAIN_QKV=1` and `TRAIN_QKV_LAST_N=4`.
This is the practical middle ground for replicated DDP: it adapts high-level
motion and scene dynamics to StreamVGGT latents while preserving lower/middle
Wan video priors and avoiding the all-QKV OOM seen on 60 GiB NPUs. Use
`TRAIN_QKV_LAST_N=40` only after moving to a memory strategy that can hold full
14B QKV optimizer state.

Move beyond the default last-4-QKV 14B run only after:

- it improves RGB/latent preview quality over from-scratch DiT;
- memory and throughput are acceptable on the assigned node count;
- the larger checkpoint's config loads through `WanCompactAdapter` without
  hidden-dimension mismatch;
- sampling works from the saved trainable-delta checkpoint.

## Data Requirements

- Deduplicate near-identical videos and prevent source/video leakage into eval.
- Store duration, FPS, resolution, motion score, scene cuts, and quality flags.
- Split videos into shots and sample a fixed-duration window; do not spread
  eight frames across an arbitrarily long source video.
- Normalize or bucket source FPS before interpreting a 32-frame window as a
  consistent duration.
- Bucket by aspect ratio, duration, and motion instead of global random seeking.
- Keep a fixed, versioned evaluation set with depth and camera-motion coverage.
- Never report periodic PSNR from the training DataLoader as validation PSNR.
- Track invalid-video rate per data shard; do not silently replace failures.

At 256x18x18 fp16, I0 plus seven residual frames cost about 1.27 MiB/clip.
Budget roughly 1.8 TiB for the current 1.46M-clip compact cache before tar
overhead and temporary files.
