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
between jobs. Diffusion resume therefore reads the manifest, statistics, and
tar shards from OBS rather than relying on local latent files.

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

Wan2.1 1.3B is an initialization experiment, not the default production path.
The current adapter bypasses Wan's native VAE patch embedding and output head,
so pretrained input/output semantics are not preserved. Legacy CLIP embeddings
also do not match Wan's pretrained UMT5 context.

The retained Wan training harness is not ready for this comparison: it does
not yet consume cached seven-frame residuals or wire the adapter's I0/native
UMT5 hooks. Do not launch it as a scale run.

For a fair A/B test:

- Use the same cached latent shards and normalization as CompactLatentDiT.
- Use native UMT5 embeddings through Wan's `text_embedding`.
- Condition on I0 and predict only seven future residual frames.
- Warm up adapters, then progressively unfreeze attention/FFN blocks.
- Compare sample quality per GPU-hour, not training loss alone.

Move to a larger image-to-video Wan checkpoint only after the compact-latent
pipeline produces a valid baseline.

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
