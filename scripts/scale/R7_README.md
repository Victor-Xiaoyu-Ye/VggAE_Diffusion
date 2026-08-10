# R7 causal reconstruction and diffusion (10K)

This is the production 10K path for the corrected R7 v3 candidate: temporal
factor 2, `geo96|tex96`, nine RGB frames -> one anchor plus four future chunks.
V3 uses framewise GroupNorm in temporal blocks and directly optimizes geometry
motion cosine; its outputs do not overwrite the prior v2 experiment.

## Complete run

```bash
STAGES=codec,joint,gate,cache,diffusion,sample \
CACHE_NUM_PARTITIONS=1 \
bash scripts/scale/14_r7_recon_then_diffusion.sh
```

The source dual-stream reconstruction artifact is the same pre-staged input used
by the completed t2/t4 probes. Its conventional filename is
`${VGGAE_REF_ROOT}/checkpoints/e5_dual_stream_r5.pt`; this is an artifact alias,
not an experiment-directory name and not a diffusion checkpoint.

The initial codec stage is a fresh v3 optimization stage initialized from the
legacy 3K t2/c192 weights. It does not reuse the legacy optimizer/scheduler.
Override the initializer when necessary:

```bash
PHASE=codec \
LEGACY_INIT_CKPT=obs://path/to/causal_dual_tokenizer_t2_c192_codec/checkpoint_latest.pt \
bash scripts/scale/11_train_causal_tokenizer.sh
```

Joint reconstruction starts from codec `checkpoint_best.pt` and trains the R7
tokenizer plus `DualStreamDecoder`; StreamVGGT, CompactCompressor, and
TextureEncoder remain frozen.

## Diagnostic diffusion

Because v3 currently passes strict causality and RGB/boundary gates but reaches
geometry-motion cosine 0.832 rather than the production threshold 0.95, an
explicitly isolated diagnostic diffusion may be launched without forging a
`gate_passed.json` marker:

```bash
STAGES=cache,diffusion \
ALLOW_DIAGNOSTIC_DIFFUSION=1 \
DIAGNOSTIC_MIN_GEO_MOTION_COSINE=0.82 \
DIAGNOSTIC_DIFFUSION_NAMESPACE=r7_diffusion_t2_c192_ctx1_fut4_v3_diag \
R7_NAMESPACE=r7_t2_c192_v3 \
CACHE_NUM_PARTITIONS=4 \
bash scripts/scale/14_r7_recon_then_diffusion.sh
```

This mode still requires the strict causality probe, production PSNR/LPIPS and
boundary thresholds, and the stated diagnostic geometry threshold. It is not
an acceptance promotion and cannot use the production diffusion namespace.
After training, run `STAGES=sample` with the same diagnostic variables.

```bash
# Reconstruction
PHASE=codec bash scripts/scale/11_train_causal_tokenizer.sh
PHASE=joint bash scripts/scale/11_train_causal_tokenizer.sh

# Cache (exact train statistics; eval uses the training stats downstream)
MODE=train bash scripts/scale/12_cache_causal_latents.sh
MODE=eval bash scripts/scale/12_cache_causal_latents.sh
MODE=merge bash scripts/scale/12_cache_causal_latents.sh

# Fixed-window clean-anchor diffusion, 1 -> 4 chunks
bash scripts/scale/13_train_causal_video_diffusion.sh

# Best EMA sample from deterministic eval anchor
MODE=sample bash scripts/scale/13_train_causal_video_diffusion.sh
```

For multiple cache partitions, either use the complete chain (which runs them
serially) or launch each partition explicitly before merge:

```bash
MODE=train CACHE_PARTITION_ID=0 CACHE_NUM_PARTITIONS=4 bash scripts/scale/12_cache_causal_latents.sh
MODE=train CACHE_PARTITION_ID=1 CACHE_NUM_PARTITIONS=4 bash scripts/scale/12_cache_causal_latents.sh
MODE=train CACHE_PARTITION_ID=2 CACHE_NUM_PARTITIONS=4 bash scripts/scale/12_cache_causal_latents.sh
MODE=train CACHE_PARTITION_ID=3 CACHE_NUM_PARTITIONS=4 bash scripts/scale/12_cache_causal_latents.sh
MODE=eval bash scripts/scale/12_cache_causal_latents.sh
MODE=merge CACHE_NUM_PARTITIONS=4 bash scripts/scale/12_cache_causal_latents.sh
```

Stages 11 and 13 report the raw `DI_throughput` token rate in
`tokens/s/npu`; no divisor or normalized throughput is applied. Stage 12
reports `DI_throughput` in the progress bar only, because a cache job writes no
metrics JSONL, and it applies no divisor — the same convention as
`03_cache_latents.sh`.

The token basis differs by stage and is intentional. Stages 11 and 12 count the
full R7 latent `[B,5,18,18,192]` = 1620 tokens/clip, because both encode all
five chunks. Stage 13 counts the diffusion target `[B,4,324,192]` = 1296
tokens/clip, because the anchor chunk is conditioning rather than a target;
this matches `train_cached_compact_diffusion.py`, which also counts its target.
The probes count latent tokens as `batch * frames * latent_grid ** 2`. All
counts are per process — never multiply by `world_size`, the unit is
`tokens/s/npu`.

## Continue diffusion from 6K to 12K

Only extend a completed 6K run in the same namespace:

```bash
EXTEND=1 \
MAX_STEPS=12000 \
EXTENSION_LR=2e-5 \
EXTENSION_WARMUP_STEPS=200 \
bash scripts/scale/13_train_causal_video_diffusion.sh
```

The trainer keeps periodic, latest, best, and final checkpoints. Evaluation and
sampling use EMA. Early stopping begins at 6K and uses deterministic fixed-cache
samples; a guarded checkpoint with collapsed variance/motion cannot become best.

## Promotion gates

Caching is blocked until the selected R7 reconstruction checkpoint passes all of:

- PSNR >= 23.9
- LPIPS <= 0.13
- chunk boundary/within transition error ratio <= 1.10
- geometry-motion cosine >= 0.95
- strict anchor/prefix causality probe

The primary downstream comparison is E9-v3 c128 at 12K. R7 is considered a
meaningful improvement only when fixed clips/seeds show at least 10% lower future
LPIPS, improved late-half structure, motion ratio closer to 1, better motion and
geometry cosine, and no generated-variance or reconstruction-upper-bound collapse.

## Normalization

Diffusion uses exact training-cache `[position, channel]` z-score statistics:
condition `[1,192]`, future `[4,192]`. The transform is FP32 and exactly inverted
before decoding. Eval statistics are diagnostic only; eval and sampling use the
training statistics frozen in the diffusion checkpoint. No spatial-position
normalization or whitening is used.

## Runtime status

Python compilation and shell syntax checks pass locally. NPU forwards, HCCL,
LPIPS memory usage, and MoXing/OBS transaction behavior still require the normal
ModelArts smoke run before a long job.
