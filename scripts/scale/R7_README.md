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

## Decoder robustness finetune (stage 11, PHASE=decoder_robust)

The v3 diagnostic diffusion showed that near-manifold generated latents decode
to noise: the decoder only ever saw exact tokenizer outputs. This phase
freezes everything except `DualStreamDecoder` and trains it on Gaussian-noised
latents (per-sample sigma ~ U(0, 0.35) raw-latent units on the four future
chunks; the anchor stays clean, matching diffusion inference). Initialization
resolves `${R7_NAMESPACE}/joint/checkpoint_best.pt`; latent-side lambdas are
forced to zero. Every eval reports the standard clean metrics plus
`eval/noised_*` at the fixed sigma 0.22; a checkpoint can only become best
when clean quality holds AND noised PSNR beats the init baseline.

Acceptance is manual: clean PSNR >= 23.9, LPIPS <= 0.13, boundary <= 1.10
(geometry cosine is unchanged by construction — the tokenizer is frozen), and a
strictly improved noised-PSNR curve. This phase never writes or modifies
`gate_passed.json`, and its checkpoint can NEVER be passed as `--r7_ckpt`
downstream: the cache binds `signatures.r7` to the accepted joint bytes.
Diffusion consumes it only through the separate `--decoder_ckpt` override,
which strictly loads `decoder.*` weights alone.

## Wan2.1-T2V-1.3B full-finetune diffusion (stages 15/16 via the 14 chain)

Stage 16 replaces the from-scratch CompactLatentDiT with a fully finetuned
Wan2.1-T2V-1.3B trunk (`WanCompactAdapter` in `full_finetune` +
`anchor_frame` mode; ~1.4B trainable, replicated DDP, no model parallel),
switches to x0/clean-target prediction with flow-matching sampling and
`time_shift_alpha=3.0`, and conditions on native UMT5-xxl embeddings
precomputed once by stage 15 (captions from the SpatialVID annotation index
keyed by cached `video_id` — the latent cache is NOT rebuilt). The launch
command must pre-copy `Wan2.1-T2V-1.3B` (with
`models_t5_umt5-xxl-enc-bf16.pth` and `google/umt5-xxl`) into
`${VGGAE_REF_ROOT}`; 14B directories are rejected.

Launching follows the chain model — one ModelArts submission runs a STAGES
list on all nodes with durable OBS publication and barriers between stages
(never a sequence of interactive bash commands):

```bash
# Complete serial chain: decoder robustness -> text sidecar -> Wan training
# -> best-EMA sample. Diagnostic bypass is required while the v3 joint
# checkpoint qualifies at geometry cosine 0.832 < 0.95.
STAGES=decoder_robust,text_embed,wan_diffusion,wan_sample \
ALLOW_DIAGNOSTIC_DIFFUSION=1 \
R7_NAMESPACE=r7_t2_c192_v3 \
bash scripts/scale/14_r7_recon_then_diffusion.sh
```

Each stage can also be its own submission (`STAGES=decoder_robust`, then
`STAGES=text_embed`, ...). The chain publishes
`decoder_robust/checkpoint_best.pt` durably and re-stages it on every node,
mirroring the codec->joint pattern; `wan_diffusion`/`wan_sample` re-verify the
gate (or diagnostic) qualification and forward `DECODER_CKPT` to stage 16.
For a 2-step smoke of stage 16 alone (after decoder_robust and text_embed
artifacts exist): `bash scripts/scale/smoke_wan_t2v.sh`.

Stage 15 prefers a durable prebuilt `annotation_index_oft.json` on OBS
(`ANNOTATION_INDEX_URL`, default
`${PERSISTENT_OBS_ROOT}/metadata/annotation_index_oft.json`); if absent it
stages the annotations tree, builds the index, and publishes it back. A
silently empty staging (missing OBS prefix — `copy_parallel` does not fail on
nonexistent sources) is a hard error with instructions, not a no-op; the
fastest manual fix is building `annotation_index.json` on the A100 box with
`scripts/build_annotation_index.sh` and uploading that single file.

Stage-16 historical defaults were 30K steps, batch 4/NPU, adapter LR 1e-4 /
trunk LR 1e-5, `wan_freeze_steps=500`, EMA 0.9999, CFG dropout 0.1, eval CFG
scale 3.0, and early stop from 10K. The completed run overturned the 30K
assumption: sampled RGB peaked near 7K while teacher-forced x0 MSE continued to
fall through 14K. The wrapper now caps the default at 16K, begins early-stop
accounting at 6K, evaluates 64 clips, and emits labelled PNG/MP4 previews every
eval. The old arm remains diagnostic; use `docs/R7_QUALITY_PLAN.md` for the
factor-1/teacher-bridge/horizon-motion redesign rather than extending it blindly.
With ~10k cached clips one epoch is ~52 optimizer steps at world 48; watch both
train/eval x0-MSE and sampled late-horizon guards before extending.
