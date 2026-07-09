# Project Context

This document is the durable handoff for VggAE-Diffusion. Keep it current when
goals, architecture, training order, paths, or important decisions change.

## Goals

- Use StreamVGGT as a frozen geometry-aware teacher/encoder.
- Learn a compact latent space that is reconstructable, cacheable, and easier
  to diffuse than raw StreamVGGT tokens.
- Prove that StreamVGGT compact feature space improves geometry-aware video
  generation when paired with a strong pretrained video prior such as Wan.
- **Thesis (stated):** repurpose geometric foundation models (VGGT /
  StreamVGGT) as the representation space for video diffusion, replacing the
  VAE. This is distinct from "Repurposing Geometric Foundation Models for
  Multi-view Diffusion" because video has temporal motion + disocclusion,
  not viewpoint change of a static scene.

## Current Phase: Reconstruction-First (H200)

Generation experiments are paused. The compact latent currently reconstructs
at ~20 PSNR with grid textures, which caps every downstream generator. We
are running diagnostic probes on the 4-card H200 machine to answer:

> Does the frozen StreamVGGT feature space carry enough information
> (especially RGB high-frequency) to reconstruct video at high PSNR?

The answer decides the architecture direction:

```
E1 (raw feature PSNR ceiling)
  >= 28 + E2 shallow levels better -> two-stream latent (z_geo + z_app)
  >= 28 + E2 levels roughly equal  -> single stream + capacity increase
  <= 23                            -> single geometry latent + decoder
                                      hallucinates RGB high-freq via
                                      perceptual + adversarial training
```

Hard gate: reconstruction PSNR < 25 (with no grid texture) blocks any
further scale diffusion runs. The 1.46M-clip scale cache is treated as
disposable until the tokenizer is finalized.

Probes run via `scripts/h200/` (H200 4-GPU), NOT `scripts/scale/` (Ascend
cluster) or `scripts/10k/` (local A100). See `scripts/h200/README.md`.

## Current State

- Active branch: `ascend-910b`.
- **Active phase: reconstruction-first diagnostics on H200.** Generation
  experiments (from-scratch DiT, Wan 14B) are paused pending a
  reconstruction fix; the latest Wan run
  (`outputs/scale/wan_compact_i2v14b480p_v1`, step 36250) shows
  `generated_std=0.256` vs `target_std=0.279` (under-dispersed) and
  `velocity_mse` stuck at ~0.53, consistent with a 20-PSNR latent ceiling.
- Active large-scale dataset: SpatialVID-HQ on OBS.
- Active local 10K dataset path:
  `/public2/LiZhen/yexiaoyu/dataset/spatial-vid-hq-oft` (A100 box).
- Active H200 dataset path:
  `/home/yexiaoyu/data/spatial-vid-hq-oft` (H200 4-GPU box).
- Active H200 encoder checkpoint:
  `/home/yexiaoyu/data/StreamVGGT/checkpoints.pth`.
- Active scale dataset path:
  `obs://yw-ads-training-gy1/data/external/personal/g00833899/y50046448/dataset/SpatialVID-HQ`.
- Active persistent owner OBS root:
  `obs://yw-ads-training-gy1/data/external/personal/g00833899/y50046448`.
- Active scale latent cache version:
  `vggae_streamvggt_256x18_v1` (treated disposable until tokenizer finalized).
- Active compact latent contract:
  `latent_dim=256`, `latent_grid=18`, `seq_len=8`, target is seven future
  residual frames.
- Active Wan checkpoint:
  `Wan2.1-I2V-14B-480P`, expected under
  `${VGGAE_REF_ROOT}/Wan2.1-I2V-14B-480P` on cluster.
- From-scratch Compact DiT has produced recognizable motion but still shows
  ghosting and blurry/duplicated structures at later frames.
- Wan 14B full-QKV DDP was attempted and failed with NPU OOM during DDP reducer
  initialization: about 3.39B trainable params and a 12.63 GiB allocation.
- Current Wan default is last-4-QKV finetuning:
  `TRAIN_QKV=1`, `TRAIN_QKV_LAST_N=4`.
- I0-conditioned decoder is on hold. The I0 path was introduced to supply
  RGB high-frequency that the latent could not encode, but it created an
  "I0 = first frame" shortcut: the decoder warps I0 to all frames and
  motion comes from I0 rather than the geometry latent. The
  reconstruction-first phase uses the I0-free `train_autoencoder.py` /
  `CompactDecoder` path so reconstruction and future t2v generation share
  the same decoder.

## Architecture

```text
RGB clip [8,3,518,518]
  -> frozen StreamVGGT
  -> selected levels [4,11,17,23]
  -> GenerativeTokenizer
  -> compact latent z [8,18,18,256]
```

Training cache stores:

```text
cond = z0                         shape [1,324,256]
target = z1..z7 - z0              shape [7,324,256]
i0_rgb optional preview condition shape [3,H,W]
```

Generator input/output:

```text
x0 ~ Normal(0,I)
x1 = normalized target residual
xt = (1-t) * x0 + t * x1
model(xt, t, cond=normalized z0) -> x1 - x0
loss = MSE(predicted_velocity, target_velocity)
```

Decoder:

```text
generated residuals r1..r7
future latents zt = z0 + rt
[z0,z1,...,z7] + AppearanceCNN(I0 RGB)
  -> I0ConditionalDecoder
  -> RGB video
```

## Active Directories

- `scripts/scale/`: primary ModelArts/Ascend training path.
- `scripts/10k/`: local CUDA bring-up and diagnostics.
- `models/`: tokenizer, decoders, Compact DiT, Wan adapter.
- `data/`: SpatialVID dataset and latent tar shard streaming.
- `utils/`: device, MoXing, distributed, training helpers.
- `Wan2.1/`: vendored Wan source; treat as third-party unless required for
  compatibility.

Avoid using:

- `scripts/legacy/` for new experiments.
- `outputs/` as source documentation.

## Research Risks

These are the open questions that determine architecture direction. Each
has a defined diagnostic probe and a decision rule.

### Risk: StreamVGGT features may not carry RGB high-frequency — CONFIRMED

VGGT is trained on geometry objectives (depth / point maps / camera), not
appearance. Empirically confirmed on SpatialVID 10k:

| Probe | Best PSNR | Best LPIPS | Note |
|-------|-----------|------------|------|
| E1 raw 4-level + generic decoder | 20.60 | 0.215 | plateau ~ep17 |
| E4 b0 DPTHead (no compact) | 20.30 | 0.324 | DPT does not beat E1 |
| E4 b1 compact + DPT decoder | 17.31 | 0.525 | ~3 dB compression cost |

- Falsified: "swap to DPT decoder and PSNR jumps" (4DLangRecon single-scene
  result does not transfer to SpatialVID).
- Falsified: "VGGT shallow levels as z_app" (E2 level4 alone = 18.62 < E1).
- Consequence: appearance must come from an explicit RGB TextureEncoder.

### Decision: Dual-stream latent (z_geo + z_tex from TextureEncoder)

- Content: `z_geo` = CompactCompressor(frozen VGGT); `z_tex` =
  TextureEncoder(per-frame RGB), multi-scale packed into the same GxG grid;
  DualStreamDecoder reconstructs from `(z_geo, z_tex)` only (no RGB skip).
- Reason: E1/E4 ceiling; TexturePredictor(z_geo->z_tex) is information-
  theoretically blocked and is disabled. Decoder-side GAN/hallucination is
  deferred until the dual-stream reconstruction gate is measured.
- Probe: `scripts/h200/probe_e5_texture_recon.sh` (`TEX_MODE=oracle|zero`).
- Gate: oracle PSNR >= 25 -> proceed to Wan diffusion on both streams
  (or z_tex conditioned on z_geo). oracle still ~20 -> raise tex capacity.
- Rejected for now: VGGT-shallow z_app; I0 AppearanceCNN shortcut;
  TexturePredictor; decoder-first adversarial icing.

### Risk: grid texture source unidentified

Current reconstructions show grid textures attributed (by hypothesis) to
three combined sources: PixelShuffle checkerboard, adaptive_avg_pool
37->18 odd/even misalignment, and VGGT patch-attention boundaries.

- Probe: `scripts/h200/probe_e3_grid_isolation.sh` (PixelShuffle vs
  resize-conv, same tokenizer).
- Decision: if resize-conv removes the grid, PixelShuffle was the cause and
  the production decoder switches to resize-conv. If the grid persists, the
  cause is upstream (pooling alignment or patch-attn) and requires a
  tokenizer-level fix.

### Risk: I0 conditioning creates a motion shortcut

The I0 decoder reaches reasonable PSNR by warping I0 to all frames, so
motion comes from I0 rather than the geometry latent. This blocks t2v and
makes the generator's motion contribution unmeasurable.

- Mitigation: reconstruction-first phase uses the I0-free
  `train_autoencoder.py` / `CompactDecoder` path. I0 is reintroduced only
  if a future i2v ablation is explicitly desired, with I0 dropout +
  wrong-frame I0 corruption to prevent the shortcut.

## Decisions

### Decision: Diffuse Future Residuals, Not Full Latents

- Content: frame 0 is condition only; generator predicts seven residual
  latents `zt - z0`.
- Reason: reduces target entropy and uses I0 as stable scene/appearance anchor.
- Rejected alternatives: predicting all eight latents including frame 0;
  predicting RGB directly; predicting raw StreamVGGT tokens.
- Impact: sampling must prepend `z0`; checkpoints with old eight-target layout
  are not resume-compatible.

### Decision: Cache Compact Latents Before Scale Diffusion

- Content: scale diffusion reads tar shards and statistics from OBS instead of
  decoding videos and running StreamVGGT online.
- Reason: full SpatialVID training cannot afford repeated online StreamVGGT
  encoding; cache enables generator-only iteration and stable normalization.
- Rejected alternatives: online full-scale diffusion; storing raw 2048-dim
  StreamVGGT features.
- Impact: tokenizer changes invalidate cache, stats, and generator checkpoints.

### Decision: Use Frame/Channel Statistics

- Content: target residual stats are `[future_frames, latent_dim]`; condition
  stats are `[1, latent_dim]`.
- Reason: residual variance grows by horizon and differs by channel; scalar
  scaling is insufficient.
- Rejected alternatives: single scalar latent scale; global `[latent_dim]`
  stats only; spatial-position normalization.
- Impact: resume refuses mismatched stats and representation signatures.

### Decision: Use Wan 14B As Pretrained Video Prior

- Content: use `Wan2.1-I2V-14B-480P` through `WanCompactAdapter` on the same
  StreamVGGT compact latent contract.
- Reason: from-scratch DiT has not removed ghosting; Wan provides pretrained
  video dynamics and spatiotemporal attention.
- Rejected alternatives: only scaling from-scratch Compact DiT; relying on Wan
  VAE latent semantics; old online Wan harness.
- Impact: checkpoint stores trainable deltas only and requires the same
  `WAN_CKPT_DIR` at resume/sampling.

### Decision: Train Last-N Wan QKV Blocks By Default

- Content: default `TRAIN_QKV=1`, `TRAIN_QKV_LAST_N=4`.
- Reason: full 14B QKV under replicated DDP OOMed; last layers adapt high-level
  motion while preserving lower/middle Wan priors.
- Rejected alternatives: freeze all QKV forever; train all 40 QKV blocks under
  DDP; full model finetune without FSDP/ZeRO.
- Impact: if last-4 improves samples, expand to last 8/12; full QKV needs a
  different memory strategy.

### Decision: Keep Text Conditioning Off

- Content: current generator is non-text, I0-conditioned.
- Reason: the project must first prove geometry-aware latent generation; text
  conditioning introduces a second variable.
- Rejected alternatives: CLIP conditioning from legacy Wan scripts; immediate
  native UMT5 conditioning.
- Impact: Wan text projection is not trained unless explicitly enabled.

## Conventions

### Shell And Paths

- Active shell scripts define editable hyperparameters near the top.
- Distributed topology is read from ModelArts variables:
  `VC_WORKER_NUM`, `VC_TASK_INDEX`, `VC_WORKER_HOSTS`.
- Preserve absolute-path helper execution:
  `"$PYTHON_BIN" "$PROJECT/scripts/..."`.
- Do not assume the current working directory is repo root.
- `VGGAE_REF_ROOT` is read-only input dependency storage, usually
  `/cache/yexiaoyu/vggae_ref`.
- `LOCAL_CACHE_ROOT` is disposable staging, usually
  `/cache/yexiaoyu/vggae_runtime`.

### Code Style

- Prefer existing utilities over new ad hoc code.
- Use `rg` for search.
- Use `apply_patch` for manual edits.
- Default to ASCII in files.
- Keep comments short and only where they clarify non-obvious behavior.
- Do not revert user or generated changes unless explicitly requested.

### Training/Checkpoint Requirements

- Active trainers must support resume.
- Save `checkpoint_latest.pt`.
- Save metrics to both TensorBoard and `metrics.jsonl`.
- Save visual previews whenever checkpoints are saved.
- Store RNG state, args, optimizer, scheduler, scaler when applicable.
- Cached generator checkpoints must store latent normalization contract.

### Verification

Run relevant checks before pushing:

```bash
/home/yexiaoyu/miniconda3/envs/rae/bin/python -m py_compile <files>
bash -n <scripts>
git diff --check
```

Local checks do not prove NPU/HCCL/MoXing runtime correctness.

## Known Issues

- Compact latent reconstruction is ~20 PSNR with grid textures. This caps
  every downstream generator: the latest Wan run shows `generated_std` below
  `target_std` and `velocity_mse` stuck at ~0.53, consistent with the
  generator learning a too-narrow distribution over an under-expressive
  latent. Reconstruction must reach PSNR >= 25 before further scale
  generation.
- Grid texture sources (PixelShuffle checkerboard, adaptive_avg_pool
  37->18 odd/even misalignment, VGGT patch-attention boundaries) are
  unverified and will be isolated by `probe_e3_grid_isolation.sh`.
- I0-conditioned decoder reaches higher PSNR by warping I0 to all frames
  (I0 = first frame = target[0]), creating a motion shortcut. Reconstruction
  training now uses the I0-free `train_autoencoder.py` path.
- From-scratch Compact DiT still has ghosting despite larger 768-dim runs.
- Full 14B QKV DDP OOMs on 60 GiB 910B cards.
- MoXing/OBS logs can show noisy multiprocessing logging rollover errors even
  when transfer succeeds.
- `_SUCCESS` missing in latent cache partitions means cache generation or
  finalization is incomplete; merge should not bypass this guard.
- Historical tokenizer temporal mixer caused I0 contract drift; active scale
  contract must keep tokenizer/cache settings aligned.

## Next Tasks

1. Run E5 dual-stream reconstruction probe on H200 (4 GPUs):

   ```bash
   bash scripts/h200/probe_e5_texture_recon.sh                 # oracle
   TEX_MODE=zero bash scripts/h200/probe_e5_texture_recon.sh   # ablation
   ```

   Decision: oracle PSNR >= 25 -> dual-stream viable for Wan. Still ~20 ->
   raise `TEX_DIM` / `TEX_BASE_CH` or keep a higher-res tex grid.

2. If E5 oracle passes the gate, wire production AE training to
   CompactCompressor + TextureEncoder + DualStreamDecoder (replace the
   single-stream GenerativeTokenizer path for reconstruction).

3. Only after reconstruction passes the gate, rebuild the latent cache
   (now both z_geo and z_tex) and resume Wan diffusion. Do NOT rebuild
   cache before the dual-stream contract is finalized by E5.

6. When diffusion resumes, before any model-parallel full-QKV Wan run, do
   the cheap QKV-capacity control:

   ```bash
   TRAIN_QKV=0 bash scripts/scale/smoke_wan_compact.sh        # adapter only
   TRAIN_QKV=1 TRAIN_QKV_LAST_N=8 bash scripts/scale/smoke_wan_compact.sh
   ```

   If adapter-only ≈ last-4, QKV is not the bottleneck and full-QKV will
   not help. Expand to last-8/12 only if last-N improves monotonically.

## Context Update Protocol

For any substantial task, update this document when the change affects:

- project goal;
- active architecture;
- training order;
- paths or storage;
- checkpoint/resume behavior;
- model contract;
- known failure modes;
- accepted or rejected experiment branches.

Final responses after substantial work should include:

```text
Context Update
新增知识
- ...

新增约定
- ...

新增决策
- ...

当前状态更新
- ...

建议写入项目文档
- ...
```
