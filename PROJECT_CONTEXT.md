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
- **Active phase: reconstruction-first diagnostics on H200.**
- **R1 RESULT (2026-07-15, e5_oracle_s2d_match_geo, 40 epochs, 4x H200):
  eval PSNR 24.92 ± 2.62 (32 clips), LPIPS 0.257, train l1 0.033 vs eval
  l1 0.037 (no overfitting — the plateau is real). This breaks the ~20 dB
  single-stream ceiling by ~4.3 dB and lands within one standard error of
  the 25 dB gate. Dual-stream + s2d packing + match_geo regularization is
  validated as the right direction; the remaining gap goes to R5 (VGGT
  feature-consistency loss).**
- Diffusability (final R1 checkpoint, 64 clips): absolute streams PASS
  (z_geo/z_tex low-freq dominant, top32_var 0.77/0.65, kurtosis ~3.2/3.6);
  RESIDUAL streams FAIL (high-freq 0.49/0.70, eff-rank 110/132, kurtosis
  6.5) — see the residual-target re-evaluation under Decisions.
- Generation experiments (from-scratch DiT, Wan 14B) remain paused; the
  latest Wan run (`outputs/scale/wan_compact_i2v14b480p_v1`, step 36250)
  showed under-dispersion consistent with both the old 20-PSNR latent and
  the whitened residual target.
- Active large-scale dataset: SpatialVID-HQ on OBS.
- Active local 10K dataset path:
  `/public2/LiZhen/yexiaoyu/dataset/spatial-vid-hq-oft` (A100 box).
- Active H200 dataset path:
  `/home/yexiaoyu/data/spatial-vid-hq-oft` (H200 4-GPU box).
- Active H200 encoder checkpoint:
  `/home/yexiaoyu/data/StreamVGGT/checkpoints.pth`.
- Active scale dataset path:
  `obs://yw-ads-training-gy1/data/external/personal/g00833899/y50046448/dataset/SpatialVID-HQ`.
- 10k subset OBS mirror (same content as the H200/A100 local copies, used by
  the small-scale E7 path): `obs://.../y50046448/spatial-vid-hq-oft` —
  NOTE its inner layout differs from the HQ tree AND from the local copies:
  videos live under `videos/SpatialVid/HQ/videos/group_*` (not
  `videos/SpatialVID/videos`); metadata at
  `data/train/SpatialVID_HQ_metadata.csv` is the FULL 360k CSV, so split
  generation against this mirror must use the availability filter
  (`ensure_spatialvid_subset_splits`), never `--skip_file_check`.
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

### Risk: StreamVGGT features may not carry RGB high-frequency — NEEDS RE-VERIFICATION

VGGT is trained on geometry objectives (depth / point maps / camera), not
appearance. Measured on SpatialVID 10k, BUT all numbers below predate the
2026-07 probe-infrastructure fixes (DDP forward bypass, single-clip eval,
unverified encoder checkpoint loading, no-residual temporal attention) and
must be re-measured before being treated as ceilings:

| Probe | Best PSNR | Best LPIPS | Note |
|-------|-----------|------------|------|
| E1 raw 4-level + generic decoder | 20.60 | 0.215 | plateau ~ep17; single-clip eval; encoder load unverified |
| E4 b0 DPTHead (no compact) | 20.30 | 0.324 | ran with DDP grad-sync bug (4x independent single-GPU) |
| E4 b1 compact + DPT decoder | 17.31 | 0.525 | same DDP bug; "~3 dB compression cost" unreliable |

- Falsified (subject to re-verification): "swap to DPT decoder and PSNR
  jumps"; "VGGT shallow levels as z_app" (E2 level4 alone = 18.62 < E1 —
  note E2 compressed 37->18 by default, conflating level info with
  compression cost).
- External context (2026-07 survey): ~20 PSNR matches what frozen semantic
  encoders give with plain reconstruction decoders; RAE/GLD-style works
  recover high fidelity by training the decoder with L1+LPIPS+**GAN** loss.
  GLD (arXiv:2603.22275) reports 35.41 dB PSNR reconstruction from frozen
  geometric-foundation features with an adversarially trained decoder. The
  ceiling may therefore be the decoder training recipe, not the features.

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

### Risk: grid texture source unidentified — RESOLVED (2026-07-15)

R1 (e5_oracle_s2d_match_geo) samples show NO grid/checkerboard texture.
The dual-stream path differs from the old grid-textured pipeline in two
ways: resize-conv upsampling in DualStreamDecoder (no PixelShuffle) and
s2d packing (no fractional adaptive_avg_pool in the tex stream). This
matches industry practice (Wan/Hunyuan/CogVideoX decoders all use
resize+conv; LTX's PixelShuffle has documented grid artifacts). The
standalone E3 isolation probe is no longer needed for the main line; run
it only if a future architecture reintroduces PixelShuffle or fractional
pooling. Remaining reconstruction gap is SOFTNESS (missing high-freq
detail), not artifacts — the known signature of L1 + weak-LPIPS decoding;
addressed by R5 (feature-consistency loss) and, in production, stronger
perceptual training per the MIRA/GLD recipes.

### Risk: I0 conditioning creates a motion shortcut

The I0 decoder reaches reasonable PSNR by warping I0 to all frames, so
motion comes from I0 rather than the geometry latent. This blocks t2v and
makes the generator's motion contribution unmeasurable.

- Mitigation: reconstruction-first phase uses the I0-free
  `train_autoencoder.py` / `CompactDecoder` path. I0 is reintroduced only
  if a future i2v ablation is explicitly desired, with I0 dropout +
  wrong-frame I0 corruption to prevent the shortcut.

## Decisions

### Decision: Diffuse Future Residuals, Not Full Latents — UNDER RE-EVALUATION (2026-07)

- Content: frame 0 is condition only; generator predicts seven residual
  latents `zt - z0`.
- Reason: reduces target entropy and uses I0 as stable scene/appearance anchor.
- Rejected alternatives: predicting all eight latents including frame 0;
  predicting RGB directly; predicting raw StreamVGGT tokens.
- Impact: sampling must prepend `z0`; checkpoints with old eight-target layout
  are not resume-compatible.
- **2026-07-15 evidence against (diffusability measurement, mid-R1
  checkpoint, 64 eval clips):** the residual streams are spectrally far
  harder to diffuse than the absolute streams — z_geo residual: high-freq
  band 0.49 vs 0.24 absolute, effective rank 110 vs 45, kurtosis 6.5 vs
  3.2; z_tex residual: high-freq 0.70, eff rank 132. Subtracting z0
  removes the shared low-frequency content and whitens the target —
  exactly the anti-SSVAE profile. Combined with (a) the old Wan run's
  under-dispersion symptom and (b) MIRA / DINO-world / VGGT-World all
  predicting ABSOLUTE latents (MIRA conditions on clean past frames
  instead of subtracting them), the leading candidate fix when diffusion
  resumes is: **predict absolute normalized z_1..z_7 with z_0 as a
  clean-past condition (MIRA-style), not residuals.** Decide after R1
  completes + final-checkpoint rerun of diagnose_latent_diffusability.py.
  The latent cache contract (cond=z0, target=residuals) must NOT be
  rebuilt before this decision.

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

- 2026-07 probe-infrastructure bugs (FIXED in code; prior results affected):
  1. DDP forward bypass: E4/E5 (and E1-3 decoder path) called sub-modules
     via `.module` instead of the DDP wrapper, so gradient all-reduce never
     ran; multi-GPU probes degraded to independent single-GPU runs. All
     probes now route the full forward through a single DDP-wrapped core
     module (`ProbeCore` / `DualStreamCore`). E4 results were affected;
     E1-3 escaped only because h200_env.sh pinned 1 GPU.
  2. E5 oracle NaN: TextureEncoder last-conv zero-init + `std()`-based
     tex_reg had a NaN gradient at zero variance, poisoning all params via
     clip_grad_norm_ on step 1. Fixed: `sqrt(var+1e-6)` and zero-init now
     opt-in (`zero_init_last=False` default).
  3. Encoder checkpoint loading: several entry points did
     `load_state_dict(strict=False)` with no unwrap/verification — a wrapped
     checkpoint would silently load ZERO weights. All entry points that load
     the StreamVGGT encoder (probes, train_autoencoder, cache_compact_latents,
     train_i0_autoencoder, train_compact_diffusion, sample_compact_i0,
     inference_autoencoder, diagnose_* scripts) now use
     `utils/encoder_loader.load_encoder_checkpoint`, which unwraps containers
     and raises unless >=90% of keys match. E1/E2/E3 historical runs used the
     unverified loader; whether their encoder actually loaded must be
     confirmed on H200 (the loader now prints match counts).
  4. Temporal attention had no residual connection (attention output REPLACED
     features) in probe/dual-stream decoders AND in the production
     `models/compact_decoder.py` CompactDecoder used by E1-3 compressed mode,
     train_autoencoder, and train_compact_diffusion. All sites now use
     pre-norm residual. E1 (num_temporal_blocks=1), E5, and all historical
     CompactDecoder-based results affected. Old CompactDecoder checkpoints
     still load (same parameters) but decode differently through the fixed
     forward.
  5. Probe eval used a single clip (batch 1 of eval loader). Single-clip
     PSNR varies more than the 2-5 dB decision gates. All probes now
     evaluate over `--eval_clips` (default 32, `EVAL_CLIPS` in wrappers)
     and report psnr±std; eval datasets use `temporal_jitter=False` and
     `num_workers=0` so every eval pass (and every probe) scores the same
     deterministic clip windows.
- Probe loss configs are inconsistent across probes (E1 lambda_lpips=1.0
  vs E4/E5 0.1); PSNR comparisons across probes are not clean, and heavy
  LPIPS depresses PSNR. A pure L1+MSE E1 variant is needed to measure the
  true feature-space PSNR ceiling.
- Compact latent reconstruction is ~20 PSNR with grid textures. This caps
  every downstream generator: the latest Wan run shows `generated_std` below
  `target_std` and `velocity_mse` stuck at ~0.53, consistent with the
  generator learning a too-narrow distribution over an under-expressive
  latent. Reconstruction must reach PSNR >= 25 before further scale
  generation. (Note: under-dispersion can also come from flow-matching
  mean-reversion; re-check after reconstruction is fixed.)
- TextureEncoder legacy `pack_mode='avgpool'` packs scales via
  adaptive_avg_pool to 18x18 — average pooling low-passes exactly the high
  frequency z_tex must carry. `pack_mode='s2d'` (E5 R1 default) resizes
  input to 576=32x18 and packs losslessly via pixel_unshuffle; avgpool is
  kept only as the A/B control arm.
- Grid texture sources (PixelShuffle checkerboard, adaptive_avg_pool
  37->18 odd/even misalignment, VGGT patch-attention boundaries) are
  unverified and will be isolated by `probe_e3_grid_isolation.sh`. The
  non-integer 37->18 adaptive pool alternates kernel sizes and is itself a
  plausible periodic-artifact source; grid=19 or crop-to-36+stride2 avoids it.
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

0. Verify encoder checkpoint loading on H200 (one-time): run any probe and
   check the new `[encoder] ... matched N/M keys` line. If historical
   E1/E2/E3 ran with an unloaded (random) encoder, all ceiling conclusions
   reset.

1. Re-run E4 (both bottleneck modes) with the fixed DDP path and multi-clip
   eval to get trustworthy "DPT vs generic" and "compression cost" numbers.
   Also run one E1 variant with `--lambda_lpips 0 --lambda_mse 0.5` to
   measure the pure-PSNR feature ceiling.

2. Run the E5 dual-stream matrix on H200 (4 GPUs) — full matrix and gates
   in `scripts/h200/README.md` ("E5 run matrix"). Summary:

   - R1 `TEX_PACK=s2d TEX_REG_MODE=match_geo` — best guess: space-to-depth
     packing (avg-pool low-passes the high freq z_tex must carry; input
     resized 518->576 internally so all scales are integer multiples of
     grid 18) + SVG-style z_tex-stats-match-z_geo regularizer.
   - R2 avgpool + match_geo — isolates packing's contribution (only if R1
     passes).
   - R3 `TEX_MODE=zero` — geo-only floor.
   - R4 `TEX_MODE=tex_only` — proves z_geo is load-bearing (reviewer
     question #1; z_geo zeroed, compressor+encoder skipped).
   - R5 `FEAT_WEIGHT=0.5` — VGGT feature-consistency loss (MIRA P-DINO
     analogue), the no-GAN fidelity lift; run if R1 lands in 23-25.

   Dual gate before any cache rebuild: (a) oracle PSNR >= 25 over 32
   clips, no grid texture; (b) `diagnose_latent_diffusability.py` on the
   winning checkpoint shows z_tex spectra/eigenspectrum not materially
   worse than z_geo (SSVAE criteria; guards against MIRA's
   "sharper-but-undiffusable" failure mode). Historical note: last-k /
   multi-level mean aggregation was already tried in the legacy phase
   (docs/legacy/PROJECT.md: mean over levels DEGRADED to 18.2 dB vs 22.3
   single-level) — do not re-run it as-is; the E5 matrix supersedes it.

3. Benchmark alternative gate (2026-07 survey): GLD/RAE recover high PSNR
   from frozen features by training the decoder with L1+LPIPS+GAN. Before
   committing to dual-stream, a cheap E6 = E1 setup + adversarial decoder
   loss would test whether the "missing high-frequency" is recoverable
   decoder-side (single geo latent, no z_tex, generative decoder). Watch
   temporal flicker of hallucinated texture (temporal LPIPS across frames).

4. If E5 oracle passes the gate, wire production AE training to
   CompactCompressor + TextureEncoder + DualStreamDecoder (replace the
   single-stream GenerativeTokenizer path for reconstruction).

5. Only after reconstruction passes the gate, rebuild the latent cache
   (now both z_geo and z_tex) and resume Wan diffusion. Do NOT rebuild
   cache before the dual-stream contract is finalized by E5. When diffusion
   resumes, add geometry-consistency eval (re-encode generated video with
   frozen VGGT; reprojection error / depth consistency — see RPE/RVE from
   Geometry Forcing, arXiv:2507.07982) alongside velocity_mse.

6. When diffusion resumes, before any model-parallel full-QKV Wan run, do
   the cheap QKV-capacity control:

   ```bash
   TRAIN_QKV=0 bash scripts/scale/smoke_wan_compact.sh        # adapter only
   TRAIN_QKV=1 TRAIN_QKV_LAST_N=8 bash scripts/scale/smoke_wan_compact.sh
   ```

   If adapter-only ≈ last-4, QKV is not the bottleneck and full-QKV will
   not help. Expand to last-8/12 only if last-N improves monotonically.

## Related Work Map (2026-07 survey)

Direct novelty threats and must-cite baselines for the paper. Verified via
web survey 2026-07-13; re-check before submission.

### 2026-07-14 survey additions (actionable)

Decoder recipe (E6 / reconstruction gate):

- **MIRA** (arXiv:2607.05352, General Intuition/Kyutai/Epic, 2026-07): video
  world model on frozen DINOv3-L RAE latent WITH RGB decode. Codec recipe
  (their Table 9): frozen DINOv3-L, **aggregate blocks {11,13,15,17,19,21,23}**,
  linear bottleneck to 32ch, 2x2 spatial + 2x temporal downsample; space-time
  causal ViT decoder (1152w/28d); losses **L1(1.0) + LPIPS(1.0, 25% frames) +
  DINO-feature perceptual "P-DINO"(1.0, 25% frames), NO GAN** — adaptive
  gradient-norm weighting; AdamW lr 2e-4, batch 32, 250k steps, 8xH100.
  Key ablation: from-scratch encoder reconstructs sharper (PSNR 32.2 vs
  29.7) but generates far worse (gFID 22.5 vs 10.7) and drifts 1.7x more in
  rollout — frozen-feature smoothness is what stabilizes generation. This is
  (a) evidence GAN is optional if LPIPS+P-DINO are combined, (b) a direct
  partial novelty threat: temporal video + frozen encoder latent + RGB
  decode, though game-domain (Rocket League), DINOv3 not geometry-aware,
  and single-scene. Must cite.
- **RAEv2** (arXiv:2605.18324, Adobe/NYU, 2026-05): summing the **last k
  encoder layers** (not just final layer) greatly improves RAE
  reconstruction with a frozen encoder; 10x faster convergence. Cheap to
  try in our tokenizer: we already fuse levels [4,11,17,23]; test a
  last-k-sum variant (e.g. sum of last 6-8 aggregator levels) before adding
  capacity. Also: REPA is complementary to RAE (usable later for the
  generator).
- **DecQ** (arXiv:2605.22777) and **LV-RAE** (arXiv:2602.08620): 2026 RAE
  follow-ups closing the pixel-fidelity gap via a lightweight low-level
  pathway reading INTERMEDIATE VFM features (learned queries / shallow
  encoder for "local variations"). Same design family as our z_tex; cite as
  concurrent image-domain precedent. RAE-AR (arXiv:2604.01545) confirms
  RAE latents match VAE rFID but lag on PSNR/SSIM — our ~20 PSNR plateau is
  the expected frozen-feature result, not a bug.
- **SVG detail branch specifics** (arXiv:2510.15301 + SVG-T2I
  arXiv:2512.11749, open-sourced incl. autoencoder): residual encoder output
  is **normalized to match the batch mean/std of the frozen DINOv3 features
  before channel-concat** — that distribution alignment is their mechanism
  for keeping the detail stream from disturbing semantic structure. Adopt
  for z_tex (align z_tex stats to z_geo stats instead of/in addition to the
  N(0,1) tex_reg). SVG-T2I also found the residual encoder becomes
  unnecessary at higher input resolution — the frozen features already
  carry detail; relevant to whether 518px VGGT features are being
  under-exploited by our 37->18 compression.

Latent diffusability (before rebuilding the cache):

- **SSVAE / latent spectral biasing** (arXiv:2512.05394, code
  zai-org/SSVAE): two properties predict video-latent diffusability —
  low-frequency-biased spatio-temporal spectrum and a channel eigenspectrum
  dominated by few modes; two cheap regularizers (local correlation +
  latent masked reconstruction) give 3x faster T2V convergence. Directly
  applicable to tokenizer training; also gives us MEASUREMENTS to run on
  our current latent (spectrum + eigenspectrum) to quantify "diffusability"
  before/after fixes.
- **Diffusing in the Right Space** (arXiv:2606.03578) and
  **Prior-Aligned Autoencoders** (arXiv:2605.07915): systematic studies of
  what makes latents diffusion-friendly; use as citation cover for the
  latent-normalization contract and for choosing regularization.
- Under-dispersion note: MIRA/DINO-world both diffuse in frozen-feature
  space successfully with x0/clean-target-style objectives; combined with
  VGGT-World's velocity collapse in 1024-dim VGGT space, the evidence now
  strongly favors **switching our Wan/DiT objective to x0(z)-prediction**
  when diffusion resumes, keeping flow-matching sampling.

Video decoder artifacts (E3 context):

- Production video VAEs (**Wan, HunyuanVideo, CogVideoX**) all use
  **interpolate/resize + stride-1 conv** upsampling, NOT PixelShuffle;
  LTX-Video uses PixelShuffle and has documented grid-artifact issues
  (Lightricks/LTX-2#202) plus a finetune released specifically to reduce
  checkerboard. If E3 confirms PixelShuffle as a grid source, switching to
  resize-conv matches industry practice; ICNR init or fixed post-blur
  (VFM-VAE style) are the fallback if PixelShuffle must stay for speed.
- **FeatUp** (ICLR 2024) + successors (LoftUp, UPLiFT, ViT-Up
  arXiv:2606.14024): learned ViT-feature upsampling to break the 14px
  patch grid; candidate fix if E3 shows patch-boundary artifacts survive
  the resize-conv switch.

Metrics/eval (for the paper):

- **GeCo** (arXiv:2512.22274): differentiable geometric-consistency metric
  (flow rigidity + depth-reprojection cues); newest standard alongside
  RPE/RVE, MEt3R, Sampson error (GeoFlow arXiv:2605.18365 reports
  MEt3R+Sampson splits; GeoVideo arXiv:2512.03453 reports MVCS+RPE on
  DL3DV). Plan: report RPE/RVE + Sampson + GeCo.
- **minWM** (arXiv:2605.30263) negative result: Wan2.1 trained on
  SpatialVID did not reach reliable camera-controllable generation —
  suspected MegaSaM pose noise. Not our task (we don't do explicit camera
  control), but cite when justifying SpatialVID and expect reviewers to ask.
- SpatialVID is now CVPR 2026 (cite the CVPR version).

Closest existing work (novelty threats):

- **GLD — "Repurposing Geometric Foundation Models for Multi-view Diffusion"**
  (arXiv:2603.22275): geometric-foundation-model feature space as the
  diffusion latent for **multi-view NVS** (static scenes). Decoder trained
  with L1+LPIPS+GAN (adaptive weight, Taming-Transformers style); reports
  35.41 dB PSNR reconstruction (vs SD-VAE 34.53) and >4.4x training speedup
  vs VAE latent. Our differentiation is exactly the one already stated in
  AGENTS.md: video = temporal motion + disocclusion, not viewpoint change.
  This is now the single most important paper to cite and compare against.
- **Gen3R** (arXiv:2601.04090): recasts VGGT as an asymmetric geometry VAE;
  adapter aligns VGGT tokens with a **pretrained video diffusion (Wan /
  CogVideoX) appearance latent**; jointly generates RGB + geometry. Key
  reported lesson: naively compressing VGGT tokens is insufficient — their
  distribution differs strongly from appearance latents; alignment is
  required. Very close in spirit to our Wan adapter path.
- **VGGT-World** (arXiv:2603.12655): frozen VGGT tokens as autoregressive
  world state (geometry forecasting only, no RGB decode). Directly relevant
  technical caveat: **velocity-prediction flow matching collapsed in the
  1024-dim VGGT feature space**; they needed clean-target (z/x0-prediction)
  parameterization + flow-forcing curriculum. Our velocity-prediction
  under-dispersion symptom may be the same phenomenon.
- **Geometry Forcing** (arXiv:2507.07982, ICLR 2026): keeps VAE latent, adds
  angular+scale alignment of diffusion internals to VGGT features. FVD
  364->243 on RE10K. This is the strong REPA-style baseline our "replace the
  representation" thesis must beat or complement. Their RPE / RVE metrics
  are the community standard we should adopt for geometric consistency.

Representation-space diffusion (image domain, establishes feasibility):

- **RAE** (arXiv:2510.11690, ICLR 2026): frozen DINOv2/SigLIP + ViT decoder
  trained with LPIPS+L1+**GAN**; rFID beats SD-VAE; DiT in that latent hits
  FID 1.13-1.51 on ImageNet. Frozen encoders are viable latent spaces when
  the decoder is trained generatively.
- **SVG** (arXiv:2510.15301, Kling): frozen DINOv3 + **lightweight residual
  branch for fine detail** — architecturally the closest published analogue
  of our dual-stream (z_geo + z_tex) design; cite as precedent that a
  semantic stream + detail stream factorization works for diffusion.
- **l-DeTok** (arXiv:2507.15856) / **DiTo** (arXiv:2501.18593): latent
  denoising / diffusion-decoder tokenizers; relevant if we later make the
  decoder generative.

Factorized video latents (precedent for dual-stream in video): CMD
(arXiv:2403.14148, content frame + motion latent), VidTwin (arXiv:2412.17726,
structure/dynamics), Video-LaVIT, CoordTok. None uses a geometry foundation
model — that remains our gap.

Geometric-consistency evaluation (for the paper's metrics section):
RPE/RVE (Geometry Forcing), DROID-SLAM-based RPE (WorldMark, MultiWorld),
DA3-based epipolar+reprojection with non-adjacent revisit pairs (MBench,
arXiv:2606.00793; WBench, arXiv:2605.25874). Plan: re-encode generated
videos with frozen VGGT/DA3 and report reprojection error + revisit error.

Positioning summary: nobody has yet shipped "geometry-foundation-model
latent space **replacing the VAE** for **temporal video** diffusion with
RGB decode" — GLD covers multi-view static, Gen3R keeps the Wan appearance
latent and aligns to it, VGGT-World forecasts geometry without RGB. The
lane is open but narrow and moving fast (GLD is 2026-03, Gen3R 2026-01).
2026-07 update: **MIRA** (2607.05352) now does temporal video diffusion in
a frozen-encoder latent WITH RGB decode — but with DINOv3 (semantic, not
geometry-aware), on a single game domain, with no geometry outputs or
geometric-consistency claims. Our lane narrows to "**geometry** foundation
model latent for **real-world** video with geometric-consistency
evaluation"; the geometry axis is now the load-bearing differentiator and
the RPE/RVE/GeCo numbers must beat semantic-encoder baselines (a
DINOv3-latent ablation arm becomes near-mandatory for the paper).

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
