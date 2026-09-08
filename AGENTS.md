# Agent Handoff

Read this file first. It defines the active project context, current goals,
rules, and safe next steps for future agents.

## Priority Update (2026-09-08)

USER CORRECTION: stages 25/26 are unfinished diagnostic scripts, not a validated
training baseline for the intended video diffusion. Do not launch or extend them
as the default research trainer. Stage 27 now performs read-only sampler checks
only and rejects RUN_N1=1. Selectively reuse audited formulas, not their model,
single-pair ladder or training objectives as a whole. Earlier 27->26 guidance is
withdrawn. A new training entry must explicitly justify its source components.

The user has requested runnable validation. Start with
`docs/R7_SAMPLER_VALIDATION_RUNBOOK.md` and stage 27 for the new oracle/deterministic
sampler checks followed by bounded n1 diagnostics. CPU checks are available;
actual NPU/OBS and quality results remain pending.

The user now explicitly permits both R7/VGGT representation replacement and
native-video-VAE geometry-guided routes, with generation quality taking priority.
Resources remain 48 Ascend 910B NPUs on ModelArts with the existing OBS/wrapper
contracts. Read `docs/QUALITY_FIRST_AUDIT_2026-09-08.md` and the latest entry in
`PROJECT_CONTEXT.md` before the historical phase/architecture below.
Native I2V plus geometry supervision is the recommended next baseline, not a
completed implementation or measured NPU result. Preserve all old artifacts and
gates; no new model or production run has been promoted.

For the subsequent research-position review, also read
`docs/RESEARCH_POSITION_AND_TRAINING_2026-09-08.md` and the newest context entry.
VideoRAE/V-RAE and geometry-video prior work narrow the novelty claim. Proposed
geometry-relation preservation experiments remain unvalidated; do not treat the
proposal as an adopted architecture or advertise encoder replacement as novel.

## Current Goal

Build and validate geometry-aware video generation using frozen StreamVGGT
features as the representation space and a Wan-initialized video generator as
the strongest current generator baseline.

The active research claim is:

```text
Geometric foundation models (VGGT / StreamVGGT) can serve as the
representation space for video diffusion, replacing the VAE. StreamVGGT
compact latents provide geometry-aware structure; a Wan-pretrained video
prior supplies spatiotemporal dynamics. Adapting Wan to StreamVGGT compact
residual latents should reduce ghosting and improve geometry consistency
compared with from-scratch Compact DiT.
```

This is distinct from "Repurposing Geometric Foundation Models for
Multi-view Diffusion": video has temporal motion + disocclusion, not
viewpoint change of a static scene.

## Current Phase: Reconstruction-First (H200)

Generation experiments are paused. E1–E4 showed frozen StreamVGGT features
plateau at ~20 PSNR on SpatialVID (DPT decoder does not break the ceiling).
Appearance high-frequency must come from an explicit TextureEncoder, not
from VGGT shallow levels and not from decoder-side hallucination first.

Active probe:

```bash
bash scripts/h200/probe_e5_texture_recon.sh                 # oracle z_tex
TEX_MODE=zero bash scripts/h200/probe_e5_texture_recon.sh   # geo-only ablation
```

Contract: decoder reconstructs from `(z_geo, z_tex)` only — no RGB skip —
so both streams remain diffusion targets. Gate: oracle PSNR >= 25.

Hard gate: reconstruction PSNR < 25 (no grid texture) blocks further scale
diffusion. The 1.46M-clip scale cache is disposable until the tokenizer is
finalized.

## Current Branch And Machine

- Main active branch: `ascend-910b`.
- Local repo path: `/home/yexiaoyu/work/VggAE-Diffusion`.
- Local Python on this machine: `/home/yexiaoyu/miniconda3/envs/rae/bin/python`.
- Scale cluster target: ModelArts Ascend 910B, usually multi-node x 8 NPU.
- Do not assume local `/cache` content persists across ModelArts jobs.

## First Files To Read

1. `PROJECT_CONTEXT.md`: durable project knowledge, architecture, decisions,
   conventions, known issues, research risks, and next tasks.
2. `scripts/h200/README.md`: active reconstruction-first probe plan (H200).
3. `scripts/spatialvid_config.sh`: active paths and persistent OBS layout.
4. `scripts/scale/README.md`: runnable scale-stage command order (paused
   until reconstruction passes the gate).
5. `TOKEN_STATS.md`: latent normalization contract.

Avoid starting from legacy scripts or old output markdown files.

## Active Architecture

```text
SpatialVID video
  -> frozen StreamVGGT
  -> GenerativeTokenizer
  -> compact geometry latent z, shape [T,18,18,256]
  -> I0 residual latent cache
  -> generator predicts seven future residual latents
  -> I0ConditionalDecoder
  -> RGB video
```

The observed first frame is condition only. Diffusion target is:

```text
cond = z0
target[t] = z[t+1] - z0, t=0..6
```

## Active Generator Experiments

- Baseline: from-scratch cached `CompactLatentDiT`.
- Current priority: `WanCompactAdapter` with `Wan2.1-I2V-14B-480P`.
- Wan default: train adapters, time path, modulation, and last 4 self-attention
  QKV blocks:

```bash
TRAIN_QKV=1
TRAIN_QKV_LAST_N=4
```

Do not default to full 14B QKV under replicated DDP. It already OOMed during
DDP reducer initialization on 60 GiB NPU cards with 3.39B trainable params.

## Current Scale Run Order

If AE, I0 decoder, and latent cache already exist, do not rebuild them.
Continue from cached generator experiments:

```bash
bash scripts/scale/smoke_wan_compact.sh
bash scripts/scale/05_train_wan_compact.sh
bash scripts/scale/06_sample_wan_compact.sh
```

Full pipeline order from scratch:

```bash
bash scripts/scale/00_train_geometry_autoencoder.sh
bash scripts/scale/01_train_i0_decoder.sh
bash scripts/scale/02_shard_metadata.sh
bash scripts/scale/03_cache_eval_latents.sh
bash scripts/scale/03_cache_latents.sh
bash scripts/scale/04_merge_latent_cache.sh
bash scripts/scale/smoke_wan_compact.sh
bash scripts/scale/05_train_wan_compact.sh
bash scripts/scale/06_sample_wan_compact.sh
```

## Output And Resume Rules

- Training outputs go to `$OUTPUT_URL` on ModelArts and are mirrored to the
  persistent owner OBS root configured in `scripts/spatialvid_config.sh`.
- Latent cache is persistent data, not a run output:

```text
obs://yw-ads-training-gy1/data/external/personal/g00833899/y50046448/cache_latents/
```

- Every active trainer should write:
  - `checkpoint_latest.pt`
  - periodic checkpoints
  - `metrics.jsonl`
  - TensorBoard `tb/`
  - previews in `samples/`
  - logs under `logs/`

## Coding Rules

- Preserve shell-wrapper execution style:
  `"$PYTHON_BIN" "$PROJECT/scripts/..."` must work from any working directory.
- Do not assume scripts are launched from repo root.
- Use `rg`/`rg --files` for search.
- Use `apply_patch` for manual edits.
- Do not commit checkpoints, generated samples, logs, caches, or `outputs/`.
- If a training script adds an argparse flag, update the corresponding shell
  script and run shell/Python static checks.

## Required Checks Before Push

For code/script changes, run the relevant subset:

```bash
/home/yexiaoyu/miniconda3/envs/rae/bin/python -m py_compile <changed .py files>
bash -n <changed .sh files>
git diff --check
git status --short --branch
```

NPU forward, HCCL, MoXing, and OBS streaming must still be validated on the
cluster. Do not claim they passed from local static checks.

## Context Update Format

After any substantial task, include this in the final answer:

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

If the task changes architecture, training order, paths, resume behavior,
failure modes, or accepted experiments, update `PROJECT_CONTEXT.md` in the same
commit.
