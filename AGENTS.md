# Agent Handoff

Read this file first. It defines the active project context, current goals,
rules, and safe next steps for future agents.

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

Generation experiments are paused. The compact latent reconstructs at ~20
PSNR with grid textures, which caps every generator. We are running
diagnostic probes on the 4-card H200 machine to decide the architecture:

```bash
source scripts/h200/h200_env.sh
bash scripts/h200/probe_e1_raw_recon.sh    # raw feature PSNR ceiling
bash scripts/h200/probe_e2_per_level.sh    # per-level info content
bash scripts/h200/probe_e3_grid_isolation.sh  # grid texture source
```

Decision rule (see `PROJECT_CONTEXT.md` `Research Risks`):
- E1 PSNR >= 28 + E2 shallow better -> two-stream latent (z_geo + z_app).
- E1 PSNR >= 28 + E2 equal -> single stream + capacity increase.
- E1 PSNR <= 23 -> single geometry latent + decoder hallucinates RGB
  high-freq via perceptual + optional adversarial training.

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
