# Project Context

This document is the durable handoff for VggAE-Diffusion. Keep it current when
goals, architecture, training order, paths, or important decisions change.

## Goals

- Use StreamVGGT as a frozen geometry-aware teacher/encoder.
- Learn a compact latent space that is reconstructable, cacheable, and easier
  to diffuse than raw StreamVGGT tokens.
- Use the first RGB frame `I0` to provide appearance and high-frequency detail.
- Generate future geometry-aware latent residuals, then decode them to RGB with
  an I0-conditioned decoder.
- Prove that StreamVGGT compact feature space improves geometry-aware video
  generation when paired with a strong pretrained video prior such as Wan.

## Current State

- Active branch: `ascend-910b`.
- Active large-scale dataset: SpatialVID-HQ on OBS.
- Active local 10K dataset path:
  `/public2/LiZhen/yexiaoyu/dataset/spatial-vid-hq-oft`.
- Active scale dataset path:
  `obs://yw-ads-training-gy1/data/external/personal/g00833899/y50046448/dataset/SpatialVID-HQ`.
- Active persistent owner OBS root:
  `obs://yw-ads-training-gy1/data/external/personal/g00833899/y50046448`.
- Active scale latent cache version:
  `vggae_streamvggt_256x18_v1`.
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

- From-scratch Compact DiT still has ghosting despite larger 768-dim runs.
- I0 decoder reconstructs recognizable structure but loses high-frequency
  details; generated latent decode can blur or duplicate objects.
- Full 14B QKV DDP OOMs on 60 GiB 910B cards.
- MoXing/OBS logs can show noisy multiprocessing logging rollover errors even
  when transfer succeeds.
- `_SUCCESS` missing in latent cache partitions means cache generation or
  finalization is incomplete; merge should not bypass this guard.
- Historical tokenizer temporal mixer caused I0 contract drift; active scale
  contract must keep tokenizer/cache settings aligned.

## Next Tasks

1. Rerun Wan smoke with current defaults:

   ```bash
   bash scripts/scale/smoke_wan_compact.sh
   ```

2. If smoke passes, start full Wan run:

   ```bash
   bash scripts/scale/05_train_wan_compact.sh
   ```

3. Compare against Compact DiT using:
   - `eval/velocity_mse`
   - generated/target latent std
   - RGB preview ghosting
   - temporal consistency
   - sample videos from `06_sample_wan_compact.sh`

4. If last-4 QKV improves but remains underfit, test:

   ```bash
   TRAIN_QKV_LAST_N=8 bash scripts/scale/smoke_wan_compact.sh
   TRAIN_QKV_LAST_N=8 bash scripts/scale/05_train_wan_compact.sh
   ```

5. Do not rebuild AE/I0/cache unless the latent representation contract changes.

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
