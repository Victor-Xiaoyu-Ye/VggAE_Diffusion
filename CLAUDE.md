# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## First reads

`AGENTS.md` and `PROJECT_CONTEXT.md` are the living project-state documents (current goal, phase, decisions, known issues, next tasks). Read them before starting any task — they change often and override anything stale here. `scripts/h200/README.md` documents the active probe plan; `TOKEN_STATS.md` documents the latent normalization contract.

## What this project is

Research code for geometry-aware video generation: frozen StreamVGGT (a streaming VGGT geometry foundation model, vendored under `streamvggt/`) replaces the VAE as the representation space for video diffusion. The pipeline:

```
SpatialVID video (8 frames, 518×518)
  -> frozen StreamVGGT aggregator (24 levels of tokens [B,S,N,2048], causal global attention)
  -> tokenizer/compressor over DPT levels [4,11,17,23]
  -> compact latent z [T, 18, 18, 256]
  -> I0-residual diffusion: cond = z0, target[t] = z[t+1] - z0
  -> generator (from-scratch CompactLatentDiT, or Wan2.1-I2V-14B via WanCompactAdapter)
  -> decoder -> RGB video
```

Frame 0 is observed condition only, never a diffusion target. Flow matching with velocity prediction. Latents are normalized per the contract in `TOKEN_STATS.md`.

Two execution environments, neither is this machine:
- **H200 box** (`/home/yexiaoyu/...`, conda env `rae`): reconstruction probes under `scripts/h200/`.
- **ModelArts Ascend 910B cluster** (branch `ascend-910b`): scale pipeline under `scripts/scale/`, data/checkpoints on OBS via MoXing. NPU/HCCL/OBS behavior cannot be validated locally — never claim it passed from static checks.

## Commands

There is no test suite or linter. Required static checks before push:

```bash
python -m py_compile <changed .py files>
bash -n <changed .sh files>
git diff --check
```

Active reconstruction probes (H200):

```bash
bash scripts/h200/probe_e1_raw_recon.sh      # raw 4-level features -> generic decoder
bash scripts/h200/probe_e2_per_level.sh      # single-level probes
bash scripts/h200/probe_e3_grid_isolation.sh # grid-artifact isolation
bash scripts/h200/probe_e4_dpt_recon.sh      # DPT decoder / compact bottleneck
bash scripts/h200/probe_e5_texture_recon.sh  # dual-stream (TEX_MODE=oracle|zero)
```

Scale pipeline (ModelArts), in order — do not rebuild stages whose artifacts already exist:

```bash
bash scripts/scale/00_train_geometry_autoencoder.sh
bash scripts/scale/01_train_i0_decoder.sh
bash scripts/scale/02_shard_metadata.sh
bash scripts/scale/03_cache_latents.sh        # + 03_cache_eval_latents.sh
bash scripts/scale/04_merge_latent_cache.sh
bash scripts/scale/smoke_wan_compact.sh       # always smoke before the real run
bash scripts/scale/05_train_wan_compact.sh
bash scripts/scale/06_sample_wan_compact.sh
```

Shell scripts source `scripts/h200/h200_env.sh` or `scripts/spatialvid_config.sh` + `scripts/lib/*.sh` for paths/env. They must work from any working directory (`"$PYTHON_BIN" "$PROJECT/script.py"` style) — preserve that when editing. If a trainer gains an argparse flag, update its wrapper shell script in the same change.

## Code map

- `probe_feature_recon.py` (E1/E2/E3), `probe_e4_dpt_recon.py`, `probe_e5_texture_recon.py` — self-contained reconstruction probes.
- `models/generative_tokenizer.py` — production tokenizer (level fusion, 37→18 spatial compression, optional TemporalMixer). `models/dpt_latent_decoder.py` — CompactCompressor + DPT-style decoder. `models/texture_encoder.py` + `models/dual_stream_decoder.py` — E5 dual-stream (z_geo + z_tex, no RGB skip).
- `models/wan_compact_adapter.py` — adapts Wan2.1-I2V-14B (vendored in `Wan2.1/`) to compact residual latents. Default finetunes adapters + last 4 QKV blocks (`TRAIN_QKV_LAST_N=4`); full-14B QKV OOMs under replicated DDP on 60 GiB NPUs.
- `models/flow_matching.py` — velocity-prediction flow matching.
- `cache_compact_latents.py` / `merge_latent_cache.py` / `data/latent_shard_dataset.py` — tar-shard latent cache on OBS with `_SUCCESS` markers, FP64 CPU moments, and resume-contract signatures. The cache is persistent data, not a run output; it is disposable only until the tokenizer is finalized.
- `data/token_utils.py` — special-token stripping, level selection, normalization helpers shared by all encode paths.
- `utils/distributed.py`, `utils/training.py` (EMA — shadow-dict only, no apply/restore API), `utils/video_io.py` (square-resizes frames to 518×518), `utils/moxing_io.py` (OBS transfer).
- `train_*.py` at repo root — generations of trainers; check `PROJECT_CONTEXT.md` for which are active vs legacy before touching one. `docs/legacy/` and `scripts/legacy/` are dead.

## Conventions and pitfalls

- After any substantial task, end your answer with the `Context Update` block defined in `AGENTS.md` (新增知识 / 新增约定 / 新增决策 / 当前状态更新 / 建议写入项目文档), and update `PROJECT_CONTEXT.md` in the same commit if architecture, training order, paths, resume behavior, failure modes, or accepted experiments changed.
- StreamVGGT applies ImageNet normalization internally and its global attention is causal — frame-0 tokens are identical whether encoded alone or in a sequence. The tokenizer's TemporalMixer breaks this (k=3 conv mixes frame 1 into frame 0), which is why the I0 cache encodes frame 0 separately; keep the mixer disabled for caching.
- Checkpoints may be raw state_dicts or wrapped in `{'model_state_dict': ...}` — always unwrap and verify matched keys; several older loaders use `strict=False` without checking, which fails silently.
- Trainers wrap modules in DDP but must call the DDP wrapper in forward, not `.module` — bypassing it silently skips gradient sync.
- Do not commit checkpoints, generated samples, logs, caches, or `outputs/`. Every active trainer writes `checkpoint_latest.pt`, periodic checkpoints, `metrics.jsonl`, `tb/`, `samples/`, `logs/`.
