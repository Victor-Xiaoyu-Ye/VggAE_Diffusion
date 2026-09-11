# Agent Handoff

Read this file first. It defines the active project context, current goals,
rules, and safe next steps for future agents.

## Priority Update (2026-09-09)

Domain v1 cache failed onmissingHQ object.42/43/44 nowusefrozenv2 selection
andnamespaces afterfullHQ objectlisting (360610present vs365362CSV). No raw
downloads. Preflight2304objects/arm beforeAE,zero replacement/failures retained.
Rerun43 ARM=single RESUME=0;do notresume oldv1cache/training. Seeupdatedrunbook.

NEW stages42/43/44: docs/DOMAIN_UNIFORM_RUNBOOK.md. FrozenCSV single/mixed
2048clip cohorts,4windows each,fullHQ root;freshuniform6000 on48NPUs.43 is
one-command cache+train entry,ARM=single default. No local raw downloads.
Strict CSV checksums/no replacement/zero failures;oldcache defaultcompatible.
24CPUtests passed;newNPU run pending. Source-level leakage still unverified.

NEW stage40_train_window_memory64_uniform.sh: same stage37 recipe except noise
law changes shifted-logit-normal3 -> unwarpedUniform;fresh6000,never resume old
memory checkpoint. Loss floor/weight unchanged, so effective bin contributions
also change; monitor existing sample/loss fractions. Stage41 runs same stage39
trajectory after uniformfinal6000. Read docs/WINDOW_MEMORY_UNIFORM_RUNBOOK.md.
Old flow identities/default draws remain compatible. New quality/NPU pending.
Stage39 now completed3584rows/48ranks/6exit0: memory x0L1 .01107@u.375 ->
.03714final, but heldout bad already at u1; do not promise generalization.

NEW stage39_diagnose_window_trajectory.sh: read docs/WINDOW_TRAJECTORY_RUNBOOK.md.
Read-only memory64 checkpoint_final step6000,16memory+16heldout,online/EMA,
seeds101/211. Production Euler observer,9nodes state/x0/GTprobe plusfinal,
3584metrics. Match original subset fingerprint and heldout noise-index64offset.
All metrics/latents before small PNG-only previews. No NPU validation yet.

LATEST RESULTS: memory64 completed6000 (main receipt), all7680global evalrows;
EMA memoryL1 .03782,heldout .13757 vsAE. Partial visual evidence shows learning,
not heldout baseline. Pure-noise one-call x0 MSE .01574 vs fullflow .12438;
next priority read-only trajectory/single-call decode. Subspace download only
94/384,nohigh/PCApairs: do not select whitening/frequency recipe yet.
See newest PROJECT_CONTEXT and projectless memory64_subspace_results_review.md.

2026-09-10 IMPLEMENTED stages37/38: read docs/WINDOW_MEMORY_SUBSPACE_RUNBOOK.md.
Stage37 trains64 train-only windows with same t2v2 legacy/noaux shift3 chain,
balanced global sample stream, full resume, all64 previews/four seeds; separate
heldout16 RAW AE gate/eval. Best checkpoint uses MEMORIZATION score explicitly.
Train cache lacksRAW; memory RGB comparisons useAE, not GT RGB. Stage38 frozen
EMA2500 oracle repairs frequency/PCA components with matched residual MSE;
PCA uses64train videos only. Neither stage has cluster/quality validation yet.

2026-09-10 RAE ROOT-CAUSE REVIEW (supersedes native-only next-step advice):
User asks why MIRA works and requests a defensible R7 generation baseline.
Read docs/RAE_VIDEO_ROOT_CAUSE_REVIEW_2026-09-10.md. Do not claim scratch RAE
is disproven or launch another aux sweep. Prioritize same-pipeline small-set
pure-noise generation fit and frozen frequency/PCA oracle diagnosis, then
narrow-domain heldout I2V. These are proposals, not implemented launchers.
New CPU audit on16 heldout seed42 latents: channel covariance PR rank17.32,
top8 energy50.29%; high spatial-frequency error energy50.21%. Descriptive,
not evidence to blindly whiten/truncate; train-derived statistics required.
DiT width768>C192; RAE width-bound argument does not directly explain failure.
Legacy temporal codec is window-dependent; do not convert to AR by changing
norm or slicing latent slots. Keep good AE/shift3/aux8 reference artifacts.
Native I2V remains a quality reference, not a substitute for this diagnosis.

2026-09-10 QUALITY REDESIGN REQUEST: user rejects further small recipe sweeps.
Recommend native pretrained I2V/VAE/input-output interfaces as quality-first
mainline, geometry residual adapters/relational supervision, richer windows,
strict first-frame/no-future-condition contract. Original R7 branch retained.
Design only, no native NPU path implemented. Do not launch another aux sweep.

2026-09-10 COMPLETE AUX AUDIT: aux6/aux8 both6000,48 completed ranks,
6 node exits0 and6 synced receipts each. Aux6 not promoted. Aux8 bestEMA3500
RAW .109237 vs no-aux best2500 .109971 (only0.67%); structure still fails.
Retain both reference checkpoints, verify actual artifact step/EMA first.
Stop auxiliary-layer sweeps; recommend native pretrained I2V quality baseline
as next route evaluation, not implemented by audit. Read newest context.

PREPARED NEXT ARM stage36_train_window_aux8_x0.sh: move auxiliary block6->8
only, still weight0.5/shift3/6000 steps. Keep current aux6 running unchanged.
EMA500 latent improves2% but RAW L1 worsens0.5%, too early to select winner.
Fresh aux8 namespace; never resume aux6 into aux8. Compare full runs later.

NEW TRAINING ENTRY stage35_train_window_aux_x0.sh: fresh shift3 control plus
block6 clean-R7 auxiliary head weight0.5, unchanged6000-step recipe/cache/AE.
Read docs/WINDOW_AUX_X0_RUNBOOK.md. Sampling main head only; full resume only
within same auxiliary contract. User accepted partial decoder audit as enough
to choose this arm; no further snapshot requirement blocks implementation.

NEXT READ-ONLY ENTRY IMPLEMENTED: stage34_diagnose_window_decoder.sh,
frozen decoder directed/random matched-RMS perturbations on shift3 EMA2500.
Read docs/WINDOW_DECODER_DIAGNOSTICS_RUNBOOK.md. Default best_reconstruction
must internally be step2500; no silent latest fallback. Stage29 dual IO reused.
CPU validation available; real384-row NPU matrix remains pending.

SHIFT3 COMPLETED:6000 steps, main-node exit0/synced receipts; local download
does not include full workers receipts or training checkpoints. EMA RAW L1
improves .118928->.112278 at6000, but best shift3 is2500 (.109971) and
previews still deform. Normalized scale is broadly healthy; high-noise x0
error improves while low-noise worsens. Prefer frozen decoder perturbation
audit with verified EMA2500; do not extend/increase shift by default.
This result supersedes shift3-pending statements below. No new recipe launched.

Literature refreshed after shift3 launch: read docs/RELATED_WORK_REVIEW_2026-09-09.md.
Clean AE replay does not exclude decoder amplification of generated errors.
Prioritize frozen perturbation audit; auxiliary clean-latent supervision is a
later controlled candidate. No running recipe changed. VideoWeave/RAEv2/LV-RAE
are relevant; no 'VGGT replacement alone' novelty claim.

CURRENT TRAINING ENTRY: stage33_train_window_shift3.sh. Single-factor fresh
time_shift3 arm, unchanged validated legacy AE/cache and6000-step recipe,
plus noise-bin/latent-scale telemetry. Read docs/WINDOW_SHIFT3_RUNBOOK.md.
Full resume applies only within that arm; real quality remains unverified.

Stage32 has now COMPLETED:320 unique samples,48 ranks completed,6 node exits0.
Sampler128 gives only1.21% RAW L1 gain at2x sampling cost; conditioning works
but train previews also fail. Read newest PROJECT_CONTEXT diagnostic results.
Do not repeat stage32 by default or treat its NPU validation as still pending.

NEXT ENTRY: stage32 diagnose_window_diffusion.sh, read-only EMA6000 sampler /
train-heldout / anchor ablations, fresh diagnostics namespace. Read
docs/WINDOW_DIAGNOSTICS_RUNBOOK.md. No new training recipe is promoted.

LATEST CLUSTER RESULT: r7_window_t2v2_legacy_x0_diag_v2 completed6000 steps,
48-NPU BF16, exit0. Legacy AE replay recovered24.48933dB on the same16 clips
as v1; compatibility repair is validated. Generation still has blur/deformation
and late plateau (EMA RAW L1 .118928 vs RGB-copy .098126). Read newest
PROJECT_CONTEXT entry. Earlier repaired-AE-pending statements below are stale.
Dual-destination publication receipt reports success; downloaded logs do not
include training checkpoints, so full resume/remote bytes remain unverified.
No new training recipe is implemented by this audit; do not promise video quality.

CRITICAL UPDATE AFTER CLUSTER RUN: v1 t2v2 AE replay is degraded (18.73dB).
Historical t2v2 uses cross-time GroupNorm; current framewise norm silently loads
the same state keys but does not reproduce the AE. The fix is explicit
window_codec_runtime, legacy mode for t2v2, new v2 cache/output namespaces, and
stage31 same-RAW re-encoding A/B audit before stage30 caches/trains. Read the
top of docs/WINDOW_DIFFUSION_RUNBOOK.md. Old v1 checkpoints/caches must not be
full-resumed into v2. Real repaired-AE quality remains to be verified on NPU;
default audit and trainer PSNR gates stop degraded AE before training updates.

Implementation now available: independent `train_r7_window_diffusion.py` with
`models/r7_window_dit.py`, stages 28/29/30 and `docs/WINDOW_DIFFUSION_RUNBOOK.md`.
This supersedes historical launch suggestions below. It trains the full future
window from a frozen historical AE, starting with t2v2 / t1geo112 controls.
Stage 30 is the current entry, NOT 25/26/27. CPU synthetic resume and incremental
two-destination I/O tests passed; real NPU/OBS/AE replay/quality remain untested.
Respect user's latest requirement: dual reads/writes, intermediate artifacts,
DI throughput, full training-state resume. Do not claim reasonable videos yet.

Latest user direction: use the already trained ~25 PSNR AE variants as frozen
reconstruction baselines and design diffusion for this project's setting. Read
`docs/AE_BASELINES_AND_DIFFUSION_DESIGN_2026-09-08.md` first. V-RAE is only an
idea reference, not a code base or task contract. Neither continued AE redesign
nor a 25/26 n1 ladder is the default next task. Peak-PSNR log rows are not proof
of checkpoint_best identity; verify actual artifacts before cache/training.

USER CORRECTION: stages 25/26 are unfinished diagnostic scripts, not a validated
training baseline for the intended video diffusion. Do not launch or extend them
as the default research trainer. Stage 27 now performs read-only sampler checks
only and rejects RUN_N1=1. Selectively reuse audited formulas, not their model,
single-pair ladder or training objectives as a whole. Earlier 27->26 guidance is
withdrawn. A new training entry must explicitly justify its source components.

Historical diagnostic phase (not the current training launch):
`docs/R7_SAMPLER_VALIDATION_RUNBOOK.md` and stage 27 for the new oracle/deterministic
sampler checks. CPU checks are available;
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
