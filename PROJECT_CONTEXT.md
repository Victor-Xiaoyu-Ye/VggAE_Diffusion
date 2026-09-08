# Project Context

This document is the durable handoff for VggAE-Diffusion. Keep it current when
goals, architecture, training order, paths, or important decisions change.

## Accepted AE Baselines and Diffusion Design (2026-09-08)

Repair implemented after the cluster audit: utils/window_codec.py explicitly
selects legacy cross-time or framewise normalization using the historical
F.group_norm computation; cache metadata now includes window_codec_runtime,
and trainer validates/matches it before loading codec. t2v2 defaults to legacy;
other supported variants to framewise. New v2 cache/output namespaces prevent
silent reuse of the degraded v1 representation. No old AE weight is overwritten.
Stage30 invokes new31 for t2v2: reconstruct the same16 RAW clips from the old
eval cache by re-encoding BOTH normalization modes, same encoder/AE signatures.
Default selected legacy PSNR>=23.5 and improvement>=2dB must pass before cache
generation. Then trainer checks new-cache AE replay>=23.5 before any update.
Actual repaired-NPU reconstruction remains pending. These thresholds are declared
experiment admission limits, not a claim to have recovered historical24.5 yet.

Generation evaluation now uses a true RGB first-frame-repeat baseline. The old
anchor-latent-repeat is labeled diagnostic only in AE baseline outputs. Raw motion
and RGB-copy distances are recorded explicitly. Status updates refresh consumed
batches and clear stale eval fields; launcher start archives the prior exit record
and writes running/exit_code=null. Current repair has9 CPU tests passing including
legacy norm equivalence, weight-shape ambiguity, runtime mismatch rejection,
reconstruction gate failures, complete resume and double-write/read tests.

CRITICAL cluster-result audit: downloaded r7_window_t2v2_x0_diag_v1 shows
AE step11500 reconstruction only18.72765dB over16 held-out clips, with severe
future-frame striping/discoloration also present in generated samples. Historical
step11500 t2v2 logs report24.53646dB over32 clips (not identical evaluation sets).
The earlier 2026-08-06 entry below already documented the cause risk: t2v2 was
trained with cross-time GroupNorm; commit7f1ae43 changed it to FramewiseGroupNorm
without changing state keys. Current load_r7_modules builds the latter, so strict
load/signatures do not detect this semantic incompatibility. Stage30's t2v2
default is NOT quality-validated and should not be continued as-is. Recommend
pause/preserve and same-weight/same-video historical-vs-current encode/decode
replay, then explicit normalization-version contracts and new cache/statistics.
No need to retrain AE before this check. Joint future generation need not require
future-prefix causality; independent first-frame availability is the condition.

Actual48-NPU/BF16 execution/resume/periodic sampling are now evidenced. Main-node
snapshot step2480, other nodes2580/2620 reflect asynchronous file snapshots.
EMA RGB L1 vs AE improves .29403 at50 to .11825 at2000, but AE itself is degraded.
Logged median update time .512s, DI5058.6tokens/s/NPU, rank0 peak11.08GiB.
All6 publisher receipts synced; old launcher exit0 belongs to the50-step pause.
The current "copy" comparator repeats anchor LATENTS, an invalid proxy for a
static RGB video with t2's distinct anchor/future distributions. A real RGB
first-frame-repeat baseline was better/tied in L1 vs RAW on4 saved2000-step clips.
Neither motion ratio vs degraded AE nor beating latent-copy proves quality.

Implementation update: new full-window trainer/model/flow and stages 28/29/30
are implemented. Current runnable instructions: `docs/WINDOW_DIFFUSION_RUNBOOK.md`.
The "not implemented" wording in the original design record below describes
that earlier review, not the current code. No NPU or quality validation exists yet.

New contracts: independent single-frame condition cache (stored comparison to
full-window anchor), full future joint generation, FP32 reversible normalization,
explicit x0/velocity noise-time semantics, BF16 honored without legacy FP16
fallback, same-rank-initialized EMA before DDP, fixed-worker deterministic data
replay plus all optimizer/scaler/scheduler/RNG state. Immutable resume arguments
include world size/data/objective/precision/eval recipe. First-frame-only I2V is
an explicit NO_TEXT=1 arm; caption arms reject missing IDs rather than silently
using an empty prompt. UMT5 runtime uses bounded shard LRU instead of a complete
embedding bank per rank.

Real AE reconstruction is replayed before updates; actual checkpoint signature
and step are logged. Evaluation is distributed by clip and aggregated on rank0;
other-node previews live under workers/nodeN in both destinations. Periodic
checkpoints precede RGB evaluation. `best_reconstruction` is a distance-to-AE
selector, not a generative-quality selector. Training throughput uses synchronized
slowest-rank step timing and reports future tokens/s/NPU plus global clips/s.

User explicitly required double writes/reads and intermediates. New incremental
snapshot publisher independently attempts current output + persistent mirror,
retries failed files, excludes partial checkpoints, persists publication status,
and makes final upload failures visible as nonzero exit. Read fallback stages to
a partial file then renames. This is copy-completion evidence, not remote checksum
validation. Cache itself remains the durable shared input dataset. Fresh-run
guards occur before expensive preparation; resume is explicit, never a silent
fresh start. Historical files and cache namespaces remain intact.

Two required infrastructure fixes: factor>1 temporal codec now returns the anchor
for a one-frame encode/decode instead of convolving an empty tail (no state-dict
or multi-frame path change); distributed_barrier explicitly imports device
registration before setup_ddp so torch_npu/HCCL detection is available. Codec
single/full first-frame equality tested at factors1/2/4 with nonzero residuals.

Local tests exercise full-window gradients/conditioning, FP32/BF16 oracle Euler
and Heun, exact CPU pause/resume including EMA/Adam, and injected two-root I/O
failures. Actual 48x910B forward/backward, HCCL, OBS, AE PSNR and generated videos
remain cluster work. Default 6000 steps is an experiment budget, not quality proof.

Latest user instruction: treat the several ~25 PSNR AEs as reconstruction
baselines, then establish diffusion for this project's own setting. V-RAE is
an idea source only; 25/26 remain unfinished diagnostics, not the trainer base.
Current consolidated design: `docs/AE_BASELINES_AND_DIFFUSION_DESIGN_2026-09-08.md`.
Earlier native-I2V-first ranking and 27->26 launch recommendations are superseded.

Re-read AE logs (13 stage records including old unprefixed psnr fields). Peak
rows: t1 equal-split joint step6000 PSNR24.6408/LPIPS.11010; t1 geo112 joint
step11000 PSNR24.6200/.10825; t2 v2 joint step10500 PSNR24.5708/.11285; t2 v3
joint step10500 PSNR24.2426/.12356. First three are primary reconstruction
baselines; t2 v3 is a useful geometry-proxy/reconstruction tradeoff control.
These are log-row peaks, not replayed checkpoint_best scores. Old t1 equal-split
boundary_ratio/composite values are invalid for factor1 and may affect historical
best selection. Confirm actual checkpoint step/metrics and matched evaluation.

Proposed diffusion design (not implemented): frozen AE, first-frame/caption
conditioning, full-window future generation (8 t1 or 4 t2 latent steps -> 8 RGB
future frames), joint C192 target, distinct anchor memory, a new conditional
spatiotemporal DiT, explicit prediction/loss/sampler contracts. Initially compare
t2 v2 and t1 geo112 under one common recipe, report sample-exposure and NPU-hour
tradeoffs. No automatic promotion of compressed-feature cosine to physical
geometry, no assumption that a random Wan interface preserves its pretrained
prior, no claim that RAE dimension heuristics alone explain the old failures.

This turn audited logs/code/history and primary literature; no new training,
AE replay or accelerator benchmark was performed. Exact AE artifact identity,
latent statistics and the new trainer remain outstanding implementation work.

## Sampler Validation Implementation (2026-09-08)

USER CORRECTION / superseding decision: 25/26 are unfinished diagnostic work,
not a completed or appropriate training base for this video diffusion project.
The automatic 27->26 chain is withdrawn. Stage 27 is read-only and RUN_N1=1 now
fails explicitly. Its validator no longer imports train_r7_flow_probe. Tested
sampler algebra can remain a selective reference; none of this promotes the
384x4 probe architecture, n1 ladder or objectives into the research baseline.
Next trainer design must separate V-RAE generative modeling contracts from
repository data/cache/DDP/OBS infrastructure and verify their historical versions;
script comments saying production are not evidence of successful training.

First cluster attempt evidence: downloaded v1 contract status remains running /
passed=false; encoder matched 1210/1210 keys, logs end around dataset/OBS setup.
No completed endpoint results or definitive OOM/exception evidence. A subsequent
launch stopped on existing-output protection. Stage 27 now defaults to a fresh
v2 namespace, checks collision before metadata enumeration, uses zero workers
for the contract-only loader, and records phases/signals/process exit codes.
The underlying cause of the first incomplete attempt remains unconfirmed.

The user requested implementation and launch scripts for the validation phase.
New entry: `scripts/scale/27_validate_r7_then_n1.sh`; see
`docs/R7_SAMPLER_VALIDATION_RUNBOOK.md`. Node 0/device 0 only. It checks actual
oracle and historical deterministic x0 through sample_flow and the frozen RGB
decoder, then launches the existing bounded stage 26 only on contract success.
It never promotes n16 automatically. Outputs use a fresh probe namespace and
preserve the existing OBS/mirror contracts. Individual training-arm resume still
uses stage 25; stage 27 rejects RESUME.

Deterministic adapter explicitly restores raw anchor statistics and preserves
the legacy FP32 predictor interface before normalizing its output into target
statistics. Sampler contract success is not random-noise generation success.
Legacy deterministic checkpoints lack materialized sample hashes, so current
identity is recorded without certifying byte-identical historical inputs.

CPU FP32/BF16 sampler and adapter/negative-gate tests passed using an isolated
local dependency directory. No real codec checkpoint, NPU or OBS validation has
yet run. This supersedes the earlier statement that all CPU tensor tests were
unavailable, but does not establish accelerator compatibility or model quality.
Existing flow formula/static/CLI and CPU tensor/model-gradient/sampler/EMA/metric
regressions also passed after installing isolated local test dependencies.

## Research Position Update (2026-09-08)

The user asks for a differentiated project beyond VideoRAE and V-RAE and points
to `scripts/scale` as the historical launch entry points. See
`docs/RESEARCH_POSITION_AND_TRAINING_2026-09-08.md`. Native I2V plus geometry is
a quality baseline, not an established research contribution. Encoder swapping,
feature alignment, future prediction, warping, and joint RGB/geometry generation
already have close prior work; do not claim novelty from those labels alone.

Proposed, not adopted or validated: investigate preservation of measurable
cross-frame geometric relations through compression and denoising. First test
geometry readability in raw GFM features, R7 latents, reconstructions and samples;
only redesign temporal pooling if the evidence identifies compression damage.
Any future correspondence/visibility needed at inference must be predicted or
generated, never obtained from the ground-truth future video.

Local V-RAE code at a7783e8 uses noise-time logit-normal sampling and converts x0
predictions to velocity-space loss with denominator max(u, 0.05), equivalent to
weighted x0 MSE. This differs from current plain-x0 probes. Applying shift > 1
to this repo's data-time convention biases the opposite physical noise regime.
Wan noise-time embedding reversal alone does not reproduce this loss/distribution.
Reference repository configs and paper tables also differ; pin the exact recipe.

Keep current frozen-R7 oracle/deterministic-sampler/random-noise memory diagnostics
before scaling. No new training code, NPU jobs, checkpoint replay, or measured
method improvement was produced in this research review.

## Quality-First Scope Update (2026-09-08)

The user explicitly allows evaluating both native-video-VAE geometry enhancement
and VGGT/R7 representation replacement, prioritizing generation quality. The
former no-VAE-at-inference constraint is therefore no longer mandatory for new
research branches. Compute remains 48 Ascend 910B NPUs; preserve ModelArts/OBS
launch, staging, resume and output conventions. Historical checkpoints retain
their original contracts.

See `docs/QUALITY_FIRST_AUDIT_2026-09-08.md` for the local review and proposed
experiments. The recommendation is to establish a native I2V quality baseline,
then compare matched LoRA-only and VGGT-supervised adaptations. Model choice and
910B compatibility are unvalidated. Keep R7 frozen for a bounded diagnostic:
actual sampler oracle and existing deterministic g(anchor) as time-independent
x0, before further random-noise n1 experiments. These are recommendations, not
new trained models or a production-gate relaxation.

New audit facts: 31 desktop metrics JSONL files parsed, 313 eval/gate rows,
zero JSON parse failures. Geo112 joint step12K: PSNR24.602, LPIPS.1083,
feature-motion cosine.8922. Old Wan7K->14K: EMA x0 MSE.6710->.3577 while
RGB PSNR versus AE falls10.952->9.076. Geometry-motion in the codec trainer is
compressed-feature delta cosine, not measured physical geometry. The concatenated
geo/texture bottleneck passes through full-channel temporal convolutions, so
the named channel split does not guarantee disentangled final latents. The
successful deterministic predictor and failed flow differ in conditioning,
normalization and architecture; they are not a noise-only controlled comparison.

Validation: existing flow formula/static/CLI checks PASS; torch unavailable on
this Windows audit host, so tensor/model/gradient/sampler checks explicitly SKIP.
No accelerator jobs, model/cache changes, or checkpoint replays were performed.
The older phase headings and literature positioning below are historical unless
confirmed by this update and raw evidence.

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

## Current Phase: Frozen-RAE Flow Recovery (2026-09-07)

The active task is first-frame/text-conditioned **video generation in the R7
geometry RAE space**, not 4D scene generation and not Wan-VAE geometry guidance.
The next diagnostic freezes the completed t1 geo112|tex80 checkpoint and tests
actual flow sampling before extending to 2/4/8 future frames. See
`docs/R7_FLOW_RECOVERY_PLAN.md`. New code is not a measured generation result;
accelerator/OBS smoke and quality experiments remain pending.

Evidence from the user-supplied desktop experiment excerpts:
- `r7_vggt_quick_geo112_tex80_v2/det_k1_n1` succeeds at deterministic one-pair
  memory: step500 PSNR vs AE 35.11, RAW 21.756 versus AE 21.836. The n16 run is
  deterministic held-out prediction; the only single-target flow is a 2-step
  smoke. Do not label this a completed single-frame diffusion failure.
- Saved old t2 samples expose a metric confound: a spatially constant future
  made from training position/channel means reproduces raw motion cosine about
  [.996,.921,.602,.241]. Near-perfect first-chunk cosine and raw latent
  correlation/std do not establish video content or motion learning.
- On four saved Wan samples, normalized std ratio shrinks from about .59 at
  7K to .50 at 14K while raw std ratio remains near 1. This is within-tensor
  variation, not a multi-seed conditional-diversity measurement.
- Old Wan14K EMA x0 MSE at data-time .9 is .2913, versus the analytical
  rescaled-input baseline's expected .01235. All noise buckets improve with
  training, so a high-noise-only explanation is insufficient. Historical EMA
  defaults (.9999 without warmup) remain a confound until online/EMA replay;
  runtime overrides are not established by the excerpts.

New isolated probe implementation: `train_r7_flow_probe.py`,
`models/r7_flow_probe.py`, and `scripts/scale/25_run_r7_flow_probe.sh`.
Old-Wan replay uses `evaluate_r7_wan_denoising.py` and stage 24 with an explicit
checkpoint/cache/text contract; it never substitutes current time semantics.
Plain-x0 and preconditioned heads are separate controlled arms; train-memory and
held-out, online and EMA, raw and position-centered metrics remain separate.
Existing codecs/caches/checkpoints and production gates are not changed.

### Evening n1 follow-up (2026-09-07)

The newer `r7_geo112_flow_n1_probe_v1/n1_plain_x0_n1_f1` excerpt reaches step500
online eval: generated-vs-AE PSNR12.893, RAW12.823 vs AE21.836, normalized
sample MSE1.333. Denoising improves (t=.1/.5/.9 MSE .223/.053/.041), but free
sampling has not memorized the pair. This gap does not prove a unique cause
such as off-manifold drift, x0 amplification, or an unsuitable R7 representation.
The user reports platform success, but the excerpt lacks final/step500-EMA
completion; do not guess MAX_STEPS or infer a kill from an uploaded temp file.
The .tmp-* MoXing warning comes from background watch retries, not trainer exit.

Next implementation uses `r7-prefix-flow-v2` and a fresh output namespace:
plain x0 / preconditioned / direct velocity; fixed-path means one noise endpoint
with cycling data-time, not one fixed (noise,t) pair. Seen noise, unseen noise,
and oracle-start rollouts are separate diagnostics, never interchangeable gates.
`run_status.json` distinguishes running/paused/completed/failed, logs print time
buckets and phase progress, and the wrapper verifies actual budget and final
checkpoint before success. Local output sync snapshots skip intermediates and
pin finalized checkpoint inodes; no original temp files or outputs are deleted.
Stage26 runs smoke plus preconditioned/direct-velocity n1, with fixed-path
fallback only if neither passes. It never automatically starts n16.

## Historical Phase: Reconstruction-First (H200)

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
  6.5) — residual contract overturned, see Decisions.
- **E7 smoke diffusion COMPLETE (2026-07-16, 48x910B): the dual-stream
  latent is learnable by flow-matching diffusion. Absolute-target arm
  healthy (vmse 0.78 @6k and falling, std_ratio 0.95, no z0-copying,
  z0-anchored coherent scenes); residual arm plateaued/oscillating —
  parameterization decision is now measured, not assumed. R3 zero-arm
  reproduced the single-stream ceiling (20.77 vs historical 20.6),
  pinning z_tex's contribution at +4.15 dB. Remaining E5 matrix item: R4
  tex_only anti-bypass proof.**
- **R6 RESULT (2026-07-23, latent_bottleneck_c128, 48x910B): final eval
  PSNR 24.420 ± 2.478, LPIPS 0.1124 at step 3001. This passes the E9 chain's
  23.9 dB reconstruction gate, so the 512->128 bottleneck is accepted for the
  compressed-latent diffusion experiment. The first E9 launch exposed a
  multi-node staging bug: rank 12 read a truncated node-local R6 checkpoint
  (`PytorchStreamReader ... failed finding central directory`) while rank 0
  reached DDP. Stage 10 now waits for node 0's final synchronous dual write,
  keeps node 0's locally produced artifact, and forces nonzero nodes to restage
  it from the current OUTPUT_URL, falling back to the persistent owner mirror,
  before torchrun.**
- **E9-v1 RESULT (2026-07-23, dual_diffusion_absolute_c128, 48x910B,
  6001 steps): compressed-latent diffusion is learnable but the additive-I0
  condition is insufficient for useful long-horizon RGB. Velocity MSE reached
  0.289, gen_std_ratio 0.966, and motion_ratio 1.008, yet fixed-clip RGB
  improved only through about step 2000 (~14.5 dB) and frames 4-7 lost subject
  structure. This isolates the bottleneck at long-horizon condition use rather
  than R6 reconstruction or latent marginal learnability. E9-v2 keeps R6,
  absolute targets, model size, and budget fixed, but prepends clean normalized
  z0 as temporal frame 0 in every DiT temporal block; outputs are isolated under
  `dual_diffusion_absolute_c128_cf0`. Clean z0 uses the flow data-endpoint
  `t=1` embedding; only future frames receive the sampled shifted flow time.**
- **E9-v3 DESIGN (2026-07-27): the next combined optimization arm is
  `dual_diffusion_absolute_c128_cf0_stmotion`. It preserves R6 c128, absolute
  targets, clean z0 with t=1, 1152-wide 10S/6T DiT, and the 6000-step budget,
  but executes `S0,T0,...,S5,T5,S6..S9` so temporal information is repeatedly
  redistributed spatially. Training adds high-t, warmup-ramped trajectory
  losses in unnormalized c128 (`lambda_motion=0.10`, `lambda_accel=0.05`) and
  expanded frozen geo256 (`lambda_geo_motion=0.05`), with warmup/ramp 1000/1000
  and shifted-t gate 0.60. Evaluation uses EMA weights, deterministic noise,
  per-horizon motion metrics, and synchronized rank-0 eval/save. This is a
  deliberate joint arm aimed at reasonable geometry-consistent video, not a
  single-variable ablation.**
- **E9-v3 CONTINUATION (2026-07-27): extend
  `dual_diffusion_absolute_c128_cf0_stmotion` from step 6001 to 12000 in the
  same namespace. Model, EMA, Adam moments, R6, latent stats, trajectory loss
  contract, and RNG are restored. The completed 6k cosine scheduler is not
  reused: the first extension launch warms from its ~1e-6 terminal LR to 1e-5
  over 200 steps, then cosine-decays to 1e-6 at 12k. Subsequent preemption
  resumes restore this continuation scheduler exactly. Launch with
  `scripts/scale/extend_dual_diffusion_12k.sh`.**
- **R7 DESIGN (2026-07-29): replace R6's per-token channel-only bottleneck
  with a causal local-3D tokenizer over the existing StreamVGGT geometry +
  TextureEncoder dual latent. Candidate contracts use 9 RGB frames with an
  independent frame-0 anchor: factor 2 gives 5 latent frames at geo96|tex96
  (192ch), factor 4 gives 3 latent frames at geo128|tex128 (256ch). Causal
  left-padded 3D blocks preserve moving content across neighboring grid cells;
  per-stream projections keep geometry identifiable. Promotion is probe-gated:
  causality invariants, factor-2 PSNR >=23.9/LPIPS <=0.13, factor-4 PSNR >=23.4/
  LPIPS <=0.15, no boundary spike, and geometry-motion cosine >=0.95. Matching
  diffusion uses multiple clean past latent chunks, joint future diffusion,
  temporal global offsets, overlapping rollout, and generated-context noise.
- **R7 IMPLEMENTATION UPDATE (2026-07-31): the 3k probes do not promote either
  candidate. t2/c192 reached PSNR 22.7184 and temporal error 0.01801; t4/c256
  reached 22.4794 and 0.02039. Both remain below the documented R7 reconstruction
  gates, but t2 is the sole continuation candidate because it is better on both
  measured metrics. The production t2 path is now a fresh v2 codec stage
  initialized from the legacy 3k weights, followed by joint tokenizer +
  DualStreamDecoder finetuning; StreamVGGT, CompactCompressor, and TextureEncoder
  stay frozen. Promotion requires PSNR >=23.9, LPIPS <=0.13, boundary ratio
  <=1.10, geometry-motion cosine >=0.95, and checkpoint causality invariants.
  Accepted t2 latents use one clean anchor plus four absolute future chunks.
  The durable cache computes exact train-split [position,channel] moments, and
  diffusion applies reversible z-score normalization only at its boundary.
  The fixed-window generator uses clean-frame0 interleaved DiT attention to
  jointly denoise all four future chunks; rollout remains deferred. Stage 11-14
  wrappers implement strict contracts, periodic/latest/best/final checkpoints,
  EMA evaluation, 6k-to-12k extension, OBS/mirror staging, and deterministic
  latent/RGB samples. Promotion against E9-v3 12k requires fixed-seed future
  RGB LPIPS to improve by at least 10%, better late-half structure, motion ratio
  closer to 1, higher latent/expanded-geometry motion cosine, and no material
  reconstruction/variance regression. Cluster NPU/HCCL/MoXing runtime remains
  to be smoke-tested.**
- **R7 CAUSALITY CORRECTION (2026-08-06): the first production t2/c192 v2 joint
  checkpoint passed PSNR, LPIPS, and boundary gates but failed strict prefix
  causality because applying GroupNorm directly to `[B,C,T,H,W]` coupled every
  frame through shared temporal statistics. R7 temporal residual blocks now use
  framewise GroupNorm by folding `T` into the batch dimension during
  normalization. The module subclasses GroupNorm so legacy `norm*.weight` and
  `norm*.bias` checkpoint keys remain strictly loadable. The old checkpoint is
  still non-promotable: changing normalization semantics invalidates its measured
  quality, and its geometry-motion cosine was 0.521 versus the required 0.95.
  Retraining/evaluation and cluster NPU validation are required.**
- **R7 V3 CORRECTION RUN (2026-08-07): preserve v2 outputs and use the default
  `r7_t2_c192_v3` namespace. Temporal residual blocks use framewise GroupNorm,
  and tokenizer training adds `lambda_geo_motion_cosine` to optimize the exact
  geometry-motion gate rather than only its L1 proxy. V3 always starts from
  weights-only initialization (legacy codec for codec, v3 codec-best for joint),
  with fresh optimizer/scheduler state; it does not full-resume v2. Downstream
  cache/diffusion defaults also point to v3 and must remain blocked until all
  quality and causality gates pass. A separately named diagnostic diffusion may
  run only with explicit `ALLOW_DIAGNOSTIC_DIFFUSION=1`; it still requires strict
  causality, production RGB/boundary gates, and geometry cosine >=0.82, writes to
  `r7_diffusion_t2_c192_ctx1_fut4_v3_diag`, and never creates or bypasses a
  production passed-gate marker.**
- **R7 DIFFUSION FAILURE DIAGNOSIS + WAN-1.3B PIVOT (2026-08-11): the v3
  diagnostic diffusion (6K steps) is numerically healthy but severely
  under-trained (velocity MSE 2.0 -> 1.26 and still falling at stop; latent
  correlation ~0.93) and its near-manifold latents decode to colorful noise
  because the DualStreamDecoder never saw perturbed latents. Motion collapses
  with horizon (chunk cosine 0.996/0.805/0.29/0.05) — the absolute-target
  velocity objective lets static content dominate and rewards copying the
  anchor. Fix order (user decision, strictly serial): (1) decoder_robust
  noise-augmentation phase in `train_causal_dual_tokenizer.py` (tokenizer
  frozen, decoder-only, Gaussian sigma~U(0,0.35) on future chunks, anchor
  clean; best requires noised-PSNR improvement over the init baseline; output
  `r7_t2_c192_v3/decoder_robust/checkpoint_best.pt` — it can NEVER be passed as
  `--r7_ckpt` because cache signatures.r7 binds the accepted joint bytes;
  diffusion consumes it via the separate `--decoder_ckpt` override that
  strictly loads only `decoder.*`); then (2) full-parameter finetune of
  Wan2.1-T2V-1.3B (`train_causal_wan_video_diffusion.py`, stage 16) with
  x0/clean-target prediction + FM sampling (VGGT-World velocity-collapse
  evidence), anchor-as-clean-frame conditioning, native UMT5-xxl text
  conditioning from a precomputed sidecar (stage 15; captions looked up from
  annotation_index by cached video_id — no cache rebuild), time_shift_alpha 3.0,
  CFG dropout 0.1, two-speed LR (adapter 1e-4 / trunk 1e-5), wan_freeze_steps
  500, 30K default steps on the existing 10k cache with train/eval-gap overfit
  monitoring. The old I2V-14B last-4-QKV path is discarded for this stage.
  Launching follows the single-submission chain model: stages
  `decoder_robust,text_embed,wan_diffusion,wan_sample` are wired into
  `14_r7_recon_then_diffusion.sh` with durable OBS publication + barriers and
  gate/diagnostic re-verification per stage — never a sequence of interactive
  bash commands. Stage 15 prefers a durable prebuilt annotation index on OBS
  and hard-fails (with instructions) on silently-empty annotation staging,
  because `copy_parallel` is a no-op for missing OBS prefixes.**
- **R7 T2/C192 + WAN-1.3B RESULT AND REDESIGN (2026-08-17):**
  `decoder_robust` completed 4K steps and improved fixed-sigma-0.22 noised
  reconstruction from about 20.33 to 20.65 dB (noised LPIPS 0.270 -> 0.248),
  while leaving the frozen tokenizer geometry cosine at about 0.832 as expected.
  The Wan T2V-1.3B x0 run reached step 14K: teacher-forced x0 MSE kept falling
  (about 1.04 -> 0.358), but deterministic sampled RGB/composite peaked around
  step 7K and then regressed; future chunk motion cosine collapsed with horizon
  (step 14K about 0.995/0.916/0.591/0.194) and expanded geometry-motion cosine
  remained near 0.02. This is objective/interface/representation failure, not
  evidence that the same arm merely needs 30K steps. The current t2/c192 codec
  and Wan run remain diagnostic and cannot be production-promoted. Training eval
  now writes labelled ANCHOR/AE_TARGET/GENERATED PNG grids, frame PNGs, MP4s, and
  preview manifests; large float RGB packs are opt-in. The redesign is gated:
  first compare factor-1 against t2 codec ceilings, then train a Wan-native
  teacher bridge (frozen Wan VAE permitted only during training; inference stays
  R7-only), add per-block anchor memory and horizon/motion losses, and finish
  with short-window overlapping rollout. Only a 10K arm passing multi-seed
  late-horizon/RGB/geometry gates may rebuild the full SpatialVID-HQ cache. The
  formal experiment plan is `docs/R7_QUALITY_PLAN.md`. Formal generation remains
  restricted to Wan2.1-T2V-1.3B; I2V-14B is not available for this redesign.**
  latest Wan run (`outputs/scale/wan_compact_i2v14b480p_v1`, step 36250)
  showed under-dispersion consistent with both the old 20-PSNR latent and
  the whitened residual target.
- **R7 QUALITY REDESIGN (2026-08-17): active next work is evaluation-first.**
  Do not extend `r7_wan13b_t2v_ctx1_fut4_v1` blindly. First replay the 7K/14K
  checkpoints through the new PNG/MP4 and horizon-guard path; then run the
  factor-1 codec ceiling and Wan-native teacher-bridge probes defined in
  `docs/R7_QUALITY_PLAN.md`. New representation experiments require new
  namespaces/caches/stats and must not overwrite v3 artifacts.
- **R7 T1 GENERATION SUPPORT (2026-08-21, code):** the diffusion trainers
  (13/16) and their wrappers are now parameterized by `temporal_factor` /
  `future_chunks` (t2/ctx1/fut4 default unchanged; t1/ctx1/fut8 supported).
  The latent contract is derived from the cached representation config
  (`expected_future = (seq_len-1)//temporal_factor`), stats/args cross-validated,
  late-horizon quality guards target the last two future chunks, and the Wan
  `--horizon_weights` defaults to 8 entries for t1. Runbook: R7_README.md
  "Factor-1 (t1) generation". NPU runtime still requires a smoke run.

- **R7 T1 AUDIT + FAIL-CLOSED CORRECTION (2026-08-31):** local metric artifacts show the t1 codec completed 6K, joint stopped at 9.8K, decoder_robust completed 4K, and the from-scratch t1 diffusion completed its initial 6K. The t1 representation did not pass: best observed joint PSNR/LPIPS were about 24.64/0.108, but geometry-motion cosine plateaued near 0.867 < 0.95. Factor-1's prior boundary metric was mathematically invalid (all transitions classified as boundaries, empty within set, ratio ~4e10); factor=1 now records boundary as not-applicable and excludes it from gate/composite, without weakening the geometry gate. The old t1 diffusion was numerically improving but generation failed (step-6K motion ratio ~3.47, chunk-8 motion cosine ~0.017, expanded-geometry cosine ~0.011); it must not be extended in place. The native trainer now uses a fresh x0 objective namespace and can strictly load decoder_robust weights via a separate decoder override. Wan had made zero optimizer steps: the current launch crashed on the first forward because horizon weights were converted to a list before `.to()`. That crash, Wan native-time reversal, raw-space motion losses, fixed-width text conditioning, guarded-best selection, sampler dtype, EMA warmup, and checkpoint retention are corrected under new objective/namespaces. Production cache/diffusion wrappers now require a checkpoint-bound gate marker; diagnostic bypasses require diagnostic/probe namespaces. The current teacher bridge remains a pooled, one-way diagnostic not connected to production Wan; teacher markers now record factor/shapes/signatures and stage 20 forwards t1 namespaces, but a held-out streaming bridge probe/native-grid inverse head remain future work. No long run has been relaunched after these code changes.**

- **VGGT/R7 SINGLE-TARGET QUICK PROBE (2026-09-04, code):** the completed
  `r7_t1_c192_geo112_tex80_probe_v1/joint` arm retains RGB reconstruction
  (best near step 11K, PSNR about 24.62 / LPIPS about 0.108) and improves
  geometry-motion cosine to about 0.892, but remains below the long-video 0.95
  gate. It is now a frozen reconstruction base for an isolated target-frame
  diagnostic, not a cache/diffusion promotion. `probe_vggt_manifold.py` measures
  raw VGGT, compressor-projected, and R7 statistics plus full-sequence,
  repeated-candidate equal-scale Euclidean/tangent/geodesic decoder sensitivity.
  `train_single_target_probe.py` first requires deterministic frame-0 ->
  frame-1 overfit and then scales 1/16/256 clips before allowing a one-frame x0
  flow arm. Every eval separates raw RGB, R7 AE target, and copy-anchor. The
  ModelArts entry points are `scripts/scale/22_probe_vggt_generation.sh` and
  `scripts/scale/smoke_vggt_generation.sh`. These runs never write a gate marker;
  held-out copy-anchor improvement and later VGGT geometry re-encoding are
  required before any generative claim. NPU runtime remains unverified.**

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

### Decision: Diffuse Future Residuals, Not Full Latents — OVERTURNED (2026-07-16, E7)

- ORIGINAL content: frame 0 is condition only; generator predicts seven
  residual latents `zt - z0`. Reason: reduces target entropy.
- **OVERTURNED by measurement.** Evidence chain:
  1. Diffusability spectra (R1+R5 checkpoints): residual streams are
     spectrally whitened (high-freq 0.49-0.70, eff-rank 110-153, kurtosis
     4.5-6.5) vs healthy absolute streams — the anti-SSVAE profile.
  2. E7 smoke diffusion (48x910B, 101.5M DiT, 10k oft subset, online
     encoding, 6000 steps): ABSOLUTE arm converged cleanly
     (velocity_mse 2.00->0.78 still falling, gen_std_ratio ~0.95,
     recognizable z0-anchored scenes); RESIDUAL arm plateaued at ~1.0-1.1
     with oscillating std_ratio (0.90<->1.18) and clip-dependent target
     stats (eval target_std 0.91 despite normalization).
  3. Motion analysis (analyze_e7_motion.py, step 6000): no z0-copying —
     gen motion magnitude ratio -> 1.0 by frame 4-7, direction cosine
     rising 0.25->0.39; frame-1 over-motion (ratio 1.66) is a constant
     sampling-noise floor, not a shortcut.
- **NEW CONTRACT: the generator predicts ABSOLUTE normalized z_1..z_7
  (dual-stream, geo||tex 512ch) with normalized z_0 as a clean-past
  condition (MIRA-style).** Frame 0 remains condition-only. Per-frame
  per-channel stats; stats travel inside the generator checkpoint.
- Impact: the 1.46M residual-contract cache design is obsolete BEFORE
  rebuild (nothing wasted); train_dual_diffusion.py is the reference
  implementation; old residual checkpoints/caches are incompatible.
- Watch item: eval/motion_ratio (now logged every eval) is the standing
  z0-copy guard; frame-1 ratio should fall toward 1 as models scale.

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

### Decision: Use Wan 14B As Pretrained Video Prior — SUPERSEDED for R7 (2026-08-11)

- Content: use `Wan2.1-I2V-14B-480P` through `WanCompactAdapter` on the same
  StreamVGGT compact latent contract.
- Reason: from-scratch DiT has not removed ghosting; Wan provides pretrained
  video dynamics and spatiotemporal attention.
- Rejected alternatives: only scaling from-scratch Compact DiT; relying on Wan
  VAE latent semantics; old online Wan harness.
- Impact: checkpoint stores trainable deltas only and requires the same
  `WAN_CKPT_DIR` at resume/sampling.
- Superseded (R7 path): stage 16 uses `Wan2.1-T2V-1.3B` with FULL-parameter
  finetuning instead. The 14B last-4-QKV compromise gave too little trainable
  capacity to re-aim the prior at the StreamVGGT latent distribution (Gen3R
  alignment lesson), while 1.3B full finetune fits replicated DDP on 60GiB
  NPUs (~1.4B trainable, ~21G optimizer state). The old E7/E8 dual paths are
  unaffected.

### Decision: Train Last-N Wan QKV Blocks By Default — SUPERSEDED for R7 (2026-08-11)

- Content: default `TRAIN_QKV=1`, `TRAIN_QKV_LAST_N=4`.
- Reason: full 14B QKV under replicated DDP OOMed; last layers adapt high-level
  motion while preserving lower/middle Wan priors.
- Rejected alternatives: freeze all QKV forever; train all 40 QKV blocks under
  DDP; full model finetune without FSDP/ZeRO.
- Impact: if last-4 improves samples, expand to last 8/12; full QKV needs a
  different memory strategy.
- Superseded (R7 path): `full_finetune=True` in `WanCompactAdapter` unfreezes
  the entire 1.3B trunk (QKV, cross-attn, FFN, norms, text_embedding, time
  path); only forward-unused modules (patch_embedding, head, img_emb, legacy
  CLIP text_proj) stay frozen for DDP correctness. A hard guard rejects >1.5B
  trainable and non-1.3B configs.

### Decision: Keep Text Conditioning Off — SUPERSEDED for R7 (2026-08-11)

- Content: current generator is non-text, I0-conditioned.
- Reason: the project must first prove geometry-aware latent generation; text
  conditioning introduces a second variable.
- Rejected alternatives: CLIP conditioning from legacy Wan scripts; immediate
  native UMT5 conditioning.
- Impact: Wan text projection is not trained unless explicitly enabled.
- Superseded (R7 path): stage 16 conditions on native UMT5-xxl embeddings from
  the stage-15 precomputed sidecar (SceneDescription captions keyed by cached
  video_id; empty-prompt embedding for missing captions and CFG dropout 0.1).
  The T5 encoder never enters trainer memory. The legacy CLIP `text_proj`
  branch remains frozen and unused.

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
- `LOCAL_CACHE_ROOT` is disposable node-local staging, usually
  `/cache/yexiaoyu/vggae_runtime`; a checkpoint written by global rank 0 is not
  automatically available at the same path on other ModelArts nodes. Chained
  stages wait for node 0's final dual write, keep node 0's local artifact, and
  force nonzero nodes to stage it from the current output URL, then the
  persistent owner mirror, before torchrun.

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
