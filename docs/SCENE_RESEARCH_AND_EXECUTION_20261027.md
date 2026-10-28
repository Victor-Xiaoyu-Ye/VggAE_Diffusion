# Fixed-window geometry-aware generation: evidence, method and execution

Updated 2026-10-27. This document separates measured failures, proposed methods,
and implemented diagnostics. It does not certify generation quality or novelty.

**2026-10-28 result update:** Stage57 is complete; read
`SCENE_END_TO_END_RESULTS_20261028.md`. FP32 does not fix the neural tracker,
and LK transport has not reliably beaten copy-last in normalized texture error.
The immediate reference-control pilot below is therefore deprioritized.
GT texture replacement nearly closes the generation-to-AE RGB gap; GT geometry
replacement after sampling does not. Prefer a bounded, explicitly oracle
geometry-conditioned texture-denoising experiment against a matched prefix-only
baseline, plus an independent structured-error decoder study retaining clean AE
quality. Neither has been implemented or proven by the completed audit. Geometry
is not proven unused, and GT branch replacement is not deployable generation.

## 1. The concrete target

The main task remains video generation in our learned StreamVGGT-derived
representation, with four observed frames and five generated frames in the
current nine-frame window. First obtain clear moving objects and stable scenes.
Long rollouts, a WASD UI and an autonomous world model are later extensions.

The current state is **not enough for a strong conference paper**. A better
story cannot substitute for a valid control comparison, strong generation,
or evidence that geometry helps. The feasible research question is:

> Can geometric correspondence reduce the uncertainty of dynamic appearance,
> while leaving unseen content to a stochastic generator, in a compact
> representation from a geometric foundation model?

This is a hypothesis, not an established contribution. It is narrower than
general interactive world simulation and can be tested with the existing
codec, ~831M generator, street videos and three Ascend nodes.

## 2. What is actually wrong or still unknown

| Evidence | Established conclusion | What it does not establish |
|---|---|---|
| Stage56 B/C had zero track/reference support throughout training and evaluation | The advertised motion/reference comparison never operated; it was camera plus constant unknown maps | Reference guidance cannot work |
| Two saved full/prefix teacher cases contain 14,976 point-frame observations; all fail both historical score thresholds | The current checkpoint/backend output cannot supply these track weights | A specific numerical or upstream training bug is proven |
| Visibility < .043, confidence about 1e-10–1e-9; forced frame-zero coordinates are correct | This is not merely future occlusion or out-of-frame points; lowering cutoffs alone still leaves transport weights effectively zero | Camera/depth outputs are correct because they are finite |
| A/B/C have similar videos and ~0.4% B/C advantage in full-frame L1 over A | No meaningful reference-control quality result yet | L1 proves no improvement on every moving object |
| Clean AE is much better than generated video | A large generation-to-reconstruction gap remains | The representation/decoder is easy to model under generated errors |
| 14/64 original eval cases were skipped for replay mismatch | Existing reported averages use a selected 50-case intersection | They are directly comparable with old 64-case means |

Details and measured numbers are in `SCENE_CONTROL_RESULTS_20261027.md` and
`SCENE_TRACK_TEACHER_FAILURE_20261027.md`. Do not describe newly working code
as newly good videos.

One additional design issue was found during this audit: legacy controls
include dynamic/static/unknown labels computed with **future depth**, reverse
depth consistency and future persistence. Dense depth is not directly passed
as a map, but information derived from it reaches the adapter. Consequently,
Stage56 v1 is a teacher-oracle control experiment, not a complete interface
that can be reconstructed from user-supplied camera poses and point paths.
The v2 input policy removes this discrepancy; see the runbook.

## 3. Relevant work and the boundary of a defensible claim

The following are primary papers/projects checked on 2026-10-27. Published
results are evidence for their own settings, not promises for our checkpoints.

| Work | What is useful here | Consequence for our project |
|---|---|---|
| [GAE v4](https://arxiv.org/html/2609.24981v4), [official code](https://github.com/TencentARC/GAE-GeometricAutoEncoder) | DA3 feature hierarchy is compressed into a compact state with RGB and frozen geometry readout; token/relation shaping targets generation. Controlled generator is about 0.93B, with nine-frame 252px experiments. | ~831M versus ~0.93B is not a sufficient failure explanation. Our learned RGB texture half and readout differ. Geometry-native compact diffusion alone is already prior art. Released continued-training weights are not automatically its controlled-table model. |
| [GLD](https://arxiv.org/html/2603.22275v1) | Diffuses geometric foundation features and uses a multi-level cascade; includes a VGGT backbone experiment. | Direct geometry-feature diffusion is already established. Its data/compute and multi-view task differ; do not imply equal-budget reproduction. |
| [Gen3R](https://arxiv.org/html/2601.04090v2) | Couples geometric representations with Wan appearance generation. | Retaining a native appearance prior differs substantially from loading Wan DiT parameters into a new C256 latent. It is a useful quality baseline, not evidence our previous weight initialization should work. |
| [OneWorld](https://arxiv.org/abs/2603.16099) | Unified appearance/geometry/semantics, correspondence loss and exposure to drifted representations. | “Add correspondence consistency” or “make the decoder robust to generated latents” alone is not a new contribution. |
| [Go-with-the-Track](https://arxiv.org/html/2606.20891v1) | Reference-anchored point tracks, multiple references and lightweight adaptation; real/static/synthetic mixture. | Track conditioning plus a reference adapter is already a strong baseline, including small Wan settings. A new method needs a distinct mechanism and matched comparisons. |
| [Geometry Forcing](https://geometryforcing.github.io/) | Geometric representation alignment for camera-conditioned world generation. | Merely aligning a video generator with VGGT or adding camera control is insufficient novelty. |
| [MIRA](https://mira-wm.com/blog-post/) | 5B generator, 600M codec and about 10k match-hours with synchronized actions, within a restricted game domain. Codec organization changes object persistence. | Its success is not a pure parameter-count argument. Our real videos have wider appearance, occlusion and uncontrolled actor dynamics; no equivalent action stream exists. |
| [CADRE, Oct 21](https://arxiv.org/html/2610.30129v1) | Compares codecs through downstream world models; perceptual regression can score well in PSNR while producing blur. Spatial compression and moving-camera motion need separate evaluation. | Measure moving-object detail and generated-latent decoder robustness, not AE PSNR alone. Its 76k-hour training and diffusion decoder are not a recipe we can simply reproduce on 24 NPUs. |
| [RAEv2](https://arxiv.org/abs/2605.18324), [frequency perspective](https://arxiv.org/abs/2511.22249) | Representation choice, alignment and frequency-sensitive errors affect generation. | These motivate controlled perturbations. They do not justify another unconditional round of auxiliary-loss or noise-schedule tuning. |

The first seven rows deliberately rule out easy narratives. We should not
claim that VGGT-as-RAE, a motion mask, geometry loss, a trajectory adapter or
noise-robust decoding is independently novel.

## 4. The method worth testing

Use geometry to separate camera-induced background displacement from object
motion. Reuse observed appearance where correspondence is reliable; generate
appearance where it is absent or occluded. Keep geometry readable independently
of the texture shortcut. A possible formulation is a transported appearance
prior plus a generated residual, with explicit unsupported regions. It is not
yet implemented as a new diffusion objective, and its benefit is unproven.

The distinction to test is not “another reference condition.” It is whether
the geometric organization reduces conditional uncertainty and preserves
appearance under camera/object motion that a matched 2D-flow-only formulation
cannot handle. If this advantage cannot be shown, do not keep a geometry-aware
claim just because the encoder is named VGGT.

Depth by itself does not specify independent object motion. Semantic masks
identify regions or categories but generally do not provide identity-preserving
correspondence. A hard cross-frame attention ban would remove useful motion
evidence and is not the current recommendation. Start with reliable sparse
correspondence, explicit uncertainty and a matched control budget.

Three condition settings must remain separate:

1. Prefix-only completion: no future trajectories; model samples plausible
   motion. Paired pixel metrics alone penalize valid alternative futures.
2. Supplied camera/point paths/visibility: an explicitly controlled task,
   with no future RGB or future texture in the conditions.
3. Full-teacher/oracle studies: richer future-derived pseudo-labels or GT
   branch replacements, used only to locate an upper bound or failure.

WASD can later map to camera trajectories with a declared scale and timing
model. It does not tell a pedestrian where to walk and is not a substitute
for actor-control data. A trajectory-conditioned video model is not by itself
an action-driven autonomous world simulator.

## 5. Experiments in a useful order

| Phase | What runs | Decision it supports |
|---|---|---|
| Stage57, now | Original eight eval cases; teacher BF16/FP32 and real/repeated frames; independent LK; raw replay; source sampling; clean/sampled geo/texture interventions; equal-MSE decoder probes; observed-reference transport versus copy/wrong paths | Whether the immediate barrier is teacher support, numerical behavior, correspondence utility, branch organization, or structured decoder error |
| Valid-control capacity pilot | Freeze AE and normalization; same 831M source checkpoint, identical windows/noise; camera-only, camera+tracks, camera+tracks+observed appearance; use the v2 inference-constructible policy | Does a genuinely available control improve dynamic appearance? Include correct/zero/shuffled controls and edited paths after adapter training |
| Mechanism experiment | Absolute-latent versus transported-residual prediction; geometry-based versus 2D-only transport at matched controls/parameters; visible versus newly exposed regions | Is there a geometry-specific mechanism beyond conditioning? |
| Representation alternative, only if indicated | Compact/multilayer organization or decoder training on measured generated errors; preserve clean reconstruction and geometry readout | Does the codec cause the remaining blur? Compare clean versus structured versus equal-MSE random latent errors |
| Quality baseline | Native Wan appearance prior plus the same controls, or an available official GAE model with its exact inputs | Whether our quality is competitive; this is not sole-VGGT-latent generation |

The next training pilot should use a declared, fixed 128–512-clip motion-focused
training subset, plus held-out source-disjoint evaluation. A high RGB-change
heuristic is not a semantic dynamic-object label; manually check a small
stratified subset for people/cars, camera motion, occlusion and shot cuts.
Do not choose evaluation cases only after seeing improvements.

Compare checkpoints at 1k and 2k matched updates before granting a long run.
This is an experiment budget, not a hard visual-quality gate that kills other
work. Existing 5k-update arms took about seven hours each on 24 NPUs; simple
scaling gives about 1.4/2.8 hours for 1k/2k **only for the same training setup**,
excluding preparation, evaluation and changed-module overhead. New timing and
memory must be measured; Stage57 is inference, not a training-throughput test.

Every arm retains intermediate model/optimizer/EMA checkpoints, per-case
outputs and declared DI units. Stop expanding a failed hypothesis, but keep
independent baselines running. Do not stop because one clip has zero support.

## 6. What would support a paper

- Clear fixed-window quality gain on a held-out real domain and a second
  source with independent motion/geometry evidence. Do not rely on a few
  selected videos, full-frame L1, or a tiny-sample FVD.
- A matched 2D-only comparison, geometry intervention and geometry readout
  test establishing that geometry is used. Merely having two channel halves
  does not establish disentanglement.
- Camera and object controls edited independently; correct/zero/wrong
  conditions and occlusion/reappearance cases. Separate control accuracy,
  perceived quality and diversity.
- Teacher-based pseudo-label measurements separated from independent truth
  or another evaluator. Avoid training and judging geometry only with the
  same VGGT head.
- Report uncertainty/failures, parameter count, compute, data exposure,
  resolution, frame timing and what future information each baseline receives.

For deterministic or fully specified control studies use paired LPIPS/PSNR
and region errors alongside distributional/perceptual evaluation. For
prefix-only stochastic futures, emphasize realism, motion and consistency;
do not reward a blurry conditional average for matching GT pixels.

If reliable trajectories reduce transported-texture error but the model
ignores them, improve conditioning/optimization rather than retrain the AE.
If reliable transport does not beat copy, examine correspondence, occlusion
and latent organization first. If GT texture replacement repairs the image
while GT geometry does not, test texture prediction/decoder sensitivity; that
alone still does not prove geometry is unused. If geometry ablation barely
changes either reconstruction or generation, the geometry-aware claim needs
redesign before a paper narrative is chosen.

## 7. Data and annotations, separately

No additional dataset or weight download is required for Stage57. It uses the
source evaluation bank, existing source DiT/AE and the already staged full
StreamVGGT checkpoint. Original source videos are replayed only for eight
cases, with a bounded cluster cache; no full raw dataset download to this PC.

Current source training: street selection with 8,192 training IDs and four
cached windows per source (32,768 windows); Stage56 used a 4,096-window pilot.
The overall RGB cohort also contains reconstruction-scene and synthetic video
sources; this does not mean every source was used in the current street DiT.
The checked cohort counts are DL3DV 7,716 train, ScanNet++ v2 962, MVS-Synth 96,
OmniWorld 63, PointOdyssey 131, Spring 34, SpatialVID 8,448 before the narrower
street selection. These are manifest records, not independently verified
physical scenes, hours or complete original benchmarks.

Priority data needs for a publication experiment:

1. **Reliable RGB continuity and source identity:** frame indices, FPS/PTS,
   duplicates, cuts and source-level splits. The old source stored frame IDs
   and nominal FPS, not original PTS. Quantify replay discrepancies instead
   of silently relabeling new RGB with old latent targets.
2. **Motion/visibility evaluation:** if original PointOdyssey or Spring motion,
   camera, visibility and official split files survive, retain them. Our WAI
   RGB manifests do not prove those annotations survived. Inspect metadata
   before requesting a large transfer. A small verified evaluation subset is
   more valuable now than another broad unverified training mixture.
3. **Static camera geometry calibration:** DL3DV with corresponding camera
   metadata can test whether camera motion is incorrectly labeled as object
   motion. Otherwise repeated-frame and synthetic tests are diagnostic only.
4. **Optional object masks:** useful for region evaluation or loss weighting;
   not a prerequisite for the current experiment, and not a replacement for
   correspondence. Do not promise semantic masks from sparse VGGT tracks.

DynamicReplica remains excluded where the official split cannot be mapped.
No new captions are needed for the present prefix/trajectory-conditioned task.
Do not call it text-to-video without a real text condition and text evaluation.

## 8. Current delivered versus proposed

Implemented: Stage57 integrated diagnostic runner/launcher, layer/logit probes,
LK fallback **diagnostic**, quantitative readiness and replay reports,
branch/oracle/error-structure audits, inference-constructible control view v2,
and prevention of silently running a motion/reference arm with absent signal.

Not established: real NPU FP32 teacher result, LK quality on these videos,
successful v2 control training, a superior residual objective, improved
dynamic video quality, geometry-specific gains, or conference acceptance.
The current run is designed to expose these uncertainties in one submission,
not to hide them behind a completed training status.
