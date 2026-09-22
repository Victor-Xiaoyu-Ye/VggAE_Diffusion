# Full-HQ scale review and camera-conditioned pivot

Evidence cutoff: 2026-09-22 11:48 China. Run `r7_fullhq_ti2v_1650m_v1`.
This is an interim review, not a completed100K training result. No running job
was stopped or changed. Evidence is stored locally under
`D:/workspace/VggAE_DataAudit/fullhq_review_0921` and `fullhq_review_0922`.

## What this run actually shows

The24-NPU1.653B run has reached14490/100000 updates; globalbatch192 means2782080
window exposures. The completed cache contains1440144 usable windows and19256
failed windows (1.3194%), so the optimizer has consumed about1.932 cache-equivalent
passes. This differs fundamentally from the prior178M single-domain6000run's
70.3 exposures/window. Three nodes and all-domain data are functioning; this is
not another AE compatibility failure. Fixed32clip AE replay is22.41186dB with
zero per-clip difference from the reviewed reference.

All rows below are EMA,32same heldout street clips x2seeds, textCFG3:

| Step | RGB L1 vs RAW | Latent MSE | Clips beating repeated first frame |
|---:|---:|---:|---:|
|2000|.133931|.630183|10/32|
|4000|.126818|.574761|17/32|
|6000|.123955|.560336|18/32|
|8000|.122922|.555798|21/32|
|10000|.121563|.548661|23/32|
|12000|.121146|.539515|24/32|
|14000|.121737|.544566|24/32|

Repeated first-frame RAW L1=.126585; AE reconstruction RAW L1=.046466.
Best observed12K beats copy by4.30%;14K by3.83%, with a0.487% regression from12K.
Training loss still improves: mean logged5–6K=.44569,11–12K=.37779,13–14K=.36760.
Mid-noise x0 MSE improves more than pure-noise prediction. This is learning, but
not a visual breakthrough or proof that scaling is useless. Pure-noise u=1 MSE
is.304906 at14K, compared with.305720 at12K: improvement there is already small.

Visual review covered EMA seed101 clips0/3/5/7, full nine-frame grids at6K and14K.
14K has more stable gross layout in some scenes, but architecture edges still
deform, foliage smears and the car/person scene loses local shape/detail. Copying
the first frame keeps sharpness but does not produce the correct future. Limited
previews do not certify all32clips; none of these counts is a perceptual pass rate.
Only street regression data are evaluated, not complete HQ domain coverage.

Recent steps average about8.37seconds and1242.27tokens/s/NPU; allocation peak
40.600GiB/reserved40.902GiB. These timings exclude full preprocessing and periodic
evaluation/publication. Checkpoint objects were listed, not downloaded locally;
best *observed*12K does not independently inspect the internal step of a live
best checkpoint. The new launcher reads and freezes its actual internal step.

Two comparison caveats matter. Old/new runs change model size, data, text, CFG,
and exposure count simultaneously. Also current sampled metrics use CFG3 while
one-call noise probes use the raw model (CFG1); their MSE gap cannot diagnose a
sampler bug. New paired camera experiments use CFG1 for both.

## Why pose conditioning is a useful next test

First image plus a scene caption leaves both scene completion and future camera
motion uncertain. Supplying a path removes part of that uncertainty. It does
not recover unseen geometry, train missing appearance priors, or determine how
people/cars move. Deformation is not proof that absent camera motion is its only
cause. The useful hypothesis is narrower: on matched data and compute, knowing
the recorded camera path makes RGB generation materially easier.

Choose continuous first-anchor-relative camera poses internally. WASD and
mouse/rotation controls can later be integrated into the same trajectory using
speed, angular velocity and timestamps. This preserves compound movement and
magnitude, whereas a forward label alone omits speed, turn angle and intrinsics.
Street/drone pose units may not share a verified metric scale. This pilot keeps
source units and uses one train-derived scalar for conditioning standardization;
it is not depth-based scene-scale calibration or a meters-per-second controller.

Actual OBS annotation audit: `group_0001/e1e1a9a9-70fa-55f4-a68d-55ae75eaeacd`
contains38poses and38intrinsics. `indexes.txt` maps ordinal0..37 to original RGB
frames0,5,...185. Nine roughly8fps sampled RGB frames require interpolation, not
pose array slicing. Parser, c2w-center interpolation and rotation SLERP are tested
against this file. The official format is w2c OpenCV `[t, qx,qy,qz,qw]` and
normalized intrinsics. Source: [SpatialVID data card](https://huggingface.co/datasets/SpatialVID/SpatialVID).

The proposed control is `C_anchor^-1 C_t`, plus K and elapsed time. The historical
R7 temporal codec combines pairs of future RGB frames and mixes time through
legacy normalization. Therefore each of four latent slots receives anchor plus
both corresponding future poses. This is a window-conditioned generator, not yet
a causal frame-by-frame world state. No future RGB, depth or VGGT features are
added as conditioning; recorded future pose is explicitly the commanded input.

## What the literature changes about the project story

| Primary source | Relevant lesson |
|---|---|
|[CameraCtrl II](https://arxiv.org/abs/2503.10592)|Pose injection and iterative dynamic scene exploration are established; camera control alone is not novelty.|
|[GEN3C](https://arxiv.org/abs/2503.03751)|Project a3D history cache into the target camera before generation; geometry can preserve observed content rather than remain an abstract condition.|
|[Yume](https://arxiv.org/html/2507.17744v1)|Real-world walking videos can supervise discrete movement, but its pretrained SkyReels14B initialization is a major difference from scratch R7.|
|[Matrix-Game2.0](https://arxiv.org/html/2508.13009v3)|Its1.8B system starts from pretrained SkyReelsI2V1.3B, then undergoes substantial action training/distillation; similar parameter count does not imply similar starting knowledge.|
|[WorldPlay](https://arxiv.org/abs/2512.14614)|Continuous/discrete control and long-term geometric consistency already coexist; a keyboard interface is not a research contribution.|
|[WorldMem](https://arxiv.org/abs/2504.12369)|Historical images/pose/time memory matters when returning to previously seen places.|
|[RealCam-I2V](https://arxiv.org/abs/2502.10059)|Camera translation needs a scene-scale convention; rotation is not the only control issue.|
|[VGGT-World](https://arxiv.org/abs/2603.12655)|VGGT-based future geometry prediction exists; strong controllable RGB is a distinct and harder objective.|

A defensible longer-term direction is **geometry-grounded revisitable scene
exploration**: use VGGT geometry, confidence and view coverage to decide which
observed content should be reused, which history should be retrieved, and which
unseen/occluded region needs synthesis. Test backtracking, closed loops and
cross-view return, not only forward motion. This is a contribution hypothesis,
not established novelty: compare with pose-only, RGB/depth projection and generic
memory baselines under the same RGB generator and budget.

If correctly conditioned R7 still cannot produce satisfactory RGB, quality-first
development should preserve a pretrained video generator's native codec/interfaces
and use VGGT for geometry/memory/visibility conditions. Loading Wan DiT weights
into a different R7 latent space did not preserve that native generation prior;
its failure does not invalidate native-codec geometry conditioning. This route
still needs a budgeted implementation and fair baseline, not another unsupported
promise that a module will solve quality.

## Concrete next experiment and decision

Stage50 implements a bounded matched pilot; see CAMERA_PILOT_RUNBOOK.md.
Both arms start from one frozen shared EMA and the same audited approximately6K
cached windows, same2000updates and seeds. One arm supplies recorded paths with
10%camera dropout; the other trains the identical adapter with explicit absent
control. Online/EMA recorded-path previews plus EMA null/wrong-path previews are
saved. L1 remains a paired diagnostic, not the sole quality criterion.

Promote to full-HQ camera training only if recorded paths produce visibly more
coherent/sharp video than the matched null arm, and changing paths changes motion
without destroying appearance. If pose is ignored, diagnose injection/alignment;
if pose is followed but RGB remains broken, spend the next major budget on a
stronger appearance prior/geometry-guided reuse. A negative small pilot alone
cannot rule out all camera-conditioned architectures.

The present deliverable is not a real-time world model: it has no interactive
keyboard frontend, causal long-horizon rollout, persistent scene memory, or
validated physical interactions. Those claims require separate demonstrations.
