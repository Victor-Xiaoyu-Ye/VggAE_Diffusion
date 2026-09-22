# Stage50: recorded-camera versus absent-camera pilot

2026-09-22. This is the user's next camera-control experiment, not a finished
interactive world model. Frozen R7/legacy AE and expensive full-HQ latent caches
are reused. Existing stage49 training is not stopped, overwritten or resumed by
this launcher. Use separate jobs/resources, or finish/stop that job yourself
before reusing its NPUs.

## What is compared

Both arms share one frozen source EMA, selected latent shards, metadata, accepted
camera windows, normalization, random seed and optimizer schedule. Default source
is stage49 `checkpoint_best_reconstruction.pt` at preparation time; the launcher
reads its internal step and records hashes. It does not assume6000/12000 or fetch
a new moving best for each arm. Source checkpoint is reduced on the cluster to
EMA-only initialization (~6.6GB), with immutable receipt and double publication.
This is model initialization with a new optimizer, not full training resume.

Select96 existing tar shards deterministically (roughly6144 candidate windows),
audit their poses and retain the same valid subset for both arms. Keep all128
existing eval windows valid so the exact32clip AE reference cannot silently
change. Preparation stops with a failure report if camera coverage/alignment is
insufficient. No new AE encoding and no raw video download are required.

- `pose`: continuous recorded trajectory with10% explicit camera dropout.
- `null`: same network/parameter count/RNG calls but all camera presence bitsfalse.
- Both:24NPUs, batch1/device, accum8/global192,2000updates, LR2e-5,warmup100,
  uniform x0, frozen legacy AE, BF16, EMA.999, text dropout.1, text CFG1.
- Optional camera MLP adds2,428,416parameters (<0.2%) to the1.653B DiT. Its final
  layer starts atzero so initialization preserves the source model's output.
  First update/log/eval memory acceptance uses52GiB; actual camera NPU execution
  remains unvalidated. Base full-HQ training peaked at40.6GiB. Extra persistent
  model-state estimate is about0.045GiB/device, not an end-to-end memory guarantee.
- Full DiT fine-tunes in both arms. Adapter-only training is not this experiment.

Controls use first-sampled-camera-relative c2w poses,6D rotation, translation,
normalized intrinsics and time:9x14 features. Each of4latent slots receives anchor
plus both RGB frames represented by that slot. Camera null has an explicitpresence
bit and is different from a stationary camera. A fixed training-only translation
scale preserves speed differences; source units are not verified metric units.
Sparse annotations are aligned by indexes.txt; camera centers interpolate linearly,
rotations by SLERP, with no extrapolation and a bounded gap. Original resized RGB
is518square without cropping, so normalized K stays unchanged.

## Launch on exactly three eight-NPU nodes

Use the existing bootstrap and branch `ascend-910b`. Run the same launch command
on each node. First freeze and audit shared inputs once:

```bash
CAMERA_STAGE=prepare bash scripts/scale/50_train_camera_pilot.sh
```

Then run the two training jobs, each on3x8 NPUs. Sequential jobs are sufficient:

```bash
CAMERA_ARM=null CAMERA_STAGE=train bash scripts/scale/50_train_camera_pilot.sh
CAMERA_ARM=pose CAMERA_STAGE=train bash scripts/scale/50_train_camera_pilot.sh
```

Do not simultaneously create the shared input namespace for the first time.
`CAMERA_STAGE=all` prepares/reuses shared inputs then trains the selected arm.
Preparation uses leader CPU/I/O while other nodes remain alive through a TCP
coordinator; actual training uses24NPUs. Staging waiting never holds a long HCCL
barrier. The temporary26GB optimizer-bearing source is removed after the compact
snapshot is committed locally; ample cluster staging space is still required.
Progress and failed-window audit snapshots are also copied into the watched
node logs, so a preparation failure leaves a detailed report in both output roots.

For an interrupted arm after it has saved a checkpoint:

```bash
CAMERA_ARM=pose CAMERA_STAGE=train RESUME=1 bash scripts/scale/50_train_camera_pilot.sh
```

Keep its initialization, camera bank, arm, topology and schedule unchanged.
`STOP_AFTER_STEPS=200` is an optional early pause, not a change to2000-step LR
schedule. Do not use RESUME=1 merely because the shared inputs already exist.

## Outputs and how to judge them

- Shared immutable inputs: `output/scale/r7_camera_pilot_inputs_v1` under both
  current output and persistent mirror. Inspect `initialization.json` for actual
  parentstep/SHA and `camera_audit.json` for accepted windows, failures and scale.
- Runs: `r7_camera_pilot_null_v1` and `r7_camera_pilot_pose_v1`.
- Checkpoints/previews every500updates; same double read/write, numeric JSON
  metrics and `DI_throughput: <value> tokens/s/npu` convention as stage49.
- Pose run evaluates `online` and `ema` under recorded paths, `ema_null` with
  trained missing-control semantics, and `ema_wrong` with a deterministic path
  from another heldout clip. Wrong paths replace pose only; recipient K/time stay
  fixed. The wrong-path comparison is a conditioning diagnostic, not a valid
  ground-truth pixel-quality comparison. Both text-CFG branches, if enabled in
  other runs, keep the identical camera and first image.
- Preview grids show AE, copy, generated and raw. Matched no-camera training is
  the null *arm*; the pose model's `ema_null` is an additional ablation, not a
  substitute for that independently trained control.

Look for sharper, stable structure and correctly changed viewpoint at matched
noise, not only falling L1. If all controls produce the same motion, the branch
is ignored; if changing the path changes motion but images still break, movement
uncertainty was not the only bottleneck. Paired RAW error alone can reward freezing
or blurring. Perceptual/estimated-pose metrics are not yet added automatically;
this pilot retains existing metrics plus explicit visual control comparisons.

The small cohort tests whether control helps at a finite cost. Its negative result
cannot rule out stronger spatial camera encodings, better poses or pretrained RGB
priors. Promote to full-HQ camera training only after this evidence is reviewed.
Keyboard UI, user-trajectory inference CLI, persistent memory and long rollouts
remain future work; current inference evaluation consumes heldout recorded paths
and explicit null/wrong controls through the production sampler.

## Local validation (2026-09-22)

49 relevant CPU tests passed: camera coordinate math, exact-frame joins,
conditioning and CFG behavior, model-only initialization, deterministic full
resume, preparation failure ledgers, interrupted double-publication recovery,
and the existing full-HQ/window trainer. A separate three-process CPU TCP staging
exercise passed. Changed Python modules compile and both shell launchers pass
`bash -n`. These checks do not establish NPU/HCCL/MoXing compatibility or quality.

The actual full HQ CSV has 365,362 unique video IDs across 74 groups; all paths
match the annotation grouping rule. Its SHA256 is
`96d920aa1951e356cbc5cbee91ea1299b34c55c75584f4b45ba77b408d607ada`.
Four windows sampled with the production frame-index function from the real
29.970 FPS example all produce finite float32 `[9,14]` controls. The full selected
96-shard coverage audit still runs on the cluster during `CAMERA_STAGE=prepare`.
