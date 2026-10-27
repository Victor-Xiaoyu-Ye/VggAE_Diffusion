# Stage57: one submission, independent chain diagnostics

## Launch

Use the same checkout on all three ModelArts nodes, with the existing
`scripts/spatialvid_config.sh` local/primary/mirror settings. No public network
or new pretrained weights are required. The existing full StreamVGGT file,
Stage53 source assets and production frozen AE are used read-only.

```bash
SCENE_AUDIT_STAGE=dry bash scripts/scale/57_audit_scene_end_to_end.sh
bash scripts/scale/57_audit_scene_end_to_end.sh
```

Default output: `scale/scene_end_to_end_audit_v1/` under local, primary and
mirror roots. This command is **diagnostic inference, not another long
training run**. It uses one independent NPU0 worker per node. All eight
allocated devices on each node remain under the existing idle guard during
staging, CPU work and peer waits. No HCCL process group waits during I/O.

The default coordinator timeout is 86,400 seconds, an upper bound rather than
an estimated duration. `SCENE_AUDIT_TIMEOUT_SECONDS` may override it. Default
RGB replay cache is 12 GiB per node. `DI_throughput` is diagnostic input RGB
frames per second including instrumentation/artifact work, not optimizer
throughput; idle-guard operations are never counted.

## Exact sources and workload

- Source: `scene_street_prefix4_48h_v1/train/weights_step0031069.pt`, online.
- Frozen AE: the exact source-pinned receipt under
  `scene_rae_c256_15day_v1/production/`; no AE replacement.
- Original source evaluation indices: 0,4,7,11,24,36,41,48. Cases 7,24,41,48
  include prior replay failures; these are not renumbered from the 50 survivors.
- Teacher cases 0 and 4: BF16 versus FP32 backbone, real versus repeated
  first-frame video, independent full9 and prefix4 passes. Heads remain FP32.
- Generator: eight cases with the production 32-step FP32 Euler state, unchanged
  statistics, observed4/generated5, seed101 and exact source model arguments.
- Twenty-four independent jobs: eight replay, eight paired teacher, eight
  generation. Partition by job index modulo three nodes.

The teacher's size and bounded sample hash must match the actual failed-run
checkpoint. This prevents silently comparing different weights as a precision
effect. A different checkpoint requires an explicitly changed configuration
and namespace. Full source DiT/AE/eval payloads retain SHA/size receipt checks.

## What is saved

`summary.json` separates execution failures, support/utility findings and
visual-quality review. Each component saves `report.json`; successful jobs
also have `done.json` with hashes for their files.

- `teacher/{bfloat16,float32}/{real,repeated_first}/caseNNN/`: bounded
  intermediate activation/logit statistics, final score distributions,
  full/prefix teacher tensors, controls, track-support overlay, LK alternative
  tensors/overlay, and reference-versus-copy/wrong-destination errors on the
  exact same supported region. LK uses forward/backward, texture and
  photometric tests; scores are engineering quality values, not calibrated
  neural confidence or semantic masks. Camera/depth still come from Stream.
- `replay/caseNNN/`: RGB mismatch magnitudes, affected fractions and original
  frame provenance. Replayed RGB never replaces the sealed latent/raw target.
- `generation/caseNNN/`: labelled RAW / AE_GT_RECONSTRUCTION / GENERATED / COPY
  video, separate visibly labelled GT-geometry/GT-texture oracle videos,
  generated latent, Euler traces, supervised noisy-target probes, and clean
  versus sampled branch interventions. Structured generated errors versus
  equal-RMS Gaussian errors are counterfactual decoder probes, not outputs of
  a new trained model.
- `nodes/nodeN/`: pinned source/code/runtime-version contract, attempted-work
  completion and device memory reports.
- `launcher/nodeN/`: live log, effective config, phase/exit codes, idle guard.

Red/green track dots mean rejected/accepted correspondence support, not
dynamic/static regions. A static textured building can have valid moving
pixel tracks under camera motion; it must not be called a dynamic object on
that basis. The repeated-frame control and geometric residual diagnostics
check different aspects of this problem.

## Failure and resumption

Captured teacher/precision/LK/replay/generation failures do not cancel the
other components. Missing teacher weights still allow cached AE/DiT tests.
OOM is a failed precision test, not evidence that FP32 has bad tracks.
Unrecoverable source contract/data failure remains fatal because diagnostics
cannot be interpreted against an unknown input.

The merge phase runs even after a worker process failure and writes partial
summary when its source metadata remains readable. Missing or failed components
make final merge nonzero. All-zero tracks can be a **completed diagnostic with
invalid conditions**; exit0 never certifies useful motion control or quality.

Successful components are content-verified and recovered from either durable
copy on restart. Failed jobs retry. A corrupted prior component is recomputed
without canceling independent jobs. Code/config/source changes require a new
`SCENE_AUDIT_NAMESPACE`; identical interrupted jobs keep their existing one.
Source files are never cleaned or overwritten. Small receipts stay with outputs.

## Training readiness and the new control policy

Preparation now records train/eval support separately and preserves actual raw
teacher score statistics where available. Individual unsupported clips are
legal. At Stage56 training startup, an arm with no evidence of its required
motion/reference signal writes `skipped_invalid_controls.json` and returns
consistently on every rank; baseline A and independent work continue. This is
experiment validity, not a PSNR gate. Old caches without readiness metadata
can be scanned at eval plus up to four training shards; zero in a bounded
scan is explicitly **unverified**, not a claim about the entire dataset.
Completed historical runs remain unchanged.

Legacy `scene_control_reference_v1.json` retains its historical teacher-oracle
semantics. The separate `scene_control_reference_v2.json` enables
`supplied_camera_tracks_visibility_only_v2`: use supplied camera rays, point
paths and binary visibility; preserve observed-only reference appearance;
do not pass future-depth-derived dynamic/static labels or uncalibrated score
magnitudes into the adapter. Unknown remains unknown. This input interface
can be reproduced without future images, provided the user supplies paths
and visibility. Producing a physically sensible path/visibility plan is a
separate problem; the model does not infer it automatically.

Do not rerun the old 4,096-window controls merely to produce another zero bank.
First inspect Stage57 support, overlays and same-region transport utility.
The LK alternative is not automatically enabled for training. If it is judged
usable, the preparation backend must be selected explicitly and its source
identity sealed before a matched training pilot. The new v2 policy already
has CPU interface/invariance tests, but no real v2 training result is claimed.

The separate camera-only arm is available with exactly the same adapter size
as motion/reference; it never reads point paths or appearance. It is allowed
to train when camera rays exist even if the tracker has no support. After a
usable teacher/backend has been selected, the explicit four-arm training
configuration is:

```bash
SCENE_CONTROL_CONFIG="$(pwd)/configs/scene_control_reference_v2.json" \
SCENE_CONTROL_NAMESPACE=scene_street_control_reference_v2 \
SCENE_CONTROL_ARMS="baseline camera motion reference" \
bash scripts/scale/56_train_scene_control_reference.sh
```

This is a prepared follow-up interface, **not the recommended immediate run**:
with the present failed tracker, motion/reference still skip and camera-only
cannot demonstrate object-motion control. The immediate command remains Stage57.
Do not reuse a v1 namespace with the v2 policy.

## Reading the result

1. If BF16 fails but FP32 works on the same repeated/real clips, inspect the
   first discrepant layer; measure the cost of a targeted precision change.
2. If both fail even on repeated frames, investigate checkpoint/head behavior
   and compare LK evidence; do not lower thresholds until weights appear useful.
3. If correspondence exists, compare transported prefix texture to last-frame
   copy on identical visible regions. Better supported-subset error does not
   prove full-video improvement or correct handling of new entrants.
4. If oracle future texture helps but geometry does not, use branch/error
   diagnostics to distinguish texture predictability and decoder sensitivity.
   Neither oracle is an honest generated result.
5. If adequate conditions and representation still fail, move to a declared
   native appearance-prior baseline; avoid another blind full-HQ scaling run.

Publication hypothesis, baseline requirements and separately listed data needs:
`SCENE_RESEARCH_AND_EXECUTION_20261027.md`.

## Local verification on 2026-10-27

129 relevant CPU/Bash tests passed together, including real tiny-model
optimization/EMA, two-rank consistency, mirror recovery, future-appearance
isolation, v2 information invariance, camera-only parameter/input matching,
teacher/LK probes, error-matched decoder diagnostics, partial-failure/resume
and the actual launcher with stubbed cluster operations. All 20 changed Python
files compile; both Stage56/57 Bash syntax checks and git diff checks passed.

The two actual user-supplied teacher payloads were also passed through the new
readiness/v2 view: each still has zero future tracks, reference pairs and motion
support, with 1,620 camera-present future token slots. This is the expected
honest result, not a repaired tracker. Record:
`D:/workspace/VggAE_DataAudit/scene_control_20261027/readiness_v2_actual_teacher.json`.

Local tests do not certify actual NPU operators, full-checkpoint FP32 memory,
remote publication availability, LK accuracy on the real clips or image quality.
Those measurements are the purpose of the next Stage57 run.
