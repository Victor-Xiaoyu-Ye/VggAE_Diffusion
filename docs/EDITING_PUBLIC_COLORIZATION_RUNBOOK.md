# Stage71: frozen public appearance-transfer evaluation

Implemented 2026-11-13. Three nodes, eight Ascend 910B devices per node. No new
AE, LoRA, bridge or DiT training. Existing Stage69/70 outputs remain read-only.
This is a public **gray-to-color transfer** experiment, not an arbitrary
instruction-editing benchmark or an official VACE leaderboard reproduction.

## Dataset inspection and justified task

Official sources:

- [VACE-Benchmark dataset](https://huggingface.co/datasets/ali-vilab/VACE-Benchmark)
- [Official input guide](https://huggingface.co/datasets/ali-vilab/VACE-Benchmark/blob/main/UserGuide.md)
- [VACE implementation](https://github.com/ali-vilab/VACE)

The three index files contain 272 tasks: 120 real, 120 synthetic, 32 composed.
`src_video` is often a **derived condition** (depth, gray, pose, flow, etc.).
`raw_video` is the original RGB, not an edited counterpart of an RGB source.
Calling all such pairs text edits would be incorrect; treating depth as an RGB
source would test a different model interface than the one trained here.

The selected, predeclared compatible subtask is V2V/gray. All 20 such entries
were inspected, not chosen by method outputs:

| Domain | Official sample IDs | Eligible | Local decode/alignment pass |
| --- | --- | ---: | ---: |
| Real | 000040–000049 | 10 | 10 |
| Synthetic | 000160–000169 | 10 | 10 |

Downloaded only their 60 source/mask/RGB files (173.05 MiB / 181.46 MB) to
`D:/workspace/VggAE_DataAudit/vace_benchmark_20261113/`. All source channels are
gray; all official masks are white. RGB labels have matching original geometry
and physical timestamps. Source/target nominal FPS rounding is handled by
timestamp sampling, not stretching. All windows supply 17 distinct frames.

The local Stage69 screen plus cohort contains 2,199 completed exposure records.
No selected gray source matched those fingerprints, the four prior benchmark
IDs, or their indexed original-asset paths. All 20 remain eligible locally.
This does not certify scene-disjointness or unknown foundation-model pretraining
exposure. Cluster selection repeats the audit against its actual frozen bank.
Missing/corrupt sources are logged and skipped **before** generation; observed
output quality never changes selection. All exclusions remain in `cohort.json`.

Index SHA256 from the inspected official snapshot:

```text
real.txt        2b9c4fc6378270a2e6a3f049a2194861281e735b8349f6138e42dd99ebd64497
synthetic.txt   5def50c4ec23b98ee273fc1e52e5acaba17e3991f7e8722858a9911e24b4bc6c
composition.txt 7bc884fd1195916b79317dcfd08e520bcbe25c20d654b87c7a2079c28d9e0ca2
```

OBS returned 502 on this workstation during inspection. The local check used
the official Hugging Face snapshot matching the user's stated asset source;
it does not prove the cluster copy is identical. Runtime records and freezes
the actual local index hashes. The cluster never downloads from Hugging Face:
it uses `/cache/yexiaoyu/vggae_ref/VACE-Benchmark`, staging only specifically
missing named files from the existing OBS reference path.

## Frozen experiment

| Arm | Frozen weights | Native condition | Writable state |
| --- | --- | --- | --- |
| Native | Released Wan2.1-VACE-14B | Official gray video, white mask, shared model anchor | None |
| SFT | Stage69 selected matched reference-SFT | Shared propagated reference, same mask and anchor | None |
| RGB | Stage65 selected RGB LoRA + bridge | Same as SFT | RGB state, adaptive online feedback |
| RAE | Stage65 selected RAE LoRA + bridge | Same as SFT | Trained Scene C256 state, decoder calibration + adaptive online feedback |

- Parent is `editing_paper_evidence_v1`; the local export name `v2` does not
  change this durable namespace. SFT must have completed nonzero updates.
- State parent is `vace_edit_state_v1`; normalization is the TRAIN-only
  `source_scale` from `edit_transport_learned_v1`. Scene C256 and StreamVGGT
  checkpoint hashes must equal the frozen parent. No new test-data statistics.
- Source input is the official gray clip. Prompt is its official English
  output description, used verbatim, not rewritten after inspecting results.
- Generate one legal anchor using frozen native VACE from gray + description;
  its first output frame is shared by all four arms. **No original RGB frame,
  future target, target-derived mask or SAM segmentation enters inference.**
- All arms share source, anchor, official mask, sampling seed and 30-step
  schedule. SFT/RGB/RAE share the same propagated video reference. Native here
  includes that shared anchor; it is not an untouched official no-anchor run.
- RGB/RAE retain Stage69's fixed native context and Stage67 adaptive solver,
  three writeback locations, strength .35 and existing loss weights. No new
  synchronization strategy, representation tuning, or public-data training.
- Physical 17-frame/8fps window (timestamps 0 through 2s), preserving FOV.
  Native area budget is 352×640, max side768. For audited 1280×720 clips this
  gives 352×627 valid pixels on a 352×640 pad16 canvas. No square stretching.
- RAE uses the same nine observations and 518-square letterbox as training.
  Native video conditioning and exported videos retain the original aspect.
- Final MP4 panels pad an odd width by at most one pixel for the codec;
  metrics and NPZ arrays use the exact valid crop, excluding padding/text.

The expected full run is **20 anchors + 80 final videos**. Smoke uses three
preselected inputs, one on each node, six sampling steps and reduced fitting
iterations. Smoke checks execution only and is not paper evidence. Do not tune
the full experiment based on smoke quality.

## Start and recover

Use the usual project/reference staging already in the user's ModelArts
command. Once in `/cache/yexiaoyu/VggAE_Diffusion`, run this identical command
on each of the three workers:

```bash
PUBLIC_STAGE=smoke bash scripts/scale/71_audit_public_editing.sh &&
PUBLIC_STAGE=all bash scripts/scale/71_audit_public_editing.sh
```

Default namespace: `editing_public_colorization_v1`; smoke adds `_smoke`.
Default base port31500. Node addresses and ranks come from the existing
ModelArts environment; do not force all workers to rank0. No separate entry
point, dependency upgrade or full `vggae_ref` download was added.

Resume only missing work in the full run, including on fresh nodes:

```bash
PUBLIC_STAGE=resume bash scripts/scale/71_audit_public_editing.sh
```

Resume a failed smoke without accidentally changing it to the full protocol:

```bash
PUBLIC_NAMESPACE=editing_public_colorization_v1_smoke \
PUBLIC_STAGE=resume bash scripts/scale/71_audit_public_editing.sh &&
PUBLIC_STAGE=all bash scripts/scale/71_audit_public_editing.sh
```

For completed generation with only evaluation/publication failure:

```bash
PUBLIC_STAGE=evaluate bash scripts/scale/71_audit_public_editing.sh
# Or only repair/rebuild the final report and media replicas:
PUBLIC_STAGE=summarize bash scripts/scale/71_audit_public_editing.sh
```

`PUBLIC_STAGE=audit` checks indices without loading any model or parent;
`PUBLIC_STAGE=dry` prints the plan. Restore operations read the exact config
from the durable manifest, including smoke settings. Code/input/weight changes
are not silently mixed under an existing run identity.

## Reliability and resource behavior

Each phase uses a coordinated three-node job. Cases are interleaved across
nodes: smoke owners are ranks0,8,16; 20 full cases use 20 different NPUs.
Heavy models are loaded lazily only on ranks owning pending cases. Four arms
run sequentially; 14B + UMT5 + StreamVGGT are not kept together on a device.

Source decoding, source flow preparation, evaluation, and summary run as CPU
phases with Ascend autoload disabled. NPU idle guard starts before all phases
and is cleaned up on every exit. Join timeout7200s tolerates uneven bootstrap.
No 2-hour CPU job is intentionally required. GPU/NPU phases retain existing
allocator settings and one-case-per-rank inference memory behavior. Production
peak memory is recorded; this workstation cannot validate 910B peak memory.

Native/SFT/RGB are generated before RAE staging/preparation, so a codec failure
cannot erase all baselines. Individual failures are saved; other cases/arms
continue. Final incomplete coverage returns nonzero while retaining successful
videos and metrics. There are no image-quality thresholds that discard outputs.

Artifacts are written to both `$OUTPUT_URL/scale/<namespace>` and the owner
persistent OBS `output/scale/<namespace>`. Final, anchor and applied-feedback
media have size/SHA receipts. Summary reads back **both** roots and repairs
missing/corrupt replicas. Resume does not rerun valid generation or calibration.
`DI_throughput` counts newly completed frames, excludes cache hits, and states
its units; preprocessing, sampling, setup, peak memory and verification remain
separate measurements.

Plan a **few hours including smoke, initialization and OBS I/O**, rather than
a new multi-day training. This is an engineering allowance, not a measured
Stage71 runtime; the previous Stage67 five-arm/24-case run took about22m21
after bootstrap, but this run additionally constructs new anchors and states.
Use the live phase timings to revise the estimate once the cluster smoke runs.

## Outputs and interpretation

Persistent output:

```text
obs://yw-ads-training-gy1/data/external/personal/g00833899/y50046448/output/scale/editing_public_colorization_v1/
```

- `benchmark_audit.json`, `manifest.json`, `cohort.json`: task inventory,
  exact models/config/code, source screening and actual denominator.
- `render/{native,sft,rgb_full,rae_full}/seed42/<case>/`: result tensor,
  source/reference/generated MP4, lossless generated NPZ, receipts and events.
- `render/.../feedback/`: actual intermediate corrected states and three-panel
  MP4s, generated by the real sampler callback.
- `comparisons/seed42/<case>/comparison.mp4`: gray source + four outputs,
  same timestamps and valid FOV. Every selected case is present or marked failed.
- `report.html`, `summary.json`, `media_sync.json`, `tables/*.csv|tex`:
  coverage, paired statistics, failures, timing, media verification and tables
  with captions above their contents. No human-evaluation claims or rating pack.

RGB ground truth is opened only in evaluation, with the exact source physical
time and FOV. Report PSNR, SSIM, LPIPS (local AlexNet required), Lab AB chroma
error, grayscale structure, and independent source-flow temporal diagnostics.
Also report future-only frames1–16 for paired RGB metrics so gains cannot be
ascribed solely to the shared first frame. A copy-gray diagnostic exposes the
trivial no-color solution. Missing LPIPS is explicitly unavailable, never zero.
Metrics operate on decoded output tensors rather than lossy labeled MP4s.

Tables separate all observed outputs from the complete four-arm intersection,
and real from synthetic. Source-group paired bootstrap (10,000 draws) gives
RAE-minus-control differences and confidence intervals; frames are not treated
as independent samples. With only20 sources, this supplements the123-source
main experiment rather than replacing it. A positive result would support
cross-dataset appearance-transfer utility. It would not alone establish stronger
arbitrary instruction following, unique VGGT geometry, or state-of-the-art editing.

## Validation boundary

Local checks exercise actual tiny native VACE/UniPC, state calibration/alignment,
both real online solvers, MP4/NPZ writing/readback, target-absent generation,
source/mask/time/FOV rejection, cross-task/source-overlap exclusion, four-arm
coverage, cold recovery, unchanged sample hashes, media replica repair and
the actual Bash launcher with process stubs. Pretrained codecs/backbones are
explicit fixtures, not production checkpoint tests. All20 official gray/source
mask/original RGB windows were additionally decoded with the production input
and target functions. The full NPU/HCCL/MoXing execution and quality remain for
the cluster smoke/full run; no new measured paper scores are claimed here.

Validation completed: 46 CPU/Bash tests (11 new public-suite tests plus35
publication/native-runtime/joint-solver regressions); six changed Python files
compile; launcher Bash syntax and execution from another working directory
pass. Public feedback configuration exactly equals the frozen Stage69 config,
including bridge updates and decoder checkpointing. Resume preserves successful
result hashes and restores a smoke config from durable storage on a fresh node.

## Completed production run, reviewed 2026-11-14

See `EDITING_PUBLIC_COLORIZATION_RESULTS_20261114.md`. All 80 outputs on 20
sources, all 120 feedback events, and all three node exits are complete.
Local verification covers 420 receipt-declared files, 240 decoded movies and
80 lossless archives, with no empty files. Runtime source/code/checkpoint
identities match the frozen manifest and official benchmark snapshot.
RAE improves SFT LPIPS by 2.46% on all 20 sources; overall RGB difference is
uncertain, Native LPIPS is tied and Native preserves gray structure better.
The complete public-transfer result is integrated into the manuscript with
its real/synthetic strata and unchanged-gray diagnostic. Earlier pending
validation statements above describe the pre-run state, now superseded here.
