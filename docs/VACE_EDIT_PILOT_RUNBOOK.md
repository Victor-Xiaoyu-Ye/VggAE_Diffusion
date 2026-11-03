# Stage60: native VACE editing pilot (2026-11-02)

## What this run answers

First establish instruction-conditioned appearance editing with a pretrained
native editor. Then measure whether edited-target Scene C256 RAE supervision
improves its **native video output** over matched LoRA SFT. This is RAE-assisted
post-training, not diffusion in RAE space, not yet a new geometry method, and
not a claim of publishable quality. A negative RAE result should change the
method, not launch another series optimizing only an auxiliary reconstruction.

The user explicitly permits changing the initial task/method if construction
reveals a mismatch. No from-scratch, no-VAE or tiny-backbone constraint applies.

## Assets supplied by the user

- Data: `obs://yw-ads-training-gy1/data/external/personal/g00833899/y50046448/dataset/Senorita-pilot`
- Existing startup stages `vggae_ref` to `/cache/yexiaoyu/vggae_ref`:
  `Wan2.1-VACE-14B`, `VACE-Benchmark`, `sam2.1-hiera-large`.
- Frozen teacher: existing `scene_rae_c256_15day_v1/ae/best_joint.pt`, plus existing
  `StreamVGGT/checkpoints.pth`. No new encoder/head download is required.
- SAM2 is inventoried but not invoked. It needs an object prompt/seed; inventing
  a target-difference mask would leak the edited answer into the condition.

These OBS assets are user-reported. This workstation has not re-audited their
remote bytes. The entry point checks local staged assets on each node. Native
config, all checkpoint shard names, tokenizer availability and sampled weight
signatures are recorded; signatures are not whole-file checksum verification.

## Actual data contract, including resolution and aspect ratio

The local `4.tar.gz` really contains `./4/<hash>_org.mp4` and `<hash>.mp4`.
The official metadata uses `local_style_transfer/4/<hash>...`, so lookup uses
the package+filename suffix, not basename alone. Only the twelve agreed pilot
archives are staged (4,14,30,50,70,90,110,130,150,170,180,193), plus metadata.
There is no recursive full-Señorita download. Missing metadata entries outside
these packages are expected and counted separately from decode failures.

Twelve checked pairs in package4 all decoded successfully at 8 FPS and 336×592
pixels (width×height); this is a same-package check, not a dataset-wide sample.
The first has source33/target34 frames. Metadata has instruction/enhanced_instruction
and source/target paths; it has **no descriptive target caption**.

- Señorita: first 17 timestamps at 8 FPS (2 seconds first-to-last).
  No duplicate-frame upsampling, no stretching an entire clip to this duration.
- Official benchmark development cases: 17 timestamps at 16 FPS. Their native
  descriptive prompts and provided masks remain distinct from edit instructions.
- Decode and align both streams by timestamps. Permit a short trailing-duration
  difference of at most two frames; require equal source/target pixel sizes.
  Record counts, FPS and selected frame indices. Unverified spatial remapping,
  short/low-FPS/corrupt clips are logged and skipped, not silently repaired.
- Native editor input has a **maximum area of 352×640**, maximum side 768.
  Preserve the original aspect ratio, never upscale native RGB, and add only
  the padding needed for spatial multiples of 16. This is not a fixed landscape
  crop and not a claim to high-resolution quality. Square/portrait/wide inputs
  receive different tensor shapes under the same area budget.
- The inspected 592×336 pairs fit this budget and are already multiples of16,
  so they retain **all their original resolution**, with valid rectangle
  `[top=0,left=0,height=592,width=336]`. The earlier 256×448-area proposal was
  replaced after actual data inspection to avoid throwing away available detail.
- Previews and pixel metrics crop back to that valid rectangle. MP4 encoding
  pads at most one pixel for even dimensions, without 16-pixel macroblock resize.
- The production RAE needs 518×518. Its edited-target input is aspect-preserving
  letterboxing of the valid image, not stretching. Its content may be enlarged
  to meet the fixed encoder input; this does not add target detail. Both latent
  and decoded-RGB auxiliary losses exclude letterbox padding. Frozen-AE dev
  reconstruction videos/PSNR are saved to expose any new-domain teacher weakness.

Source-byte hashes group edit variants before split: nominal 2048 train pairs,
24 dev, 128 reserved test, with whole-group rounding. Test is not cached/trained.
Byte hashes cannot detect re-encoded duplicates or certify physical-scene
independence. This is internal development, not a formal public test protocol.

## Model and controls

Official native VACE interfaces are retained:

`source RGB + instruction → native 96-channel context + UMT5 → VACE → native Wan VAE → edited video`

The 96 channels are 16 inactive VAE channels + 16 reactive VAE channels + 64
packed mask channels. Even an all-zero inactive image is VAE-encoded; its latent
is not filled with zeros. Señorita uses an all-white generation mask and no
target reference. Target RGB is only the supervised answer. VACE-Benchmark
development cases use their provided source/mask and descriptive output prompt;
reference-image cases are excluded from this first implementation and reported
as limited task coverage. The benchmark raw video is not edited-target GT.

Train `x_sigma=(1-sigma)*z_target+sigma*noise`, velocity `noise-z_target`, timestep
`1000*sigma`, shifted uniform sigma with shift 3. Inference uses native UniPC,
30 steps, CFG 5, same condition for positive/negative branches. Drop text to
empty prompt at probability 0.1 during SFT. This explicitly adapts instructions;
instruction zero-shot performance is not the native model's quality ceiling.

Arms:

| Arm | Trainable components | Supervision / output |
|---|---|---|
| native | none | Same source/seed; reduced-resolution native inference |
| sft | rank-16 LoRA on self/cross attention Q/K/V/O in base and VACE blocks | Native velocity MSE; native output |
| rae | same LoRA + layer-19 hidden→C256 readout | Same MSE + edited-target RAE latent and frozen-decoder RGB auxiliary losses; native output |
| rae_detached (optional) | same LoRA + detached readout | Head-only negative control; native output |

RAE readout maps to nine 18×18×256 frames. The first 50 steps train that readout
without its gradient entering the backbone; afterward auxiliary gradients enter
LoRA at sigma ≤0.7. Decoder parameters remain frozen, but predicted-input
decoding is differentiable. No auxiliary RAE RGB is substituted for the final
native output. Separate gradient clipping prevents the detached readout from
changing LoRA indirectly. Data order, LoRA initialization, noise and text drops
match across SFT/RAE arms. This still needs evaluation, not only a falling loss.

## Memory and runtime

Official configuration instantiated on meta tensors gives **17,337,592,896**
parameters including the VACE control branch; “14B” names its base model.
BF16 storage alone is about 32.29 GiB, plus FP32 time/head/norm/modulation islands.
Rank-16 LoRA adds **62,914,560** trainable parameters. Its FP32 parameters,
gradients, Adam states and EMA are about 1.17 GiB before DDP buckets/allocator.
Per-block non-reentrant activation checkpointing, batch one per NPU, and cached
T5/VAE inputs keep live training memory bounded. T5 and VAE encoding run in a
separate phase; the main training process does not retain T5/StreamVGGT.

Load native weights from meta tensor-by-tensor directly to the assigned NPU;
do not hold eight complete FP32 CPU model copies per node. Replicated DDP skips
its initial 32+ GiB parameter broadcast after cross-node identity checks.
**The full 14B accelerator peak is not measured locally**; two-update preflight
logs peak allocated memory before the long phase. No automatic OOM fallback
silently changes resolution or LoRA rank between comparison arms.

Default is three nodes × eight 910B, 1000 optimizer steps per arm, effective
batch 24. This is a pilot budget, not a convergence assertion. Every 100 steps
checkpoint, every 250 steps RAW/EMA previews, and final full 24-case development
evaluation. The smoke has three updates and four-step samples: correctness only.

## Launch using the existing cluster startup

After the usual repository/environment/vggae_ref staging, execute the same
commands on all three nodes through the existing ModelArts launcher:

```bash
cd /cache/yexiaoyu/VggAE_Diffusion
export EDIT_JOIN_TIMEOUT_SECONDS=3600
EDIT_STAGE=smoke bash scripts/scale/60_train_vace_edit_pilot.sh && \
EDIT_STAGE=all bash scripts/scale/60_train_vace_edit_pilot.sh
```

The `&&` starts all only after smoke succeeds. Do not place
bare smoke/all commands on consecutive lines: an outer shell without `set -e`
will continue to all after smoke fails. To run only smoke while investigating:

```bash
cd /cache/yexiaoyu/VggAE_Diffusion
EDIT_STAGE=smoke bash scripts/scale/60_train_vace_edit_pilot.sh
```

Smoke uses namespace `vace_edit_pilot_v1_smoke`. For the real pilot:

```bash
EDIT_STAGE=all bash scripts/scale/60_train_vace_edit_pilot.sh
```

The all entry stages data, runs small on-NPU native forward/backward checks,
caches on all 24 ranks, samples the native editor, prepares the historical RAE
teacher, executes two-update tests for both arms, then **starts fresh processes
and resumes each arm** to 1000 steps. Both short tests precede either long arm.
If a teacher/arm fails, independent SFT work continues, completed results remain,
and the final exit is nonzero. Numerical/setup failures remain errors; quality
scores never gate the next legal phase. Re-running the same command resumes.

### Coordinator disconnects (2026-11-03)

The complete three-worker logs identify this incident's cause: node0 timed out
in TCPStore construction after 601s with only 2/3 clients joined. Worker2 was
still in environment/asset setup; one OBS copy alone took 1946.14s. Worker1 had
already finished smoke indexing (48 train / 4 dev / 8 test) because TCPStore's
implicit worker wait blocks the owner but not clients. When node0 timed out,
worker1 lost its socket. This was before model/cache training, not an OOM or
two-hour idle reclaim. The user's consecutive smoke/all commands then allowed
all to start despite the smoke error, explaining the subsequent manifest change.

The coordinator now starts a listening store without that asymmetric implicit
wait and explicitly registers every rank before starting ANY child. Joining
logs and phase JSON show joined/missing nodes, hostname, endpoint and run ID.
Stage60 allows 3600s for connection/registration (`EDIT_JOIN_TIMEOUT_SECONDS`),
separate from its child-runtime budget. Existing idle guards stay active during
this wait. Duplicate ranks and mixed smoke/all namespace/phase registrations
are rejected with a shared original error before doing work. On join timeout,
connected peers acknowledge that error before node0 closes the store.
Setup before entering this shell still belongs to the external cluster startup;
this change does not speed up downloads or install an idle guard in that code.

`TCPStore.check: failed to recv, got 0 bytes` means the store connection closed;
it is not evidence of a diffusion loss failure. The old `node1 exit=0` final
line could merely describe its own child process even while the coordination
raised an exception. The new log distinguishes `local_exit`, the overall `exit`,
observed peer results and `all_results_observed`. Node0's earlier exception or
termination remains necessary to identify an incident's original cause.

A command launch exception now participates in result/ack shutdown, keeping
the node0 server alive until live peers receive the failure. Normal shutdown
already had an ack barrier; this correction does not prove that an arbitrary
disconnect was caused by a normal-shutdown race or fix external node death.

Every node writes `launcher/nodeN/coordinator_phase_<offset>.json`, mirrored by
the existing launcher publisher. It identifies the Python/shell entry, stage,
arm, local and phase outcomes, TCP endpoint and original exception. No model,
data, cache identity or checkpoint format changed. Update all three nodes and
retry the same stage/namespace; do not erase cached work for this change.

Useful overrides (same values on all nodes):

- `EDIT_STAGE=smoke|all|prepare|native|train|dry`, `EDIT_ARMS="sft rae rae_detached"`.
- `EDIT_JOIN_TIMEOUT_SECONDS=3600` for slow per-node environment/asset staging.
- `EDIT_CONFIG=/path/config.json`, `EDIT_NAMESPACE=new_name` for changed protocol.
- `EDIT_MODEL`, `EDIT_BENCHMARK`, `EDIT_DATA_SOURCE`, `EDIT_DATA_LOCAL`.
- `EDIT_RAE`, `EDIT_RAE_URL`, `EDIT_RAE_MIRROR_URL` for the existing teacher.

The cluster never reaches HF. SAM2 is not a required precondition for these arms.
No remote training has been submitted from this workstation.

## Outputs, recovery and first review

Run output is `$OUTPUT_URL/scale/vace_edit_pilot_v1/`, mirrored to:

`obs://yw-ads-training-gy1/data/external/personal/g00833899/y50046448/output/scale/vace_edit_pilot_v1/`

Cache is mirrored under the run's `cache_latents` and persists at:

`obs://yw-ads-training-gy1/data/external/personal/g00833899/y50046448/cache_latents/vace_edit_pilot_v1/`

- `bank.json`, `coverage.json`, `runtime_preflight/`, `teacher_bank.json`,
  `teacher_previews/` are in the cache roots; tests are not cached.
- `native/`, `sft/`, `rae/` each contain evaluation JSON and labelled videos.
- `sft|rae/samples/s000002/raw|ema/` is the first actual trained result.
- Every comparison video has **SOURCE / TARGET / NATIVE VACE OUTPUT**, with the
  RAW/EMA variant in its directory. Benchmark raw-reference panels are labelled
  separately and are not edited ground truth. Frozen-RAE preview has its own label.
- `metrics.jsonl`, TensorBoard, `checkpoint_latest.pt`, immutable periodic
  checkpoints, per-case sampled native latents, and launcher/NPU logs are saved.
- A committed pointer selects an immutable checkpoint for resume; a partially
  overwritten latest alias does not destroy that checkpoint. Hash/receipt-based
  dual read falls back to the other root, and replica failures remain visible.
- `DI_throughput` in training is **global target video frames/s**, including IO,
  evaluation and checkpoint wall time since that process resumed. Preprocessing
  and sampling metrics explicitly say per-rank frames/s; do not add them blindly.
- The job-scoped idle guard covers all eight assigned NPUs per node throughout
  staging and waits, and is cleaned up with the log publisher when the job ends.

First assess motion preservation, intended edit execution, unedited content,
flicker/shape degradation, and portrait/wide/square cases. Pixel PSNR to an expert
target and temporal frame delta are diagnostics only, not a geometry metric or
an adequate editing evaluation. Later claims require formal held-out editing
metrics, human preference, independent geometry evaluation and GAD/GF/DINO/extra
RGB auxiliary comparisons. A better auxiliary reconstruction alone is a failure
of the proposed transfer, not a successful paper result.

## Validation and sources

Local checks cover actual tiny VACE forward equivalence, sharded meta loading,
LoRA and RAE gradient paths, detached control, mask packing, time/size/aspect
handling, source-group separation, checkpoint/FP32 EMA recovery, dual read,
three-node shell phase ordering/failure cleanup and fresh-process two-rank Gloo
training/resume. Actual 910B attention backward, HCCL, full-model memory, MoXing,
remote data availability and output quality still require the cluster smoke.

Primary interface references:

- [Official VACE](https://github.com/ali-vilab/VACE)
- [VACE input semantics and descriptive prompts](https://huggingface.co/datasets/ali-vilab/VACE-Benchmark/blob/main/UserGuide.md)
- [Native VACE-14B configuration](https://huggingface.co/Wan-AI/Wan2.1-VACE-14B/blob/539c162b1387eac9dc4c20bd3f74671309e76a4c/config.json)
- [Señorita dataset](https://huggingface.co/datasets/SENORITADATASET/Senorita)
