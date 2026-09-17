# Full-HQ frozen-R7 TI2V, three nodes

2026-09-17. User-approved scale baseline, not a demonstrated quality improvement.
Entry: `scripts/scale/49_train_fullhq_ti2v.sh`. Exactly 3 nodes x 8 Ascend 910B.
CPU tests pass; real NPU memory, HCCL, full-HQ I/O and generation remain untested.

First cluster attempt exposed an Ascend FlashAttention mask shape incompatibility
([1,1,1,256] was rejected for1296 queries). Attention now uses contiguous[B,1,Q,K]
boolean masks, preserving valid-token semantics and parameter shapes. This trace
does not establish OOM. Pull the fix and rerun `FULLHQ_STAGE=probe` on3x8 nodes;
then run the full pipeline. Do not enable training RESUME solely for this failed
preflight. No need to delete existing input caches or change the run namespace.

## Model and data

- Freeze `r7_t2_c192_v2/joint/checkpoint_best.pt`, including historical **legacy**
  temporal normalization. Do not resume a previous small DiT into this run.
- StreamVGGT -> frozen R7 -> 5 x 18 x 18 x 192 latent window. Independently encoded
  first image conditions generation of the four future latent slots. Decode the
  joint window with the frozen codec/decoder. Nine RGB frames, roughly one second;
  this is not yet a long-video baseline.
- Scratch DiT: width1536, depth24, heads24, **1,652,968,512 parameters** including
  text interfaces; previous reviewed model was178,383,936 (~9.27x smaller).
- UMT5 text conditioning, up to256 tokens, video-level `SceneDescription` reused
  across four windows. One T5 per node during preprocessing, unloaded before
  training. A Wan directory supplies T5/tokenizer only; no14B diffusion backbone.
- Full HQ CSV365362 rows, exclude all512 historical eval/test IDs -> **364850**
  candidate training videos / **1459400** deterministic windows before bad-data
  filtering. No street/drone restriction or old10K subset. Local full-CSV selection
  verified; actual successful remote windows must come from cache receipts.
- Image-only examples use explicit empty text when a caption is unavailable;
  missing caption receipts are retained, default maximum fraction5%.
- BF16 autocast, FP32 optimizer/EMA, activation checkpointing, batch1/device,
  accumulation8 -> globalbatch192. AdamW lr1e-4, warmup2000,100000 optimizer steps,
  uniform noise, weighted x0 loss (floor.05), EMA.999, text dropout.1. No aux head.
  Sampling uses Euler64, text CFG3 (image anchor kept in both branches).

## Memory choice

FP32 weights + gradients + Adam first/second moments + EMA total20bytes/parameter:
**30.79 GiB/device** before activations, communication, frozen decoder and buffers.
DDP uses gradient bucket views; optimizer disables foreach temporary tensor lists.
EMA evaluation backs up online weights on CPU (~6.16GiB/rank, ~49.3GiB/node), and
RGB decoding uses frame chunks of1. Host RAM must also cover other model/data
state; 49.3GiB is an incremental estimate, not total host memory.

Three nodes increase throughput/global batch; replicated DDP does **not** pool
their memory into one model. This motivates1.65B instead of jumping to3B+.
Before `all`, `probe`, or `train`, an actual24-rank probe runs three optimizer
updates with accumulated gradients, EMA swap, CFG and R7 decode. It records max
rank allocated/reserved memory in `memory_probe.json`; default allocated limit
52GiB leaves headroom on the intended64GB-class devices. Allocator figures do not
include all driver/HCCL memory. OOM or threshold failure stops; no automatic size
change. Real training repeats memory checks after step1, logs and evaluation.
The probe is a capacity test, not a quality test or a training-speed estimate.

Sep17 corrected cluster probe passed: max allocated40.6167GiB, reserved40.8789GiB
across24ranks. Full-data training peak still requires verification. Latent caching
uses all24NPUs with disjoint global-rank strides; only rank0 has a tqdm bar.
New logs additionally print every rank at durable shard commits, with numeric
DI_throughput in status JSON. Console convention is
`DI_throughput: <value> tokens/s/npu`: cache counts5x324 latent tokens/window;
training counts4x324 future tokens/window including accumulation; text counts
newly encoded text tokens on its one active NPU/node (resumed shards not counted).
These stages have different token units and their rates must not be compared as
equal workloads. Probe throughput is explicitly synthetic.

## Launch

Use the existing ModelArts bootstrap/environment and configure exactly3nodes,
8NPUs/node. Execute the same command on each node; the launcher derives rank and
master from the existing ModelArts helpers. Existing R7, StreamVGGT, reviewed eval
cache and a local Wan T5/tokenizer directory are prerequisites.

```bash
# Full pipeline: memory probe -> caption cache -> R7 cache -> training.
T5_CKPT_DIR=/path/to/existing/Wan-directory \
  bash scripts/scale/49_train_fullhq_ti2v.sh
```

If assets are already in the standard Wan locations, omit T5_CKPT_DIR. Do not use
a placeholder path literally. Optional separate memory-only job:

```bash
FULLHQ_STAGE=probe bash scripts/scale/49_train_fullhq_ti2v.sh
```

To inspect the first200 real updates without changing the100K schedule, set
`STOP_AFTER_STEPS=200` on the full launch. After a paused checkpoint is published:

```bash
FULLHQ_STAGE=train RESUME=1 STOP_AFTER_STEPS=0 \
  bash scripts/scale/49_train_fullhq_ti2v.sh
```

Preprocessing modes `prepare`, `text`, `cache` are also supported. Caption shards
and latent shards resume automatically under their immutable contracts. Resume
training explicitly with `RESUME=1` only after a training checkpoint exists.
Interrupted preprocessing with no training checkpoint: rerun `all`, RESUME=0.
Completed preprocessing with no training checkpoint: stage=train, RESUME=0.
Do not change width/AE/data/text contracts in the same namespace. Full training
resume still replays its data cursor; at this scale that can take substantial I/O.

## Bad data, storage and outputs

- Bad download/decode or invalid requested frame windows are skipped with
  videoID/window/index/error records; no substitute video or missing-frame fill.
  Systemic failures are guarded by default5% failure limit; model/OOM errors stay
  fatal. `MAX_DATA_FAILURE_RATE` is configurable and part of the cache contract.
- Video object availability is tested by actual caching. Metadata membership
  alone does not guarantee an OBS object exists. Cache manifests/stats contain
  successful windows only, and train normalization comes from that cache.
- Text cache uses32-video shards, immutable source/tokenizer identities, checksums
  and an8-shard per-rank LRU. No process loads the full caption embedding bank.
- Approximate latent payload upper estimate:0.91TB decimal without training RGB;
  max-length text embeddings ~0.77TB before overhead, usually less with shorter
  captions. These are persistent OBS inputs, not local raw downloads. Raw video
  local cache default200GB/node (`FULLHQ_VIDEO_CACHE_GB`); reserve staging space too.
- Persistent input owners: `cache_latents/r7_fullhq_t2v2_legacy_diag_v1/train` and
  `text_embeddings/fullhq_umt5_256_v1` below PERSISTENT_OBS_ROOT. These inputs have
  one durable owner, not two full copies. Training checkpoints and reports retain
  existing current-output + persistent-mirror double writes/read recovery.
- Run output: `output/scale/r7_fullhq_ti2v_1650m_v1/`; input/probe logs:
  `output/scale/fullhq_inputs_v1/nodeN/`. Publishers run during work and on exit.
- Rolling latest checkpoint every1000steps, best reconstruction-L1, paused and
  final retained; no giant step-numbered checkpoint archive. A full optimizer
  checkpoint is roughly24.6GiB before metadata. Samples/eval every2000steps,
  logs/DI throughput every10steps. Sudden kill can lose updates after latest save.
- Full caching may dominate startup time. Estimate duration from actual cache DI
  throughput and real train optimizer-step timing; do not reuse the small model ETA.

## Interpretation and acceptance

The existing32 selected street clips retain exact AE reference checks and22.3dB
minimum; previous measured mean22.41186dB. This is a regression evaluation, **not
full-HQ domain coverage**. Heldout512 IDs are excluded, but source-video-level
disjointness remains unverified. SceneDescription may not precisely describe each
one-second window. Best reconstruction-L1 checkpoint is not necessarily visually
best; inspect previews and text/image conditioning alongside metrics.

This scale test changes capacity, data coverage and text conditioning together.
It can establish whether this configuration generates useful video, but cannot
isolate which factor caused improvement/failure. Neither convergence nor low loss
guarantees visual quality. Preserve the earlier checkpoints for comparison.
