# Version Status

## Active

| Workflow | Entry point | Status |
|---|---|---|
| 10K compact autoencoder | `scripts/10k/train_geometry_autoencoder.sh` | Active |
| 10K I0 decoder | `scripts/10k/train_i0_autoencoder.sh` | Active |
| 10K online compact diffusion | `scripts/10k/train_compact_diffusion.sh` | Active baseline |
| 10K overfit gates | `scripts/10k/overfit_*.sh` | Required validation |
| Scale representation | `scripts/scale/00_train_geometry_autoencoder.sh` | Active |
| Scale I0 decoder | `scripts/scale/01_train_i0_decoder.sh` | Active |
| Scale latent cache | `scripts/scale/02_*` through `04_*` | Active |
| Scale cached Compact DiT | `scripts/scale/05_train_compact_dit.sh` | Primary generator |
| Scale sampling | `scripts/scale/06_sample_compact_dit.sh` | Active |

All active shell arguments match their Python entry points. Python compilation
and shell syntax checks pass. Ascend 910B forward and HCCL distributed runtime
validation remain required on the training cluster.

The active scale path supports 6 ModelArts workers x 8 NPUs, derives the HCCL
topology from `VC_WORKER_*`, stages SpatialVID MP4 files from OBS on demand,
streams latent tar shards through a bounded local cache, and mirrors rank-0
outputs to `$OUTPUT_URL`.

Active trainers use atomic resumable checkpoints. Periodic and final files
contain train weights, FP32 EMA, optimizer, scheduler, scaler where applicable,
global step, RNG state, arguments, and latent normalization when required.
Checkpoint saves also trigger visual previews and write metrics to both
TensorBoard and `metrics.jsonl`. Cached scale training always saves a latent
preview and can additionally decode RGB previews from a fixed aligned I0.

New I0 residual checkpoints generate seven future frames only. The observed
first frame is prepended before decoding and is never a diffusion target. New
Compact DiT runs also store `time_scale=1000`; checkpoints without that field
retain the historical `time_scale=1` behavior during sampling. Do not resume a
historical eight-target run into the new seven-target configuration.

## Experimental

`models/wan_compact_adapter.py` is retained as an adapter prototype. Its
timestep bug is fixed, and it has native 4096-dim UMT5 and I0 conditioning
hooks. The current `train_wan_compact_diffusion.py` harness does not wire those
hooks into the active seven-frame residual/cache contract: it still uses the
legacy CLIP path, online encoding, and eight full-frame targets. Treat that
trainer as legacy, not as a valid Wan baseline. The adapter also bypasses Wan's
native VAE patch input/output interface.

## Legacy

Historical DPT decoder, raw-token diffusion, Wan training harnesses, and non-I0
scripts are under `scripts/legacy/`. They compile and their shell arguments are
valid, but their model contracts are superseded and were not upgraded to the
current I0 residual/cache contract. They should only be used to inspect or
reproduce old checkpoints.

## Removed Debug Artifacts

- Hard-coded `test_eval_sample.py`
- Single-video raw-token `overfit_single.py`
- `verify_diffusion.py` experiment harness and captured output logs
- Committed reconstruction images under `outputs/`
- Python bytecode caches

The I0 and diffusion overfit scripts were retained because they are validation
gates, not disposable debug artifacts.
