# Stage48 partial audit — 2026-09-16

## Platform termination evidence (later user-provided console tail)

The console tail resolves the termination mechanism. Line1560 shows the fourth
case advancing to38/40 (later than the last OBS snapshot), with the coordinator
still reporting leader running at7740seconds. At12:50:01 China time, line1571
reports `controller is being shutdown externally...`; line1596 explicitly
records `signal SIGTERM to user process group (pgid: 333)`. Lines1598–1599
record exit -1. This was platform-initiated termination while generation was
still progressing, not successful completion or evidence of a denoiser hang.

The15seconds in line1597 is the process termination timeout, not the training
job wall-time limit. The log says a restart will follow; it does not prove a
restart occurred. The triggering event (scheduler/platform policy, user action,
recovery event or another cause) is not identified by this tail. The trailing
Python threading shutdown traceback follows termination and does not establish
the initiating fault. Last OBS publication precedes shutdown by roughly27s,
consistent with the60s incremental publication interval; stale files do not
establish a pre-existing publisher failure.

User confirms the cluster has stopped. Preserve the original contract/code and
resume the same4clip namespace with RESUME=1: three committed native cases can
be reused; the interrupted fourth restarts, leaving five native generations
plus replay. At observed speed this is approximately2.6hours generation alone.
Check platform termination/restart events and execution-time policy before
resubmission. Do not alter model/solver/clip count to address this termination.

## Earlier OBS-only inspection

Checked the owner OBS prefix `output/scale/r7_native_i2v_roundtrip_smoke_v1/`
twice, most recently about14:50 China time. The intended4clips x2seeds run
has only3 committed native cases. The alternative default prefix
`r7_native_i2v_roundtrip_v1/` is empty. This is not a completed three-arm audit.

## Completion evidence

- Contract:4 original heldout clips, seeds101/211, native81frames16fps,
  UniPC40, shift3, CFG5, fixed generic prompt, no future information.
- Completed: clip000_seed101, clip000_seed211, clip001_seed101.
- Last status:2026-09-16 12:49:19 China time; clip001_seed211 step37/40.
- Last publication status: synced,12:49:34, both current output and owner OBS.
  This is an incremental publication receipt, not final completion evidence.
- No `launcher_exit.json`, `native_timing.json`, comparison receipts or
  `summary.json` in the observed owner prefix. No ending traceback in the
  downloaded log. Cannot distinguish job termination from stale publication
  or an alternative output location without platform logs.
- Early R7 local FileNotFoundError was followed by successful OBS fallback
  and successful preparation/generation; it did not stop this run.

Downloaded small logs,3MP4s and2inputPNGs to
`D:/workspace/VggAE_DataAudit/native_i2v_review_20260916/`. All downloaded native
MP4/input files match the corresponding receipt SHA256. No checkpoints,
raw dataset or large latent artifacts downloaded. A six-frame contact sheet
samples native frames0,8,16,32,56,80 from each MP4.

## What the available videos show

Native outputs have recognizable, relatively sharp buildings, road surfaces
and lighting across sampled frames. Clip000 has little camera movement early;
later a car/person moves through seed101, while seed211 has a different
continuation. Clip001 shows substantial foreground/camera change, with visible
late foreground/object deformation. These are not perfect videos, and three
cases from only two scenes do not establish general quality or motion quality.
Review is based on decoded MP4 sampled frames, not a perceptual benchmark.

This supplies a first native pretrained generation reference on actual selected
street anchors. It does not validate R7 generation: R7 roundtrip and current
diffusion comparison have not appeared. It cannot identify the main failure
as encoder, decoder, data scale, denoiser capacity or optimization. In particular,
native pretrained14B versus scratch178M is not a controlled latent comparison.

## Runtime evidence

| Case | Generation minutes | Reported peak allocation GiB |
|---|---:|---:|
| clip000_seed101 |31.30|38.83|
| clip000_seed211 |30.80|38.83|
| clip001_seed101 |30.84|38.83|

All completed cases record80denoiser forwards. Reported peak is the process
cumulative allocation peak, not total device residency or a per-case reset.
Generation throughput is approximately0.0436frames/s, including conditioning
and VAE. Eight cases imply approximately4.1hours of generation, excluding
initial staging/loading, export and R7 replay. This is an extrapolation, not
a completion-time guarantee. Only leader NPU0 performs this computation.

The small FP16/BF16 attention gate passed on NPU; maximum observed absolute
error0.005077, maximum RMS0.000696. Three videos demonstrate native inference
can run with this environment/shape; replay and all eight cases remain unverified.

## Next action

Obtain actual job output location or console tail before assuming success or
failure. If the same job really terminated, keep the original code/config and
resume with `RESUME=1 NATIVE_CLIPS=4` and the same smoke namespace. Verify native
weight path as before. Completed cases are checked/reused; incomplete fourth
case restarts. Do not start a duplicate while the job may still be active.
Finish frozen R7 replay before proposing another training or scaling experiment.
