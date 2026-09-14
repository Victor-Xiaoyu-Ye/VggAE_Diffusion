# Domain AE replay diagnosis

2026-09-14: single/mixed uniform v2 both stopped before any diffusion update.
The same 32 street evaluation IDs scored 22.41186/22.41298 dB against the
inherited 23.5 dB gate. Both used legacy norm and the same weight signatures.
The old 24.48933 dB result used different 16 clips; means are not a regression
test across datasets. No training threshold has been changed.

## Launch

Use the existing ModelArts environment/dependency initialization, including
the staged StreamVGGT checkpoint. One NPU on node0 is sufficient. The script
also accepts the existing six-node launch: other nodes exit without work.

```bash
bash /cache/yexiaoyu/VggAE_Diffusion/scripts/scale/45_audit_domain_ae.sh
```

Three sequential reports: single32, mixed32, historical16. Each selects the
exact video IDs from its previous ae_baseline.json, fetched current-output
first then owner mirror. Each checks AE signatures, re-encodes saved RAW with
the frozen StreamVGGT/R7, and compares fresh vs cached future and anchor
latents. Legacy/framewise both run as a semantic control; each reports fresh
and cached decode PSNR, per-frame PSNR and fresh/cache RGB L1. Missing IDs,
duplicate selected IDs, changed signatures, missing RAW and runtime failures
are errors. A missing cohort does not prevent attempting the others.

The report-only flag applies only to this diagnosis. Low quality remains
quality_gate_passed=false, but no longer aborts before other cohorts finish.
status=completed means measurements completed, NOT permission to train;
training_authorized is always false. Stage31 defaults retain their old gate.

Outputs under both current output and persistent owner output:

```
scale/window_ae_audits/r7_domain_ae_replay_single_v1/attempt<TIMESTAMP>/
scale/window_ae_audits/r7_domain_ae_replay_mixed_v1/attempt<TIMESTAMP>/
scale/window_ae_audits/r7_domain_ae_replay_historical_v1/attempt<TIMESTAMP>/
```

Each includes audit.json after every clip (before preview writing), exact
source_ae_baseline.json, logs/audit.log, four clip previews, launcher_exit.json
and publication_status.json. Incremental publication and final dual write use
stage31/window_run_io; final publication failure is nonzero. R7 checkpoint
uses existing current/mirror fallback. StreamVGGT is the existing staged input;
latent shards stream from their authoritative persistent cache, which has no
second replica. No dataset video download, cache rebuild or optimizer is used.

DI_throughput is source frames/second including encoder/decoder diagnosis and
earlier preview/I/O time, excluding model loading; it is not diffusion training
throughput. This runs 80 clips with several encode/decode passes; no runtime
estimate is claimed before NPU execution.

## Interpretation

First compare legacy/cache_psnr to previous_cache_psnr on exactly the same
IDs. Then compare fresh legacy reconstruction, per-frame errors, future and
anchor latent differences against the historical control. Saved RAW is uint8
and cached latents fp16, so fresh/cache latent equality is not expected bitwise;
RAW quantization and accelerator numerics must be considered before calling
small differences corruption. No uncalibrated numerical replay threshold is
used to approve training automatically.

If cached replay matches previous output and fresh replay agrees to the
historical control's numerical scale, the new-domain AE ceiling is supported.
If either replay disagrees materially, investigate that path before training.
Only after review should a domain-specific quality policy be adopted. Failed
v2 runs have no training checkpoint: preserve caches and use a fresh training
namespace when restarting; RESUME=1 is not appropriate.

Local static/CPU checks cannot validate real NPU replay, OBS streaming, quality
or the cluster's staged dependency paths. Those are the purpose of this run.
