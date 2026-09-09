# Frozen decoder perturbation audit

Entry: `scripts/scale/34_diagnose_window_decoder.sh` on the existing six-node,
48-NPU ModelArts configuration. It reuses stage29 dual-source staging,
fresh-output guards, 60-second incremental dual publication and exit receipts.
It never trains, saves optimizer state, changes the AE or rebuilds caches.
Diagnostic retries require a fresh namespace; partial results are retained,
but diagnostics do not resume. Existing training full-state resume is unchanged.

Default source is `r7_window_t2v2_legacy_x0_shift3_v1`,
`checkpoint_best_reconstruction.pt`, with mandatory internal step=2500.
EMA completeness, AE sampled signature, statistics, manifests, flow contract,
legacy runtime and mean clean AE PSNR>=23.5 are checked before sampling.
Wrong step is an error, never silent fallback to latest. If best is unavailable,
select a verified periodic2500 artifact via DIAGNOSTIC_CKPT_NAME or both URL
overrides; keep EXPECTED_STEP=2500. Do not rename a6000 artifact to bypass this.

Default matrix:16 heldout clips x2 seeds x2 directions x6 amplitudes=384 rows.
Same Euler64/noise seeding as the training evaluation. The original native
anchor always reaches the decoder; only the future normalized latent changes.
For target y and sampled g, directions are g-y and a seeded random vector
with equal RMS separately per future slot. Decode y+alpha*direction with
alpha=0,.05,.1,.25,.5,1 after inverse normalization. Random directions are not
claimed to be manifold normals. No geometry/texture channel split is assumed.

Artifacts: config/checkpoint signature, perturbation_contract, ae_gate,
per-rank metrics/status and aggregate metrics/summary, all32 latent input/
direction records, and first4 clips' preview grids/MP4s for every amplitude.
Metrics include per-frame RGB L1 vs AE/RAW, latent/RGB MSE, finite RMS response
ratio (undefined at zero), frame differences, decode time and sampling DI
token-steps/s per active device. Sampling time repeats across perturbation
rows and must not be summed. DI excludes decoding/IO and is not job throughput.
No LPIPS, geometry metric or spectral Jacobian estimate is implemented here.

Run the shift1 reference as a separate ModelArts job with:

```bash
export DIAGNOSTIC_SOURCE=r7_window_t2v2_legacy_x0_diag_v2
export DIAGNOSTIC_CKPT_NAME=checkpoint_latest.pt
export EXPECTED_STEP=6000
export WINDOW_NAMESPACE=r7_window_t2v2_legacy_decoder_shift1_s6000_v1
bash scripts/scale/34_diagnose_window_decoder.sh
```

Use identical environment overrides on all six nodes. The default shift3 job
and reference job must not share output namespaces or run concurrently in one
allocation. Best2500 vs reference6000 is a quality-selected comparison, not
equal-budget training evidence. Compare response at matched actual RMS too,
since equal alpha need not mean equal error across checkpoints.

Interpretation: alpha0 should reproduce clean AE; generated alpha1 should
reproduce the source evaluation. Small-amplitude severe RGB degradation
motivates investigating decoder sensitivity, not immediately retraining AE.
Compare random and generated directions and inspect previews. Only then
consider separate decoder-only robustness tuning or generator auxiliary
supervision. Pixel ratios do not prove geometry, perceptual failure or causality.

Local validation:15 window CPU tests (including slot RMS, RNG preservation,
inverse normalization, fixed anchor and endpoints), Python compilation and
shell syntax checks. Real checkpoint/NPU/OBS matrix awaits cluster execution.
