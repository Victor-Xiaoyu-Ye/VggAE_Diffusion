# Frozen step6000 diagnostics

Entry: `bash scripts/scale/32_diagnose_window_diffusion.sh` on all six nodes,
using the existing ModelArts staging/environment and 48-NPU configuration.
No cache rebuilding or optimizer updates. Stage29 is reused only for its
distributed/storage lifecycle; WINDOW_DIAGNOSTIC_ONLY dispatches the evaluator
and exits before any training/text/resume setup.

Default source: r7_window_t2v2_legacy_x0_diag_v2/checkpoint_latest.pt.
Expected step is exactly6000, weights are EMA. Current OUTPUT_URL read falls
back to the persistent owner OBS root. AE uses the existing double-read helper.
Cache is the immutable r7_window_t2v2_legacy_diag_v2 persistent cache. The full
checkpoint is read, but optimizer/RNG/cursors are not resumed or overwritten.

Default output: r7_window_t2v2_legacy_x0_diagnostics_v1. Fresh-output guard checks
local/current/mirror destinations. For a second attempt set a NEW identical
WINDOW_NAMESPACE on every node. Diagnostic interruption retains partial metrics
and samples; this evaluator has no partial-result resume. Never set the output
to a training namespace. Source overrides: DIAGNOSTIC_SOURCE,
DIAGNOSTIC_CKPT_URL, DIAGNOSTIC_CKPT_MIRROR_URL. EXPECTED_STEP must match the
chosen source (default6000). No automatic fallback to a different step/AE.

The matrix is two splits x16 clips x2 seeds x5 arms =320 samples:
Euler64, Euler128, Heun64 (127 model evaluations), zero normalized anchor,
cyclic different-video anchor. Conditions change only at DiT input; the decoder
keeps the original anchor. This isolates generator sensitivity, not end-to-end
unconditional generation. Same clip/seed uses exactly the same initial noise
for every arm. Both seeds are42/43, matching training previews; selection uses
the saved seed and unshuffled-within-shard dataset, matching heldout evaluation.
Training samples are a deterministic subset, not the full training distribution.

Every rank checks original checkpoint manifest lists, statistics digest, flow
contract, EMA completeness, AE signature/runtime. Heldout RAW replay must pass
23.5dB BEFORE sampling. Train cache has no RAW: compare train/heldout using
latent MSE and RGB L1 vs AE; RAW metrics apply only to heldout. The exact IDs,
donor IDs, checkpoint signature, sampler NFE and contract are recorded.

Outputs: ae_gate.json, config.json, per-rank metrics/status, aggregate metrics.jsonl,
summary.json, four preview clips per split/arm/seed with grids/MP4/latents. The
existing incremental publisher mirrors every60s and on exit; worker files are
under workers/nodeN. Check launcher_exit.json, publication_status.json and all
diagnostic statuses; incomplete runs must not be interpreted as a complete matrix.
DI_throughput is future token integration-steps/s per active device, excluding
decode/I/O; model_token_evals_per_second accounts for Heun's extra evaluations.
These are sampling metrics, not the training DI throughput. With16 clips some
ranks are idle; do not extrapolate to48-card utilization or full training speed.

Interpretation: sampler improvement supports integration sensitivity; train-good /
heldout-bad suggests a generalization gap; both-bad suggests insufficient fitting,
objective/model limitations or representation difficulty. Anchor output changes
measure sensitivity, not necessarily useful conditioning. Zero/shuffle are OOD
interventions. Pixel error does not establish perceptual quality, physical motion
or geometry. Inspect matching previews; do not automatically promote a new recipe.

Validation:10 CPU tests (paired-noise/anchor/sampler oracle and existing window
flow/resume/storage contracts), Python compilation and both shell syntax checks.
Actual step6000 checkpoint sampling, NPU and OBS behavior await this cluster run.
