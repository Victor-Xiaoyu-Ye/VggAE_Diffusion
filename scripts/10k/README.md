# 10K experiment scripts

These scripts preserve the current online 10K workflow. They intentionally
keep video decoding, StreamVGGT encoding, and tokenization inside the training
loop so existing experiments remain reproducible.

Edit dataset/checkpoint/output roots once in `scripts/spatialvid_config.sh`.
Every script then calls `prepare_spatialvid_splits.py`, which deterministically
selects existing SpatialVID videos into non-overlapping `train_10k.csv`,
`eval.csv`, and `overfit.csv`. Existing matching splits are reused.

- `train_geometry_autoencoder.sh`: train the compact tokenizer and RGB decoder.
- `train_i0_autoencoder.sh`: current I0-conditioned reconstruction experiment.
- `train_compact_diffusion.sh`: current online I0 residual diffusion experiment.
- `overfit_i0_autoencoder.sh`: small-set reconstruction sanity check.
- `overfit_compact_diffusion.sh`: small-set diffusion sanity check.
- `diagnose_latent_contract.sh`: compare independently encoded I0 with the
  first-frame latent produced inside a complete clip.
- `diagnose_compact_latent_stats.sh`: audit frame/channel moments, heavy tails,
  spatial energy, and channel-correlation effective rank.
- `run_validation_suite.sh`: run latent diagnostics and both overfit gates.
- `collect_results.sh`: print and save a combined metrics/checkpoint/preview
  report under `RUN_ROOT/reports`.
- `inference_autoencoder.sh`: held-out compact autoencoder reconstruction.
- `sample_compact_i0.sh`: generate a video from one reference frame.

The current residual diffusion contract is one observed I0 latent plus seven
generated future residuals. Historical online checkpoints that predicted eight
targets remain sampleable, but they are not resume-compatible with this layout.

All active training scripts write:

- resumable periodic and final checkpoints containing model, EMA, optimizer,
  scheduler, scaler where applicable, global step, RNG state, and arguments;
- TensorBoard events under `OUTPUT_DIR/tb/`;
- durable scalar records under `OUTPUT_DIR/metrics.jsonl`;
- reconstruction or generation previews under `OUTPUT_DIR/samples/` whenever
  a checkpoint is saved.

Set `RESUME` near the top of the relevant script. Resume occurs at an epoch
boundary; the optimizer, learning-rate schedule, EMA, and global step are
restored.

Paths and experiment hyperparameters are assigned near the top of each shell
script. `NUM_GPUS`, `GPU_IDS`, and `MASTER_PORT` remain environment-driven so
cluster launchers can control distributed execution.

Do not use these scripts for the 10M run. Use `scripts/scale/` instead.

The normalization contract and current measurements are documented in
`TOKEN_STATS.md`.

Active SpatialVID scripts sample 8 frames across a fixed 1.0 second interval,
because the local dataset mixes 24-60 FPS videos. Keep this setting identical
for representation training, latent caching, diagnostics, and diffusion.
