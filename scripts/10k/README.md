# 10K experiment scripts

These scripts preserve the current online 10K workflow. They intentionally
keep video decoding, StreamVGGT encoding, and tokenization inside the training
loop so existing experiments remain reproducible.

- `train_geometry_autoencoder.sh`: train the compact tokenizer and RGB decoder.
- `train_i0_autoencoder.sh`: current I0-conditioned reconstruction experiment.
- `train_compact_diffusion.sh`: current online I0 residual diffusion experiment.
- `overfit_i0_autoencoder.sh`: small-set reconstruction sanity check.
- `overfit_compact_diffusion.sh`: small-set diffusion sanity check.
- `diagnose_latent_contract.sh`: compare independently encoded I0 with the
  first-frame latent produced inside a complete clip.
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

Resume with `RESUME=/path/to/checkpoint.pt`. Resume occurs at an epoch boundary;
the optimizer, learning-rate schedule, EMA, and global step are restored.

Do not use these scripts for the 10M run. Use `scripts/scale/` instead.
