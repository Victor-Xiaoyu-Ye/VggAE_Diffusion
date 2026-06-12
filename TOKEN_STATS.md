# Compact Latent Statistics

The diffusion distribution is not the raw StreamVGGT token distribution. For
the active I0 residual objective it is:

```text
condition = tokenizer(StreamVGGT(I0))
target[t] = tokenizer(StreamVGGT(clip))[t + 1] - condition
```

Statistics must be computed after freezing the exact StreamVGGT and tokenizer
checkpoints, using only the training split. Evaluation uses the saved training
statistics and must never recompute them.

SpatialVID mixes roughly 24, 25, 30, 50, and 60 FPS videos. The active
workflow therefore samples 8 frames over a fixed 1.0 second interval. A fixed
source-frame span would make the same future-frame index represent different
physical horizons and invalidate frame-aware normalization.

## Current Contract

- Condition mean/std shape: `[1, latent_dim]`.
- Future residual mean/std shape: `[future_frames, latent_dim]`.
- Broadcast over batch and spatial tokens.
- Persist the tensors in every cache and diffusion checkpoint.
- Refuse resume when the representation signature or statistics differ.
- Clamp std only for numerical safety; do not clip latent values by default.

With correctly standardized data, `train/target_std` should be close to 1.
For OT-CFM with independent unit-variance data and Gaussian noise, a
zero-initialized velocity head starts near MSE 2. A materially different
initial loss is a useful warning that normalization, target layout, or
checkpoint loading is inconsistent.

The online 10K trainer and cached scale trainer now share this contract. Legacy
`[latent_dim]` cached statistics remain readable, but new caches are
frame-aware.

For online DDP training, `normalization_batches` is counted per rank and the
moments are all-reduced. With 4 ranks, batch size 2, and 64 batches, the
estimate covers up to 512 clips. Cached training does not estimate from a
subset: it merges exact moments from every successfully cached training clip.
Use the convergence table from `diagnose_compact_latent_stats.py` to increase
the online budget if the frame/channel std is not stable.

Each cache partition persists FP64 `sum`, `sum_sq`, and `count`. The merge step
combines those raw moments before producing FP32 mean/std for training. Legacy
partitions containing only mean/std remain readable, but a merge refuses to mix
raw-moment and legacy partitions.

The representation identity is path-independent. Cache partitions store file
size plus a SHA256 over sampled blocks from the beginning, middle, and end of
the encoder and tokenizer checkpoints. This avoids false mismatches when
moxing copies the same checkpoint to different local paths.

## Why A Scalar Scale Is Insufficient

On 16 held-out SpatialVID clips with the current 512x18x18 tokenizer, before
the fixed-duration sampling correction:

- Residual channel std p99/p01 ratio: about `1.57x`.
- Residual std grows from about `0.115` at future frame 1 to `0.192` at frame 7.
- Median residual excess kurtosis: about `2.3`, so the distribution is
  substantially heavier-tailed than Gaussian.
- Channel-correlation effective rank: about `200 / 512`.

A single global scale hides channel and temporal heteroscedasticity. Per-channel
statistics alone still hide the strong increase in uncertainty with horizon.
These measurements establish the normalization shape, but the exact values
must be recomputed with the active 1.0 second sampling contract before training.

## Whitening Decision

Full PCA/ZCA whitening is not the default yet. It may improve optimization
because the correlation effective rank is low, but it also rotates the channel
basis consumed by the decoder. A whitening experiment must:

1. Save the exact mean, whitening matrix, and inverse matrix.
2. Apply whitening only at the diffusion boundary.
3. Invert it before decoding.
4. Compare reconstruction after round-trip, tiny-set overfit speed, and sampled
   latent spectra against frame/channel z-score normalization.

Do not normalize independently by spatial position. The measured spatial RMS
variation is moderate and contains useful scene-layout priors.

## Temporal Mixer Finding

The current symmetric tokenizer `TemporalMixer` changes I0 depending on whether
it is encoded alone or inside a clip:

- Compact I0 relative L2 drift: `0.1387`.
- Drift / future residual RMS: `0.2334`.
- With the mixer bypassed: `0.000169` and `0.000376`.

The existing checkpoint must retain the mixer for compatibility. The next
geometry autoencoder should remove it or replace it with a strictly causal
module, then recompute all latent statistics and caches.

Run:

```bash
bash scripts/10k/diagnose_latent_contract.sh
bash scripts/10k/diagnose_compact_latent_stats.sh
```
