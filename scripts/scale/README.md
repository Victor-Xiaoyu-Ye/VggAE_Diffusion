# Large-scale training path

The large-scale path freezes the representation before the 10M generator run.
It does not decode MP4 or run StreamVGGT inside the diffusion training loop.

Edit paths once in `scripts/spatialvid_config.sh`. Stage 02 automatically
creates deterministic full-train and held-out metadata from the original
SpatialVID CSV. No manually prepared evaluation CSV is required.

1. `00_train_geometry_autoencoder.sh`
   - Train tokenizer and decoder on a diverse representation subset.
   - Depth supervision is required so compression is explicitly geometry-aware.
2. `01_train_i0_decoder.sh`
   - Train the appearance-conditioned RGB decoder with the tokenizer frozen.
3. `02_shard_metadata.sh`
   - Build/reuse full-train and evaluation CSV shards.
4. `03_cache_latents.sh`
   - Run as an array job for the full training split. The scheduler only needs
     to provide `INDEX_SHARD_ID` and distributed launch variables.
5. `03_cache_eval_latents.sh`
   - Run once to cache the automatically selected held-out split.
6. `04_merge_latent_cache.sh`
   - Merge tar manifests and exact per-channel normalization statistics.
   - Training and evaluation caches are merged separately.
7. `05_train_compact_dit.sh`
   - Train by optimizer step from streaming tar shards.
   - Set `EVAL_CACHE_DIR` to a held-out cache for fixed validation metrics.
   - Every checkpoint writes TensorBoard/JSONL metrics and a target/generated
     latent preview.
   - For RGB previews, set `EVAL_I0_PATH` and `I0_DECODER_CKPT`. The image must
     correspond to the first sample in `EVAL_CACHE_DIR/manifest.txt`.
8. `06_sample_compact_dit.sh`
   - Generate seven future frames from one observed RGB frame.

At 512 channels, an eight-frame fp16 cache is about 2.65 MB/video, or about
26.5 TB for 10M videos before filesystem replication. Do not cache the four raw
StreamVGGT levels; that is roughly 1.8 PB for 10M videos.

Distributed launch uses HCCL. Set `NUM_NPUS`, `ASCEND_DEVICE_IDS`, `NNODES`,
`NODE_RANK`, `MASTER_ADDR`, and `MASTER_PORT` in the cluster launcher.

The representation checkpoint is a data contract. Do not continue changing the
tokenizer after latent caching starts. If the tokenizer changes, rebuild the
cache and its statistics.

All stage 00, 01, and 05 training checkpoints are resumable. Use
`RESUME=/path/to/checkpoint.pt`. Stage 05 restores model, FP32 EMA, optimizer,
schedule, global step, normalization contract, and RNG state. The streaming
shard iterator restarts rather than resuming at an exact tar byte offset, so
sample order after recovery is not bit-identical.

Wan initialization is deliberately not the default scale script. The current
compact adapter bypasses Wan's native VAE patch interface, and the legacy CLIP
text path does not match Wan's pretrained UMT5 context. Establish the compact
DiT baseline first, then compare Wan initialization on the same cached latents.
