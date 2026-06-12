#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/spatialvid.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"

AUTOENCODER_CKPT="${SCALE_GEOMETRY_AE_CKPT}"
SAMPLES_PER_TAR=128
BATCH_SIZE=1
NUM_WORKERS=2
CLIP_DURATION_SECONDS=1.0
MASTER_PORT=29603

configure_modelarts_distributed
require_scale_cluster
ensure_spatialvid_scale_splits
ensure_local_checkpoint \
  "${AUTOENCODER_CKPT}" "${SCALE_GEOMETRY_AE_CKPT_URL}" \
  "scale geometry autoencoder checkpoint"

run_torchrun "${PROJECT}/cache_compact_latents.py" \
  --csv "${SPATIALVID_EVAL_CSV}" \
  --video_root "${SPATIALVID_VIDEO_ROOT}" \
  --encoder_ckpt "${STREAMVGGT_CKPT}" \
  --autoencoder_ckpt "${AUTOENCODER_CKPT}" \
  --output_dir "${SCALE_EVAL_CACHE_DIR}" \
  --partition_id 0 --num_partitions 1 \
  --store_i0_rgb \
  --samples_per_tar "${SAMPLES_PER_TAR}" \
  --latent_dim "${SCALE_LATENT_DIM}" \
  --latent_grid "${SCALE_LATENT_GRID}" \
  --levels 4 11 17 23 \
  --disable_temporal_mixer \
  --seq_len 8 --target_size 518 --max_frame_span 32 \
  --clip_duration_seconds "${CLIP_DURATION_SECONDS}" \
  --batch_size "${BATCH_SIZE}" --num_workers "${NUM_WORKERS}" \
  --dtype fp16
