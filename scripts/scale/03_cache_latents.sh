#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/spatialvid.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"

# ----------------------------- editable settings -----------------------------
AUTOENCODER_CKPT="${SCALE_GEOMETRY_AE_CKPT}"
CACHE_PARTITION_ID=0
CACHE_NUM_PARTITIONS=1
SAMPLES_PER_TAR=512
BATCH_SIZE=1
NUM_WORKERS=4
CLIP_DURATION_SECONDS=1.0
MASTER_PORT=29602
# -----------------------------------------------------------------------------

configure_modelarts_distributed
require_scale_cluster
ensure_spatialvid_scale_splits
require_file "${AUTOENCODER_CKPT}" "scale geometry autoencoder checkpoint"

run_torchrun "${PROJECT}/cache_compact_latents.py" \
  --csv "${SPATIALVID_FULL_TRAIN_CSV}" \
  --video_root "${SPATIALVID_VIDEO_ROOT}" \
  --encoder_ckpt "${STREAMVGGT_CKPT}" \
  --autoencoder_ckpt "${AUTOENCODER_CKPT}" \
  --output_dir "${SCALE_TRAIN_CACHE_DIR}" \
  --index_shard_id "${CACHE_PARTITION_ID}" \
  --index_num_shards "${CACHE_NUM_PARTITIONS}" \
  --partition_id "${CACHE_PARTITION_ID}" \
  --num_partitions "${CACHE_NUM_PARTITIONS}" \
  --samples_per_tar "${SAMPLES_PER_TAR}" \
  --latent_dim 512 --latent_grid 18 \
  --levels 4 11 17 23 \
  --disable_temporal_mixer \
  --seq_len 8 --target_size 518 --max_frame_span 32 \
  --clip_duration_seconds "${CLIP_DURATION_SECONDS}" \
  --batch_size "${BATCH_SIZE}" --num_workers "${NUM_WORKERS}" \
  --dtype fp16
