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
require_output_url
ensure_spatialvid_scale_splits
ensure_local_checkpoint \
  "${AUTOENCODER_CKPT}" "${SCALE_GEOMETRY_AE_CKPT_URL}" \
  "scale geometry autoencoder checkpoint"

LOG_DIR="${SCALE_ROOT}/cache_generation/train"
REMOTE_LOG_DIR="${SCALE_REMOTE_ROOT}/cache_generation/train"
start_output_sync "${LOG_DIR}" "${REMOTE_LOG_DIR}"
trap 'stop_output_sync "${LOG_DIR}" "${REMOTE_LOG_DIR}"' EXIT

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
  --resume_cache \
  --samples_per_tar "${SAMPLES_PER_TAR}" \
  --clips_per_video "${SCALE_CLIPS_PER_VIDEO}" \
  --latent_dim "${SCALE_LATENT_DIM}" \
  --latent_grid "${SCALE_LATENT_GRID}" \
  --levels 4 11 17 23 \
  --disable_temporal_mixer \
  --seq_len 8 --target_size 518 --max_frame_span 32 \
  --clip_duration_seconds "${CLIP_DURATION_SECONDS}" \
  --batch_size "${BATCH_SIZE}" --num_workers "${NUM_WORKERS}" \
  --dtype fp16

if [[ "${NODE_RANK}" -eq 0 ]]; then
  echo "Latent shards were uploaded incrementally to:"
  echo "  ${SCALE_TRAIN_CACHE_DIR}"
  echo "Local files under ${MOX_CACHE_WRITER_DIR} are temporary staging only."
fi
