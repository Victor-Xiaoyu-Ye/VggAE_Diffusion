#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/spatialvid.sh"

# Cache the held-out evaluation split once.

# ----------------------------- editable settings -----------------------------
AUTOENCODER_CKPT="${SCALE_GEOMETRY_AE_CKPT}"
SAMPLES_PER_TAR=128
BATCH_SIZE=1
NUM_WORKERS=4
CLIP_DURATION_SECONDS=1.0
# -----------------------------------------------------------------------------

NUM_GPUS=${NUM_GPUS:-8}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29603}

require_file "${AUTOENCODER_CKPT}" "scale geometry autoencoder checkpoint"
CSV_PATH="${SCALE_CSV_SHARD_ROOT}/eval/part-00000.csv"
require_file "${CSV_PATH}" "evaluation metadata shard"

"${TORCHRUN_BIN}" --nnodes="${NNODES}" --node_rank="${NODE_RANK}" \
  --nproc_per_node="${NUM_GPUS}" --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${PROJECT}/cache_compact_latents.py" \
  --csv "${CSV_PATH}" \
  --video_root "${SPATIALVID_VIDEO_ROOT}" \
  --encoder_ckpt "${STREAMVGGT_CKPT}" \
  --autoencoder_ckpt "${AUTOENCODER_CKPT}" \
  --output_dir "${SCALE_EVAL_CACHE_DIR}" \
  --partition_id 0 --num_partitions 1 \
  --samples_per_tar "${SAMPLES_PER_TAR}" \
  --latent_dim 512 --latent_grid 18 \
  --levels 4 11 17 23 \
  --seq_len 8 --target_size 518 --max_frame_span 32 \
  --clip_duration_seconds "${CLIP_DURATION_SECONDS}" \
  --batch_size "${BATCH_SIZE}" --num_workers "${NUM_WORKERS}"
