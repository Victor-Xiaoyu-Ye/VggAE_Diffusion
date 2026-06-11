#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/spatialvid.sh"

# ----------------------------- editable settings -----------------------------
AUTOENCODER_CKPT="${SCALE_GEOMETRY_AE_CKPT}"
OUTPUT_DIR="${SCALE_ROOT}/i0_decoder"
RESUME=""
REPRESENTATION_MAX_VIDEOS=1000000

EPOCHS=3
BATCH_SIZE=1
ACCUM_STEPS=8
LEARNING_RATE=1e-4
WEIGHT_DECAY=1e-2
WARMUP_STEPS=5000
NUM_WORKERS=8
# -----------------------------------------------------------------------------

NUM_NPUS=${NUM_NPUS:-8}
ASCEND_DEVICE_IDS=${ASCEND_DEVICE_IDS:-0,1,2,3,4,5,6,7}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29601}

ensure_spatialvid_scale_splits
require_file "${AUTOENCODER_CKPT}" "scale geometry autoencoder checkpoint"

EXTRA_ARGS=()
if [[ -n "${RESUME}" ]]; then
  EXTRA_ARGS+=(--resume "${RESUME}")
fi

ASCEND_RT_VISIBLE_DEVICES="${ASCEND_DEVICE_IDS}" torchrun \
  --nnodes="${NNODES}" --node_rank="${NODE_RANK}" \
  --nproc_per_node="${NUM_NPUS}" --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${PROJECT}/train_i0_autoencoder.py" \
  --csv "${SPATIALVID_FULL_TRAIN_CSV}" \
  --video_root "${SPATIALVID_VIDEO_ROOT}" \
  --eval_csv "${SPATIALVID_EVAL_CSV}" \
  --eval_video_root "${SPATIALVID_VIDEO_ROOT}" \
  --encoder_ckpt "${STREAMVGGT_CKPT}" \
  --autoencoder_ckpt "${AUTOENCODER_CKPT}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_videos "${REPRESENTATION_MAX_VIDEOS}" \
  --latent_dim 512 --latent_grid 18 \
  --decoder_base_dim 384 --decoder_num_resblocks 2 \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --accum_steps "${ACCUM_STEPS}" \
  --lr "${LEARNING_RATE}" \
  --wd "${WEIGHT_DECAY}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --max_grad_norm 1.0 --lambda_lpips 1.0 \
  --dtype fp16 --seq_len 8 --target_size 518 \
  --max_frame_span 32 --num_workers "${NUM_WORKERS}" \
  --log_every 50 --save_every 1 --eval_every 1 \
  "${EXTRA_ARGS[@]}"
