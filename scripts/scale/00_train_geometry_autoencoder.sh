#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/spatialvid.sh"

# ----------------------------- editable settings -----------------------------
OUTPUT_DIR="${SCALE_ROOT}/geometry_autoencoder"
RESUME=""
REPRESENTATION_MAX_VIDEOS=1000000

EPOCHS=3
BATCH_SIZE=1
ACCUM_STEPS=8
LEARNING_RATE=1e-4
WEIGHT_DECAY=1e-2
WARMUP_STEPS=5000
EMA_DECAY=0.9999
NUM_WORKERS=8
# -----------------------------------------------------------------------------

NUM_NPUS=${NUM_NPUS:-8}
ASCEND_DEVICE_IDS=${ASCEND_DEVICE_IDS:-0,1,2,3,4,5,6,7}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29600}

ensure_spatialvid_scale_splits
require_dir "${SPATIALVID_DEPTH_ROOT}" "SpatialVID depth root"

EXTRA_ARGS=()
if [[ -n "${RESUME}" ]]; then
  EXTRA_ARGS+=(--resume "${RESUME}")
fi

ASCEND_RT_VISIBLE_DEVICES="${ASCEND_DEVICE_IDS}" torchrun \
  --nnodes="${NNODES}" --node_rank="${NODE_RANK}" \
  --nproc_per_node="${NUM_NPUS}" --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${PROJECT}/train_autoencoder.py" \
  --csv "${SPATIALVID_FULL_TRAIN_CSV}" \
  --video_root "${SPATIALVID_VIDEO_ROOT}" \
  --eval_csv "${SPATIALVID_EVAL_CSV}" \
  --eval_video_root "${SPATIALVID_VIDEO_ROOT}" \
  --depth_root "${SPATIALVID_DEPTH_ROOT}" \
  --encoder_ckpt "${STREAMVGGT_CKPT}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_videos "${REPRESENTATION_MAX_VIDEOS}" \
  --latent_dim 512 --latent_grid 18 \
  --levels 4 11 17 23 \
  --decoder_base_dim 384 --decoder_num_resblocks 2 \
  --output_depth --lambda_depth 0.2 \
  --latent_noise_std 0.05 --latent_noise_warmup 5000 \
  --lambda_l1 1.0 --lambda_lpips 1.0 \
  --lambda_grad 0.05 --lambda_temporal 0.1 \
  --lambda_latent_reg 0.01 \
  --batch_size "${BATCH_SIZE}" \
  --accum_steps "${ACCUM_STEPS}" \
  --epochs "${EPOCHS}" \
  --lr "${LEARNING_RATE}" \
  --wd "${WEIGHT_DECAY}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --ema_decay "${EMA_DECAY}" \
  --seq_len 8 --target_size 518 --max_frame_span 32 \
  --num_workers "${NUM_WORKERS}" --dtype fp16 \
  --log_every 100 --eval_every 1 --save_every 1 \
  "${EXTRA_ARGS[@]}"
