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
PRETRAINED_LR_SCALE=0.1
WEIGHT_DECAY=1e-2
WARMUP_STEPS=5000
NUM_WORKERS=8
CLIP_DURATION_SECONDS=1.0
# -----------------------------------------------------------------------------

NUM_GPUS=${NUM_GPUS:-8}
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

"${TORCHRUN_BIN}" --nnodes="${NNODES}" --node_rank="${NODE_RANK}" \
  --nproc_per_node="${NUM_GPUS}" --master_addr="${MASTER_ADDR}" \
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
  --disable_temporal_mixer \
  --decoder_base_dim 384 --decoder_num_resblocks 2 \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --accum_steps "${ACCUM_STEPS}" \
    --lr "${LEARNING_RATE}" \
    --pretrained_lr_scale "${PRETRAINED_LR_SCALE}" \
  --wd "${WEIGHT_DECAY}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --max_grad_norm 1.0 --lambda_lpips 1.0 \
  --dtype bf16 --seq_len 8 --target_size 518 \
  --max_frame_span 32 \
  --clip_duration_seconds "${CLIP_DURATION_SECONDS}" \
  --num_workers "${NUM_WORKERS}" \
  --log_every 50 --save_every 1 --eval_every 1 \
  "${EXTRA_ARGS[@]}"
