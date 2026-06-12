#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/spatialvid.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"

# ----------------------------- editable settings -----------------------------
AUTOENCODER_CKPT="${GEOMETRY_AE_CKPT}"
OUTPUT_DIR="${RUN_ROOT}/10k/i0_decoder"
REMOTE_OUTPUT_DIR="${REMOTE_RUN_ROOT}/10k/i0_decoder"
RESUME=""

EPOCHS=50
BATCH_SIZE=1
ACCUM_STEPS=4
LEARNING_RATE=1e-4
PRETRAINED_LR_SCALE=0.1
WEIGHT_DECAY=1e-2
WARMUP_STEPS=500
EMA_DECAY=0.999
LPIPS_WEIGHT=1.0
NUM_WORKERS=4
DECODE_RETRIES=8
CLIP_DURATION_SECONDS=1.0
SAVE_EVERY=5
EVAL_EVERY=5
LOG_EVERY=50
# -----------------------------------------------------------------------------

MASTER_PORT=${MASTER_PORT:-29530}

configure_modelarts_distributed
ensure_spatialvid_splits
require_file "${AUTOENCODER_CKPT}" "geometry autoencoder checkpoint"

EXTRA_ARGS=()
if [[ -n "${RESUME}" ]]; then
  EXTRA_ARGS+=(--resume "${RESUME}")
fi

start_output_sync "${OUTPUT_DIR}" "${REMOTE_OUTPUT_DIR}"
trap 'stop_output_sync "${OUTPUT_DIR}" "${REMOTE_OUTPUT_DIR}"' EXIT

run_torchrun "${PROJECT}/train_i0_autoencoder.py" \
  --csv "${SPATIALVID_TRAIN_10K_CSV}" \
  --video_root "${SPATIALVID_VIDEO_ROOT}" \
  --eval_csv "${SPATIALVID_EVAL_CSV}" \
  --eval_video_root "${SPATIALVID_VIDEO_ROOT}" \
  --encoder_ckpt "${STREAMVGGT_CKPT}" \
  --autoencoder_ckpt "${AUTOENCODER_CKPT}" \
  --output_dir "${OUTPUT_DIR}" \
  --latent_dim 512 --latent_grid 18 \
  --decoder_base_dim 384 --decoder_num_resblocks 2 \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --accum_steps "${ACCUM_STEPS}" \
  --lr "${LEARNING_RATE}" \
  --pretrained_lr_scale "${PRETRAINED_LR_SCALE}" \
  --wd "${WEIGHT_DECAY}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --max_grad_norm 1.0 \
  --ema_decay "${EMA_DECAY}" \
  --lambda_lpips "${LPIPS_WEIGHT}" \
  --dtype fp16 --seq_len 8 --target_size 518 --max_frame_span 32 \
  --clip_duration_seconds "${CLIP_DURATION_SECONDS}" \
  --disable_temporal_mixer \
  --num_workers "${NUM_WORKERS}" \
  --decode_retries "${DECODE_RETRIES}" \
  --log_every "${LOG_EVERY}" \
  --save_every "${SAVE_EVERY}" \
  --eval_every "${EVAL_EVERY}" \
  "${EXTRA_ARGS[@]}"
