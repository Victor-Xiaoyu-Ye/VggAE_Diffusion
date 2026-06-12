#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/spatialvid.sh"

# ----------------------------- editable settings -----------------------------
AUTOENCODER_CKPT="${GEOMETRY_AE_CKPT}"
OUTPUT_DIR="${RUN_ROOT}/validation/i0_decoder_overfit"
RESUME=""
DEVICE_ID=3

EPOCHS=500
BATCH_SIZE=1
ACCUM_STEPS=1
LEARNING_RATE=3e-4
PRETRAINED_LR_SCALE=0
WEIGHT_DECAY=0
WARMUP_STEPS=20
LPIPS_WEIGHT=0
SAVE_EVERY=25
EVAL_EVERY=25
LOG_EVERY=10
NUM_WORKERS=0
CLIP_DURATION_SECONDS=1.0
# -----------------------------------------------------------------------------

ensure_spatialvid_splits
require_file "${AUTOENCODER_CKPT}" "geometry autoencoder checkpoint"

EXTRA_ARGS=()
if [[ -n "${RESUME}" ]]; then
  EXTRA_ARGS+=(--resume "${RESUME}")
fi

CUDA_VISIBLE_DEVICES="${DEVICE_ID}" "${PYTHON_BIN}" \
  "${PROJECT}/train_i0_autoencoder.py" \
  --csv "${SPATIALVID_OVERFIT_CSV}" \
  --video_root "${SPATIALVID_VIDEO_ROOT}" \
  --eval_csv "${SPATIALVID_OVERFIT_CSV}" \
  --eval_video_root "${SPATIALVID_VIDEO_ROOT}" \
  --encoder_ckpt "${STREAMVGGT_CKPT}" \
  --autoencoder_ckpt "${AUTOENCODER_CKPT}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_videos "${OVERFIT_VIDEOS}" \
  --disable_temporal_jitter \
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
  --lambda_lpips "${LPIPS_WEIGHT}" \
  --dtype bf16 --seq_len 8 --target_size 518 --max_frame_span 32 \
  --clip_duration_seconds "${CLIP_DURATION_SECONDS}" \
  --num_workers "${NUM_WORKERS}" \
  --log_every "${LOG_EVERY}" \
  --save_every "${SAVE_EVERY}" \
  --eval_every "${EVAL_EVERY}" \
  --disable_temporal_mixer \
  "${EXTRA_ARGS[@]}"
