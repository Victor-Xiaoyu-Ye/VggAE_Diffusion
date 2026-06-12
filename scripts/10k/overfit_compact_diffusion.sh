#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/spatialvid.sh"

# ----------------------------- editable settings -----------------------------
AUTOENCODER_CKPT="${GEOMETRY_AE_CKPT}"
I0_CKPT="${OVERFIT_I0_DECODER_CKPT}"
OUTPUT_DIR="${RUN_ROOT}/validation/compact_diffusion_overfit"
RESUME=""
DEVICE_ID=1

EPOCHS=3000
BATCH_SIZE=1
ACCUM_STEPS=1
LEARNING_RATE=3e-4
WEIGHT_DECAY=0
WARMUP_STEPS=100
MODEL_DIM=384
SPATIAL_DEPTH=4
TEMPORAL_DEPTH=2
NUM_HEADS=6
SAVE_EVERY=100
EVAL_EVERY=100
LOG_EVERY=10
NUM_WORKERS=0
CLIP_DURATION_SECONDS=1.0
NORMALIZATION_BATCHES=1
# -----------------------------------------------------------------------------

ensure_spatialvid_splits
require_file "${AUTOENCODER_CKPT}" "geometry autoencoder checkpoint"
require_file "${I0_CKPT}" "overfit I0 decoder checkpoint"

EXTRA_ARGS=()
if [[ -n "${RESUME}" ]]; then
  EXTRA_ARGS+=(--resume "${RESUME}")
fi

CUDA_VISIBLE_DEVICES="${DEVICE_ID}" "${PYTHON_BIN}" \
  "${PROJECT}/train_compact_diffusion.py" \
  --csv "${SPATIALVID_OVERFIT_CSV}" \
  --video_root "${SPATIALVID_VIDEO_ROOT}" \
  --eval_csv "${SPATIALVID_OVERFIT_CSV}" \
  --eval_video_root "${SPATIALVID_VIDEO_ROOT}" \
  --encoder_ckpt "${STREAMVGGT_CKPT}" \
  --autoencoder_ckpt "${AUTOENCODER_CKPT}" \
  --i0_decoder_ckpt "${I0_CKPT}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_videos "${OVERFIT_VIDEOS}" \
  --disable_temporal_jitter \
  --latent_dim 512 --latent_grid 18 \
  --model_dim "${MODEL_DIM}" \
  --spatial_depth "${SPATIAL_DEPTH}" \
  --temporal_depth "${TEMPORAL_DEPTH}" \
  --num_heads "${NUM_HEADS}" \
  --time_scale 1000 \
  --i0_condition --i0_residual --levels 4 11 17 23 \
  --decoder_base_dim 384 --decoder_num_resblocks 2 \
  --decoder_pixel_shuffle --decoder_temporal_blocks 2 \
  --no_decoder_aux --rescale \
  --normalization_batches "${NORMALIZATION_BATCHES}" \
  --batch_size "${BATCH_SIZE}" \
  --accum_steps "${ACCUM_STEPS}" \
  --epochs "${EPOCHS}" \
  --lr "${LEARNING_RATE}" \
  --wd "${WEIGHT_DECAY}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --ema_decay 0.999 \
  --seq_len 8 --target_size 518 --max_frame_span 32 \
  --clip_duration_seconds "${CLIP_DURATION_SECONDS}" \
  --num_workers "${NUM_WORKERS}" --dtype bf16 \
  --log_every "${LOG_EVERY}" \
  --eval_every "${EVAL_EVERY}" \
  --save_every "${SAVE_EVERY}" \
  --disable_temporal_mixer \
  "${EXTRA_ARGS[@]}"
