#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/spatialvid.sh"

# ----------------------------- editable settings -----------------------------
AUTOENCODER_CKPT="${GEOMETRY_AE_CKPT}"
I0_CKPT="${I0_DECODER_CKPT}"
OUTPUT_DIR="${RUN_ROOT}/10k/compact_diffusion"
RESUME=""

EPOCHS=50
BATCH_SIZE=2
ACCUM_STEPS=4
LEARNING_RATE=1e-4
WEIGHT_DECAY=1e-2
WARMUP_STEPS=1000
EMA_DECAY=0.9999
MODEL_DIM=768
SPATIAL_DEPTH=8
TEMPORAL_DEPTH=4
NUM_HEADS=12
NUM_WORKERS=4
SAVE_EVERY=5
EVAL_EVERY=5
LOG_EVERY=50
# -----------------------------------------------------------------------------

NUM_NPUS=${NUM_NPUS:-4}
ASCEND_DEVICE_IDS=${ASCEND_DEVICE_IDS:-0,1,2,3}
MASTER_PORT=${MASTER_PORT:-29540}

ensure_spatialvid_splits
require_file "${AUTOENCODER_CKPT}" "geometry autoencoder checkpoint"
require_file "${I0_CKPT}" "I0 decoder checkpoint"

EXTRA_ARGS=()
if [[ -n "${RESUME}" ]]; then
  EXTRA_ARGS+=(--resume "${RESUME}")
fi

ASCEND_RT_VISIBLE_DEVICES="${ASCEND_DEVICE_IDS}" torchrun \
  --nproc_per_node="${NUM_NPUS}" --master_port="${MASTER_PORT}" \
  "${PROJECT}/train_compact_diffusion.py" \
  --csv "${SPATIALVID_TRAIN_10K_CSV}" \
  --video_root "${SPATIALVID_VIDEO_ROOT}" \
  --eval_csv "${SPATIALVID_EVAL_CSV}" \
  --eval_video_root "${SPATIALVID_VIDEO_ROOT}" \
  --encoder_ckpt "${STREAMVGGT_CKPT}" \
  --autoencoder_ckpt "${AUTOENCODER_CKPT}" \
  --i0_decoder_ckpt "${I0_CKPT}" \
  --output_dir "${OUTPUT_DIR}" \
  --latent_dim 512 --latent_grid 18 \
  --model_dim "${MODEL_DIM}" \
  --spatial_depth "${SPATIAL_DEPTH}" \
  --temporal_depth "${TEMPORAL_DEPTH}" \
  --num_heads "${NUM_HEADS}" \
  --time_scale 1000 \
  --i0_condition --i0_residual \
  --levels 4 11 17 23 \
  --decoder_base_dim 384 --decoder_num_resblocks 2 \
  --decoder_pixel_shuffle --decoder_temporal_blocks 2 \
  --rescale --no_decoder_aux \
  --batch_size "${BATCH_SIZE}" \
  --accum_steps "${ACCUM_STEPS}" \
  --epochs "${EPOCHS}" \
  --lr "${LEARNING_RATE}" \
  --wd "${WEIGHT_DECAY}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --ema_decay "${EMA_DECAY}" \
  --seq_len 8 --target_size 518 --max_frame_span 32 \
  --num_workers "${NUM_WORKERS}" --dtype fp16 \
  --log_every "${LOG_EVERY}" \
  --eval_every "${EVAL_EVERY}" \
  --save_every "${SAVE_EVERY}" \
  "${EXTRA_ARGS[@]}"
