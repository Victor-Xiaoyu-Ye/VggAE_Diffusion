#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/spatialvid.sh"

# ----------------------------- editable settings -----------------------------
OUTPUT_DIR="${RUN_ROOT}/10k/geometry_autoencoder"
RESUME=""

EPOCHS=120
BATCH_SIZE=2
ACCUM_STEPS=15
LEARNING_RATE=1e-4
WEIGHT_DECAY=1e-2
WARMUP_STEPS=500
EMA_DECAY=0.999
NUM_WORKERS=10
CLIP_DURATION_SECONDS=1.0
SAVE_EVERY=5
EVAL_EVERY=5
LOG_EVERY=100
# -----------------------------------------------------------------------------

# Distributed launch settings are intentionally read from the cluster.
NUM_GPUS=${NUM_GPUS:-4}
GPU_IDS=${GPU_IDS:-2,3,4,5}
MASTER_PORT=${MASTER_PORT:-29510}

ensure_spatialvid_splits
EXTRA_ARGS=()
if [[ -n "${RESUME}" ]]; then
  EXTRA_ARGS+=(--resume "${RESUME}")
fi

CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${TORCHRUN_BIN}" \
  --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" \
  "${PROJECT}/train_autoencoder.py" \
  --csv "${SPATIALVID_TRAIN_10K_CSV}" \
  --video_root "${SPATIALVID_VIDEO_ROOT}" \
  --eval_csv "${SPATIALVID_EVAL_CSV}" \
  --eval_video_root "${SPATIALVID_VIDEO_ROOT}" \
  --encoder_ckpt "${STREAMVGGT_CKPT}" \
  --output_dir "${OUTPUT_DIR}" \
  --latent_dim 512 --latent_grid 18 \
  --levels 4 11 17 23 \
  --decoder_base_dim 384 --decoder_num_resblocks 2 \
  --latent_noise_std 0.05 --latent_noise_warmup 1000 \
  --lambda_l1 1.0 --lambda_lpips 1.0 \
  --lambda_grad 0.05 --lambda_temporal 0.05 \
  --lambda_latent_reg 0.01 \
  --batch_size "${BATCH_SIZE}" \
  --accum_steps "${ACCUM_STEPS}" \
  --epochs "${EPOCHS}" \
  --lr "${LEARNING_RATE}" \
  --wd "${WEIGHT_DECAY}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --ema_decay "${EMA_DECAY}" \
  --seq_len 8 --target_size 518 --max_frame_span 32 \
  --clip_duration_seconds "${CLIP_DURATION_SECONDS}" \
  --num_workers "${NUM_WORKERS}" --dtype bf16 \
  --log_every "${LOG_EVERY}" \
  --eval_every "${EVAL_EVERY}" \
  --save_every "${SAVE_EVERY}" \
  "${EXTRA_ARGS[@]}"
