#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/spatialvid.sh"

# ----------------------------- editable settings -----------------------------
OUTPUT_DIR="${SCALE_ROOT}/compact_dit"
I0_CKPT="${SCALE_I0_DECODER_CKPT}"
EVAL_I0_PATH=""
RESUME=""

MAX_STEPS=300000
BATCH_SIZE=2
ACCUM_STEPS=8
LEARNING_RATE=1e-4
WEIGHT_DECAY=1e-2
WARMUP_STEPS=5000
MODEL_DIM=768
SPATIAL_DEPTH=8
TEMPORAL_DEPTH=4
NUM_HEADS=12
NUM_WORKERS=4
SHUFFLE_BUFFER=512
SAVE_EVERY=10000
EVAL_EVERY=10000
LOG_EVERY=50
SAMPLE_STEPS=20
# -----------------------------------------------------------------------------

NUM_GPUS=${NUM_GPUS:-8}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29604}

require_file "${SCALE_TRAIN_CACHE_DIR}/manifest.txt" "training cache manifest"
require_file "${SCALE_TRAIN_CACHE_DIR}/stats.pt" "training cache statistics"
require_file "${SCALE_EVAL_CACHE_DIR}/manifest.txt" "evaluation cache manifest"
require_file "${SCALE_EVAL_CACHE_DIR}/stats.pt" "evaluation cache statistics"

EXTRA_ARGS=(
  --eval_manifest "${SCALE_EVAL_CACHE_DIR}/manifest.txt"
  --eval_stats "${SCALE_EVAL_CACHE_DIR}/stats.pt"
)
if [[ -n "${RESUME}" ]]; then
  EXTRA_ARGS+=(--resume "${RESUME}")
fi
if [[ -n "${EVAL_I0_PATH}" ]]; then
  require_file "${EVAL_I0_PATH}" "aligned evaluation I0 image"
  require_file "${I0_CKPT}" "scale I0 decoder checkpoint"
  EXTRA_ARGS+=(
    --eval_i0_path "${EVAL_I0_PATH}"
    --i0_decoder_ckpt "${I0_CKPT}"
  )
fi

torchrun --nnodes="${NNODES}" --node_rank="${NODE_RANK}" \
  --nproc_per_node="${NUM_GPUS}" --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${PROJECT}/train_cached_compact_diffusion.py" \
  --manifest "${SCALE_TRAIN_CACHE_DIR}/manifest.txt" \
  --stats "${SCALE_TRAIN_CACHE_DIR}/stats.pt" \
  --output_dir "${OUTPUT_DIR}" \
  --latent_dim 512 --latent_grid 18 --seq_len 7 \
  --model_dim "${MODEL_DIM}" \
  --spatial_depth "${SPATIAL_DEPTH}" \
  --temporal_depth "${TEMPORAL_DEPTH}" \
  --num_heads "${NUM_HEADS}" \
  --time_scale 1000 \
  --batch_size "${BATCH_SIZE}" \
  --accum_steps "${ACCUM_STEPS}" \
  --max_steps "${MAX_STEPS}" \
  --lr "${LEARNING_RATE}" \
  --wd "${WEIGHT_DECAY}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --ema_decay 0.9999 --max_grad_norm 1.0 \
  --num_workers "${NUM_WORKERS}" \
  --shuffle_buffer "${SHUFFLE_BUFFER}" \
  --log_every "${LOG_EVERY}" \
  --save_every "${SAVE_EVERY}" \
  --eval_every "${EVAL_EVERY}" \
  --sample_steps "${SAMPLE_STEPS}" --dtype bf16 \
  "${EXTRA_ARGS[@]}"
