#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/spatialvid.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"

# Large Compact DiT experiment.
# This is intentionally a fresh DiT run with a versioned output directory. It
# reuses the existing AE, I0 decoder, and merged latent cache.

# ----------------------------- editable settings -----------------------------
EXPERIMENT_NAME="${EXPERIMENT_NAME:-compact_dit_768d10s6t_h12_v1}"
OUTPUT_DIR="${SCALE_ROOT}/${EXPERIMENT_NAME}"
REMOTE_OUTPUT_DIR="${SCALE_REMOTE_ROOT}/${EXPERIMENT_NAME}"
I0_CKPT="${SCALE_I0_DECODER_CKPT}"

# Keep this disabled by default: this script is for training a new larger DiT.
RESUME="${RESUME:-}"
AUTO_RESUME="${AUTO_RESUME:-0}"
RESUME_MODE="${RESUME_MODE:-full}"

EXPECTED_NNODES="${EXPECTED_NNODES:-32}"
MAX_STEPS="${MAX_STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
ACCUM_STEPS="${ACCUM_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-2}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
MODEL_DIM="${MODEL_DIM:-768}"
SPATIAL_DEPTH="${SPATIAL_DEPTH:-10}"
TEMPORAL_DEPTH="${TEMPORAL_DEPTH:-6}"
NUM_HEADS="${NUM_HEADS:-12}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SHUFFLE_BUFFER="${SHUFFLE_BUFFER:-512}"
SAVE_EVERY="${SAVE_EVERY:-2000}"
EVAL_EVERY="${EVAL_EVERY:-2000}"
LOG_EVERY="${LOG_EVERY:-50}"
SAMPLE_STEPS="${SAMPLE_STEPS:-20}"
echo "  DI_throughput reports raw tokens/s/npu"
MASTER_PORT="${MASTER_PORT:-29614}"
# -----------------------------------------------------------------------------

configure_modelarts_distributed
require_scale_cluster
require_output_url

echo "Large Compact DiT fresh run"
echo "  experiment=${EXPERIMENT_NAME}"
echo "  output=${OUTPUT_DIR}"
echo "  remote=${REMOTE_OUTPUT_DIR}"
echo "  NNODES=${NNODES}, NUM_NPUS=${NUM_NPUS}, WORLD_SIZE=${WORLD_SIZE}"
echo "  model_dim=${MODEL_DIM}, spatial_depth=${SPATIAL_DEPTH}, temporal_depth=${TEMPORAL_DEPTH}, heads=${NUM_HEADS}"
echo "  max_steps=${MAX_STEPS}, batch=${BATCH_SIZE}, accum=${ACCUM_STEPS}, lr=${LEARNING_RATE}"

EXTRA_ARGS=(
  --eval_manifest "${SCALE_EVAL_CACHE_DIR}/manifest.txt"
  --eval_stats "${SCALE_EVAL_CACHE_DIR}/stats.pt"
)

if [[ "${AUTO_RESUME}" -eq 1 || -n "${RESUME}" ]]; then
  RESUME=$(resolve_resume_checkpoint \
    "${RESUME}" \
    "${OUTPUT_DIR}/checkpoint_latest.pt" \
    "${REMOTE_OUTPUT_DIR}/checkpoint_latest.pt" \
    "${LOCAL_CACHE_ROOT}/resume/${EXPERIMENT_NAME}.pt" \
    "${SCALE_MIRROR_ROOT}/${EXPERIMENT_NAME}/checkpoint_latest.pt")
fi
if [[ -n "${RESUME}" ]]; then
  echo "Resuming large Compact DiT from ${RESUME}"
  EXTRA_ARGS+=(--resume "${RESUME}" --resume_mode "${RESUME_MODE}")
fi

ensure_local_checkpoint \
  "${I0_CKPT}" "${SCALE_I0_DECODER_CKPT_URL}" \
  "scale I0 decoder checkpoint" \
  "${SCALE_I0_DECODER_MIRROR_CKPT_URL}"
EXTRA_ARGS+=(--i0_decoder_ckpt "${I0_CKPT}")

start_output_sync "${OUTPUT_DIR}" "${REMOTE_OUTPUT_DIR}"
trap 'stop_output_sync "${OUTPUT_DIR}" "${REMOTE_OUTPUT_DIR}"' EXIT

run_torchrun "${PROJECT}/train_cached_compact_diffusion.py" \
  --manifest "${SCALE_TRAIN_CACHE_DIR}/manifest.txt" \
  --stats "${SCALE_TRAIN_CACHE_DIR}/stats.pt" \
  --output_dir "${OUTPUT_DIR}" \
  --latent_dim "${SCALE_LATENT_DIM}" \
  --latent_grid "${SCALE_LATENT_GRID}" --seq_len 7 \
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
  --sample_steps "${SAMPLE_STEPS}" \
  --dtype fp16 \
  "${EXTRA_ARGS[@]}"
