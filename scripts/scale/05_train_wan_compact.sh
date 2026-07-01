#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/spatialvid.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"

# Wan-initialized cached latent diffusion.
# This is a fresh A/B run against from-scratch Compact DiT. It reuses the same
# geometry AE, I0 decoder, merged latent cache, and normalization statistics.

# ----------------------------- editable settings -----------------------------
EXPERIMENT_NAME="${EXPERIMENT_NAME:-wan_compact_i2v14b480p_v1}"
OUTPUT_DIR="${SCALE_ROOT}/${EXPERIMENT_NAME}"
REMOTE_OUTPUT_DIR="${SCALE_REMOTE_ROOT}/${EXPERIMENT_NAME}"
I0_CKPT="${SCALE_I0_DECODER_CKPT}"

# Default checkpoint selection prefers Wan2.1-I2V-14B-480P when available and
# falls back to Wan2.1-T2V-1.3B. Override WAN_CKPT_DIR for explicit control.
WAN_CKPT_DIR="${WAN_CKPT_DIR}"

RESUME="${RESUME:-}"
AUTO_RESUME="${AUTO_RESUME:-0}"
RESUME_MODE="${RESUME_MODE:-weights}"

EXPECTED_NNODES="${EXPECTED_NNODES:-32}"
MAX_STEPS="${MAX_STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
ACCUM_STEPS="${ACCUM_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-2}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SHUFFLE_BUFFER="${SHUFFLE_BUFFER:-512}"
SAVE_EVERY="${SAVE_EVERY:-2000}"
EVAL_EVERY="${EVAL_EVERY:-2000}"
LOG_EVERY="${LOG_EVERY:-50}"
SAMPLE_STEPS="${SAMPLE_STEPS:-20}"
THROUGHPUT_DIVISOR="${THROUGHPUT_DIVISOR:-20}"
MASTER_PORT="${MASTER_PORT:-29624}"
TRAIN_TEXT_ADAPTER="${TRAIN_TEXT_ADAPTER:-0}"
TRAIN_QKV="${TRAIN_QKV:-0}"
# -----------------------------------------------------------------------------

configure_modelarts_distributed
require_scale_cluster
require_output_url

if [[ ! -d "${WAN_CKPT_DIR}" ]]; then
  echo "Wan checkpoint directory not found: ${WAN_CKPT_DIR}" >&2
  echo "Set WAN_CKPT_DIR to the local copied Wan checkpoint directory." >&2
  exit 1
fi

echo "Wan Compact cached diffusion run"
echo "  experiment=${EXPERIMENT_NAME}"
echo "  output=${OUTPUT_DIR}"
echo "  remote=${REMOTE_OUTPUT_DIR}"
echo "  wan_ckpt=${WAN_CKPT_DIR}"
echo "  NNODES=${NNODES}, NUM_NPUS=${NUM_NPUS}, WORLD_SIZE=${WORLD_SIZE}"
echo "  max_steps=${MAX_STEPS}, batch=${BATCH_SIZE}, accum=${ACCUM_STEPS}, lr=${LEARNING_RATE}"
echo "  DI_throughput divisor=${THROUGHPUT_DIVISOR}"
echo "  train_qkv=${TRAIN_QKV}, train_text_adapter=${TRAIN_TEXT_ADAPTER}"

EXTRA_ARGS=()

if [[ "${TRAIN_TEXT_ADAPTER}" -eq 1 ]]; then
  EXTRA_ARGS+=(--train_text_adapter)
fi
if [[ "${TRAIN_QKV}" -eq 0 ]]; then
  EXTRA_ARGS+=(--freeze_wan_qkv)
fi

if [[ "${AUTO_RESUME}" -eq 1 || -n "${RESUME}" ]]; then
  RESUME=$(resolve_resume_checkpoint \
    "${RESUME}" \
    "${OUTPUT_DIR}/checkpoint_latest.pt" \
    "${REMOTE_OUTPUT_DIR}/checkpoint_latest.pt" \
    "${LOCAL_CACHE_ROOT}/resume/${EXPERIMENT_NAME}.pt" \
    "${SCALE_MIRROR_ROOT}/${EXPERIMENT_NAME}/checkpoint_latest.pt")
fi
if [[ -n "${RESUME}" ]]; then
  echo "Resuming Wan Compact from ${RESUME}"
  EXTRA_ARGS+=(--resume "${RESUME}" --resume_mode "${RESUME_MODE}")
fi

ensure_local_checkpoint \
  "${I0_CKPT}" "${SCALE_I0_DECODER_CKPT_URL}" \
  "scale I0 decoder checkpoint" \
  "${SCALE_I0_DECODER_MIRROR_CKPT_URL}"
EXTRA_ARGS+=(--i0_decoder_ckpt "${I0_CKPT}")

start_output_sync "${OUTPUT_DIR}" "${REMOTE_OUTPUT_DIR}"
trap 'stop_output_sync "${OUTPUT_DIR}" "${REMOTE_OUTPUT_DIR}"' EXIT

run_torchrun "${PROJECT}/train_cached_wan_compact_diffusion.py" \
  --manifest "${SCALE_TRAIN_CACHE_DIR}/manifest.txt" \
  --stats "${SCALE_TRAIN_CACHE_DIR}/stats.pt" \
  --eval_manifest "${SCALE_EVAL_CACHE_DIR}/manifest.txt" \
  --eval_stats "${SCALE_EVAL_CACHE_DIR}/stats.pt" \
  --wan_ckpt_dir "${WAN_CKPT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --latent_dim "${SCALE_LATENT_DIM}" \
  --latent_grid "${SCALE_LATENT_GRID}" --seq_len 7 \
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
  --throughput_divisor "${THROUGHPUT_DIVISOR}" \
  --dtype fp16 \
  "${EXTRA_ARGS[@]}"
