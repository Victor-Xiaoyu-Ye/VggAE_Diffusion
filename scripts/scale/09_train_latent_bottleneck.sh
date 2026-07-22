#!/bin/bash
# Stage 09: R6 latent bottleneck finetune (512 -> COMP_DIM) on the cluster.
#
# MIRA-style narrow diffusion space: freezes the R5 tokenizer, trains
# LatentBottleneck + DualStreamDecoder to reconstruct through COMP_DIM
# channels. Output checkpoint carries the FULL dual-AE contract
# (compressor + tex_encoder + bottleneck + decoder) — the only artifact
# stage 10 (compressed diffusion) needs.
#
# Gate vs R5 (24.41 dB / 0.208 LPIPS): PSNR drop <= 0.5 dB.
#
#   bash scripts/scale/09_train_latent_bottleneck.sh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/spatialvid.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"

# --- dataset override: the sparse -oft 10k mirror (same as 07/08) -----------
SPATIALVID_OFT_ROOT="obs://yw-ads-training-gy1/data/external/personal/g00833899/y50046448/spatial-vid-hq-oft"
SPATIALVID_METADATA_URL="${SPATIALVID_OFT_ROOT}/data/train/SpatialVID_HQ_metadata.csv"
SPATIALVID_VIDEO_ROOT="${SPATIALVID_OFT_ROOT}/videos/SpatialVid/HQ/videos"
SPATIALVID_METADATA="${LOCAL_CACHE_ROOT}/metadata/SpatialVID_HQ_metadata_oft.csv"
SPATIALVID_SPLIT_DIR="${RUN_ROOT}/metadata/spatialvid_oft_seed${SPLIT_SEED}"
SPATIALVID_TRAIN_10K_CSV="${SPATIALVID_SPLIT_DIR}/train_10k.csv"
SPATIALVID_EVAL_CSV="${SPATIALVID_SPLIT_DIR}/eval.csv"
SPATIALVID_OVERFIT_CSV="${SPATIALVID_SPLIT_DIR}/overfit.csv"

# ----------------------------- editable settings -----------------------------
COMP_DIM="${COMP_DIM:-128}"
OUTPUT_DIR="${SCALE_ROOT}/latent_bottleneck_c${COMP_DIM}"
REMOTE_OUTPUT_DIR="${SCALE_REMOTE_ROOT}/latent_bottleneck_c${COMP_DIM}"
RESUME="${RESUME:-}"
AUTO_RESUME="${AUTO_RESUME:-1}"

DUAL_AE_CKPT="${DUAL_AE_CKPT:-${VGGAE_REF_ROOT}/checkpoints/e5_dual_stream_r5.pt}"
DUAL_AE_CKPT_URL="${DUAL_AE_CKPT_URL:-}"

MAX_STEPS="${MAX_STEPS:-3000}"
# 910B ~61 GiB: StreamVGGT + DualStreamDecoder already leave <1 GiB free.
# Prefer micro-batch 1 + accum; LPIPS is frame-chunked in the trainer.
BATCH_SIZE="${BATCH_SIZE:-1}"
ACCUM_STEPS="${ACCUM_STEPS:-2}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
BOTTLENECK_LR="${BOTTLENECK_LR:-2e-4}"
LPIPS_WEIGHT="${LPIPS_WEIGHT:-0.5}"
LPIPS_CHUNK_SIZE="${LPIPS_CHUNK_SIZE:-1}"
LPIPS_RESIZE="${LPIPS_RESIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_EVERY="${EVAL_EVERY:-500}"
SAVE_EVERY="${SAVE_EVERY:-500}"
LOG_EVERY="${LOG_EVERY:-50}"
EVAL_CLIPS="${EVAL_CLIPS:-32}"
MASTER_PORT=29660
# -----------------------------------------------------------------------------

configure_modelarts_distributed
require_scale_cluster
require_output_url
ensure_spatialvid_subset_splits

if [[ ! -s "${DUAL_AE_CKPT}" && -n "${DUAL_AE_CKPT_URL}" ]]; then
  mkdir -p "$(dirname "${DUAL_AE_CKPT}")"
  "${PYTHON_BIN}" "${PROJECT}/scripts/moxing_transfer.py" \
    "${DUAL_AE_CKPT_URL}" "${DUAL_AE_CKPT}"
fi
require_file "${DUAL_AE_CKPT}" "R5 dual-stream AE checkpoint"

echo "R6 launch: comp_dim=${COMP_DIM} NNODES=${NNODES} WORLD_SIZE=${WORLD_SIZE}"

EXTRA_ARGS=()
if [[ "${AUTO_RESUME}" -eq 1 || -n "${RESUME}" ]]; then
  RESUME=$(resolve_resume_checkpoint \
    "${RESUME}" \
    "${OUTPUT_DIR}/checkpoint_latest.pt" \
    "${REMOTE_OUTPUT_DIR}/checkpoint_latest.pt" \
    "${LOCAL_CACHE_ROOT}/resume/latent_bottleneck_c${COMP_DIM}.pt" \
    "${SCALE_MIRROR_ROOT}/latent_bottleneck_c${COMP_DIM}/checkpoint_latest.pt")
fi
if [[ -n "${RESUME}" ]]; then
  echo "Resuming from ${RESUME}"
  EXTRA_ARGS+=(--resume "${RESUME}")
  if [[ ! -s "${OUTPUT_DIR}/metrics.jsonl" ]]; then
    mkdir -p "${OUTPUT_DIR}"
    "${PYTHON_BIN}" "${PROJECT}/scripts/moxing_transfer.py" \
      "${REMOTE_OUTPUT_DIR}/metrics.jsonl" \
      "${OUTPUT_DIR}/metrics.jsonl" 2>/dev/null || true
  fi
fi

start_output_sync "${OUTPUT_DIR}" "${REMOTE_OUTPUT_DIR}"
trap 'stop_output_sync "${OUTPUT_DIR}" "${REMOTE_OUTPUT_DIR}"' EXIT

run_torchrun "${PROJECT}/train_latent_bottleneck.py" \
  --comp_dim "${COMP_DIM}" \
  --csv "${SPATIALVID_TRAIN_10K_CSV}" \
  --video_root "${SPATIALVID_VIDEO_ROOT}" \
  --eval_csv "${SPATIALVID_EVAL_CSV}" \
  --encoder_ckpt "${STREAMVGGT_CKPT}" \
  --dual_ae_ckpt "${DUAL_AE_CKPT}" \
  --max_steps "${MAX_STEPS}" \
  --batch_size "${BATCH_SIZE}" \
  --accum_steps "${ACCUM_STEPS}" \
  --lr "${LEARNING_RATE}" \
  --bottleneck_lr "${BOTTLENECK_LR}" \
  --lambda_lpips "${LPIPS_WEIGHT}" \
  --lpips_chunk_size "${LPIPS_CHUNK_SIZE}" \
  --lpips_resize "${LPIPS_RESIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --eval_clips "${EVAL_CLIPS}" \
  --eval_every "${EVAL_EVERY}" \
  --save_every "${SAVE_EVERY}" \
  --log_every "${LOG_EVERY}" \
  --seq_len 8 --target_size 518 --clip_duration_seconds 1.0 \
  --dtype bf16 \
  --output_dir "${OUTPUT_DIR}" \
  "${EXTRA_ARGS[@]}"
