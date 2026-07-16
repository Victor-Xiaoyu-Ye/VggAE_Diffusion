#!/bin/bash
# E7: dual-stream latent diffusion smoke probe on the 910B cluster.
#
# Trains a small CompactLatentDiT on the frozen R5 dual-stream latent with
# ONLINE encoding (no latent cache). Two serial arms share this launcher:
#
#   bash scripts/scale/07_train_dual_diffusion.sh                    # arm a
#   TARGET_MODE=residual bash scripts/scale/07_train_dual_diffusion.sh  # arm b
#
# Data: the spatial-vid-hq-oft OBS mirror (identical to the H200 AE training
# data), seed-42 10k/64 split — the same clips R5 was trained and evaled on,
# so the latent stats match the tokenizer's training distribution and eval
# numbers are directly comparable to E5's.
#
# 910B notes: bf16 (no scaler), PYTORCH_NPU_ALLOC_CONF=expandable_segments
# via run_torchrun, no RGB decoding on NPU (sampled latents are saved as .pt
# and decoded on the H200 box — DualStreamDecoder's interpolate ops are the
# historically NPU-fragile path).
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/spatialvid.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"

# --- dataset override: the -oft mirror (same content/layout as the H200 box)
SPATIALVID_OFT_ROOT="obs://yw-ads-training-gy1/data/external/personal/g00833899/y50046448/spatial-vid-hq-oft"
SPATIALVID_METADATA_URL="${SPATIALVID_OFT_ROOT}/data/train/SpatialVID_HQ_metadata.csv"
SPATIALVID_VIDEO_ROOT="${SPATIALVID_OFT_ROOT}/videos/SpatialVID/videos"
SPATIALVID_DEPTH_ROOT="${SPATIALVID_OFT_ROOT}/depths/SpatialVID/depths"
SPATIALVID_METADATA="${LOCAL_CACHE_ROOT}/metadata/SpatialVID_HQ_metadata_oft.csv"
SPATIALVID_SPLIT_DIR="${RUN_ROOT}/metadata/spatialvid_oft_seed${SPLIT_SEED}"
SPATIALVID_TRAIN_10K_CSV="${SPATIALVID_SPLIT_DIR}/train_10k.csv"
SPATIALVID_EVAL_CSV="${SPATIALVID_SPLIT_DIR}/eval.csv"
SPATIALVID_OVERFIT_CSV="${SPATIALVID_SPLIT_DIR}/overfit.csv"

# ----------------------------- editable settings -----------------------------
TARGET_MODE="${TARGET_MODE:-absolute}"   # absolute | residual
OUTPUT_DIR="${SCALE_ROOT}/dual_diffusion_${TARGET_MODE}"
REMOTE_OUTPUT_DIR="${SCALE_REMOTE_ROOT}/dual_diffusion_${TARGET_MODE}"
RESUME="${RESUME:-}"
AUTO_RESUME="${AUTO_RESUME:-1}"

# R5 dual-stream checkpoint (compressor + tex_encoder loaded frozen from it).
# Expected in the ref bundle staged by the ModelArts launch command; override
# DUAL_AE_CKPT (local path) or DUAL_AE_CKPT_URL (obs:// path, staged here).
DUAL_AE_CKPT="${DUAL_AE_CKPT:-${VGGAE_REF_ROOT}/checkpoints/e5_dual_stream_r5.pt}"
DUAL_AE_CKPT_URL="${DUAL_AE_CKPT_URL:-}"

MAX_STEPS="${MAX_STEPS:-6000}"
BATCH_SIZE="${BATCH_SIZE:-2}"
ACCUM_STEPS="${ACCUM_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-300}"
MODEL_DIM="${MODEL_DIM:-768}"
SPATIAL_DEPTH="${SPATIAL_DEPTH:-8}"
TEMPORAL_DEPTH="${TEMPORAL_DEPTH:-4}"
NUM_HEADS="${NUM_HEADS:-12}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_EVERY="${EVAL_EVERY:-500}"
SAVE_EVERY="${SAVE_EVERY:-500}"
LOG_EVERY="${LOG_EVERY:-50}"
SAMPLE_STEPS="${SAMPLE_STEPS:-30}"
EVAL_CLIPS="${EVAL_CLIPS:-16}"
MASTER_PORT=29640
# -----------------------------------------------------------------------------

configure_modelarts_distributed
require_scale_cluster
require_output_url
# Subset splits: the -oft mirror is SPARSE (only the 10k subset's videos);
# the availability-filtered split reproduces the H200 seed-42 split instead
# of sampling nonexistent files from the full 360k metadata.
ensure_spatialvid_subset_splits

# Stage the dual-AE checkpoint from OBS if a URL is given and the local ref
# copy is absent.
if [[ ! -s "${DUAL_AE_CKPT}" && -n "${DUAL_AE_CKPT_URL}" ]]; then
  mkdir -p "$(dirname "${DUAL_AE_CKPT}")"
  "${PYTHON_BIN}" "${PROJECT}/scripts/moxing_transfer.py" \
    "${DUAL_AE_CKPT_URL}" "${DUAL_AE_CKPT}"
fi
require_file "${DUAL_AE_CKPT}" "R5 dual-stream AE checkpoint"

echo "E7 launch: arm=${TARGET_MODE} NNODES=${NNODES} NUM_NPUS=${NUM_NPUS}" \
     "WORLD_SIZE=${WORLD_SIZE}"

EXTRA_ARGS=()
if [[ "${AUTO_RESUME}" -eq 1 || -n "${RESUME}" ]]; then
  RESUME=$(resolve_resume_checkpoint \
    "${RESUME}" \
    "${OUTPUT_DIR}/checkpoint_latest.pt" \
    "${REMOTE_OUTPUT_DIR}/checkpoint_latest.pt" \
    "${LOCAL_CACHE_ROOT}/resume/dual_diffusion_${TARGET_MODE}.pt" \
    "${SCALE_MIRROR_ROOT}/dual_diffusion_${TARGET_MODE}/checkpoint_latest.pt")
fi
if [[ -n "${RESUME}" ]]; then
  echo "Resuming from ${RESUME}"
  EXTRA_ARGS+=(--resume "${RESUME}")
fi

start_output_sync "${OUTPUT_DIR}" "${REMOTE_OUTPUT_DIR}"
trap 'stop_output_sync "${OUTPUT_DIR}" "${REMOTE_OUTPUT_DIR}"' EXIT

run_torchrun "${PROJECT}/train_dual_diffusion.py" \
  --target_mode "${TARGET_MODE}" \
  --csv "${SPATIALVID_TRAIN_10K_CSV}" \
  --video_root "${SPATIALVID_VIDEO_ROOT}" \
  --eval_csv "${SPATIALVID_EVAL_CSV}" \
  --encoder_ckpt "${STREAMVGGT_CKPT}" \
  --dual_ae_ckpt "${DUAL_AE_CKPT}" \
  --model_dim "${MODEL_DIM}" \
  --spatial_depth "${SPATIAL_DEPTH}" \
  --temporal_depth "${TEMPORAL_DEPTH}" \
  --num_heads "${NUM_HEADS}" \
  --max_steps "${MAX_STEPS}" \
  --batch_size "${BATCH_SIZE}" \
  --accum_steps "${ACCUM_STEPS}" \
  --lr "${LEARNING_RATE}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --num_workers "${NUM_WORKERS}" \
  --eval_clips "${EVAL_CLIPS}" \
  --sample_steps "${SAMPLE_STEPS}" \
  --eval_every "${EVAL_EVERY}" \
  --save_every "${SAVE_EVERY}" \
  --log_every "${LOG_EVERY}" \
  --seq_len 8 --target_size 518 --clip_duration_seconds 1.0 \
  --dtype bf16 \
  --output_dir "${OUTPUT_DIR}" \
  "${EXTRA_ARGS[@]}"
