#!/bin/bash
# Quick single-target VGGT/R7 generation and manifold probe.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/spatialvid.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"

MODE="${MODE:-train}"                 # train | manifold
TARGET_INDEX="${TARGET_INDEX:-1}"
[[ "${TARGET_INDEX}" == 1 ]] || {
  echo "Quick-probe v1 supports only TARGET_INDEX=1." >&2; exit 2;
}
R7_NAMESPACE="${R7_NAMESPACE:-r7_t1_c192_geo112_tex80_probe_v1}"
R7_CKPT="${R7_CKPT:-${SCALE_ROOT}/${R7_NAMESPACE}/joint/checkpoint_best.pt}"
R7_CKPT_URL="${R7_CKPT_URL:-${SCALE_REMOTE_ROOT}/${R7_NAMESPACE}/joint/checkpoint_best.pt}"
R7_CKPT_MIRROR_URL="${R7_CKPT_MIRROR_URL:-${SCALE_MIRROR_ROOT}/${R7_NAMESPACE}/joint/checkpoint_best.pt}"
SPATIALVID_OFT_ROOT="${SPATIALVID_OFT_ROOT:-obs://yw-ads-training-gy1/data/external/personal/g00833899/y50046448/spatial-vid-hq-oft}"
SPATIALVID_METADATA_URL="${SPATIALVID_OFT_ROOT}/data/train/SpatialVID_HQ_metadata.csv"
SPATIALVID_VIDEO_ROOT="${SPATIALVID_OFT_ROOT}/videos/SpatialVid/HQ/videos"
SPATIALVID_METADATA="${LOCAL_CACHE_ROOT}/metadata/SpatialVID_HQ_metadata_oft.csv"
SPATIALVID_SPLIT_DIR="${RUN_ROOT}/metadata/spatialvid_oft_seed${SPLIT_SEED}"
SPATIALVID_TRAIN_10K_CSV="${SPATIALVID_SPLIT_DIR}/train_10k.csv"
SPATIALVID_EVAL_CSV="${SPATIALVID_SPLIT_DIR}/eval.csv"
OUTPUT_NAME="${OUTPUT_NAME:-r7_vggt_quick_${MODE}_t1}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCALE_ROOT}/${OUTPUT_NAME}}"
REMOTE_OUTPUT_DIR="${REMOTE_OUTPUT_DIR:-${SCALE_REMOTE_ROOT}/${OUTPUT_NAME}}"

[[ "${MODE}" == "train" || "${MODE}" == "manifold" ]] || {
  echo "MODE must be train or manifold." >&2; exit 2;
}
configure_modelarts_distributed
require_scale_cluster
require_output_url
ensure_spatialvid_subset_splits
if [[ "${NODE_RANK}" -ne 0 ]]; then
  echo "Quick probe runs on node 0 only; node ${NODE_RANK} idle."
  exit 0
fi
mkdir -p "${OUTPUT_DIR}"
ensure_local_checkpoint "${R7_CKPT}" "${R7_CKPT_URL}" \
  "R7 quick-probe checkpoint" "${R7_CKPT_MIRROR_URL}"

start_output_sync "${OUTPUT_DIR}" "${REMOTE_OUTPUT_DIR}"
trap 'stop_output_sync "${OUTPUT_DIR}" "${REMOTE_OUTPUT_DIR}"' EXIT
mkdir -p "$(dirname "${STAGE_LOG_FILE}")"

if [[ "${MODE}" == "manifold" ]]; then
  PYTHONPATH="${PROJECT}" "${PYTHON_BIN}" \
    "${PROJECT}/probe_vggt_manifold.py" \
    --csv "${SPATIALVID_EVAL_CSV}" --video_root "${SPATIALVID_VIDEO_ROOT}" \
    --encoder_ckpt "${STREAMVGGT_CKPT}" --r7_ckpt "${R7_CKPT}" \
    --output_dir "${OUTPUT_DIR}" --samples "${SAMPLES:-64}" \
    --preview_samples "${PREVIEW_SAMPLES:-4}" \
    --target_index "${TARGET_INDEX}" --num_workers "${NUM_WORKERS:-2}" \
    --dtype "${DTYPE:-bf16}" 2>&1 | tee -a "${STAGE_LOG_FILE}"
  STATUS=${PIPESTATUS[0]}
else
  EXTRA_ARGS=()
  [[ "${EVAL_LPIPS:-1}" == 1 ]] && EXTRA_ARGS+=(--eval_lpips)
  PYTHONPATH="${PROJECT}" "${PYTHON_BIN}" \
    "${PROJECT}/train_single_target_probe.py" \
    --csv "${SPATIALVID_TRAIN_10K_CSV}" \
    --eval_csv "${SPATIALVID_EVAL_CSV}" \
    --video_root "${SPATIALVID_VIDEO_ROOT}" \
    --encoder_ckpt "${STREAMVGGT_CKPT}" --r7_ckpt "${R7_CKPT}" \
    --output_dir "${OUTPUT_DIR}" --mode "${GEN_MODE:-deterministic}" \
    --target_index "${TARGET_INDEX}" --max_samples "${MAX_SAMPLES:-1}" \
    --eval_samples "${EVAL_SAMPLES:-16}" --max_steps "${MAX_STEPS:-500}" \
    --batch_size "${BATCH_SIZE:-1}" --hidden_dim "${HIDDEN_DIM:-768}" \
    --depth "${DEPTH:-4}" --lr "${LEARNING_RATE:-2e-4}" \
    --lambda_latent "${LAMBDA_LATENT:-1.0}" \
    --lambda_rgb "${LAMBDA_RGB:-0}" \
    --lambda_lpips "${LAMBDA_LPIPS:-0}" \
    --sample_steps "${SAMPLE_STEPS:-20}" \
    --eval_every "${EVAL_EVERY:-50}" --log_every "${LOG_EVERY:-10}" \
    --num_workers "${NUM_WORKERS:-2}" --dtype "${DTYPE:-bf16}" \
    "${EXTRA_ARGS[@]}" 2>&1 | tee -a "${STAGE_LOG_FILE}"
  STATUS=${PIPESTATUS[0]}
fi

if [[ "${STATUS:-1}" -ne 0 ]]; then
  echo "VGGT quick probe failed with status ${STATUS}; see ${STAGE_LOG_FILE}" >&2
  exit "${STATUS}"
fi

printf 'VGGT quick probe complete: %s -> %s\n' "${MODE}" "${OUTPUT_DIR}"
