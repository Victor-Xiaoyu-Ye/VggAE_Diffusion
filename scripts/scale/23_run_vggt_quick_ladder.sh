#!/bin/bash
# Fail-closed quick ladder: smoke -> manifold -> exact overfit -> held-out 16.
# The 256 arm is deliberately not automatic; it requires reviewing the 16-arm.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"

R7_NAMESPACE="${R7_NAMESPACE:-r7_t1_c192_geo112_tex80_probe_v1}"
QUICK_NAMESPACE="${QUICK_NAMESPACE:-r7_vggt_quick_geo112_tex80_v1}"
MASTER_PORT="${MASTER_PORT:-29870}"
configure_modelarts_distributed

barrier() {
  run_distributed_barrier
}

if [[ "${SKIP_SMOKE:-0}" != 1 ]]; then
  R7_NAMESPACE="${R7_NAMESPACE}" \
  OUTPUT_NAME="${QUICK_NAMESPACE}/smoke_deterministic" \
  FLOW_OUTPUT_NAME="${QUICK_NAMESPACE}/smoke_flow" \
  bash "${SCRIPT_DIR}/smoke_vggt_generation.sh"
fi
barrier

if [[ "${STOP_AFTER_SMOKE:-0}" == 1 ]]; then
  echo "STOP_AFTER_SMOKE=1: smoke stage complete; stopping before manifold."
  exit 0
fi

MODE=manifold R7_NAMESPACE="${R7_NAMESPACE}" TARGET_INDEX=1 \
OUTPUT_NAME="${QUICK_NAMESPACE}/manifold" \
SAMPLES="${MANIFOLD_SAMPLES:-64}" PREVIEW_SAMPLES="${PREVIEW_SAMPLES:-4}" \
bash "${SCRIPT_DIR}/22_probe_vggt_generation.sh"
barrier

MODE=train GEN_MODE=deterministic R7_NAMESPACE="${R7_NAMESPACE}" \
TARGET_INDEX=1 MAX_SAMPLES=1 EVAL_SAMPLES=1 \
MAX_STEPS="${OVERFIT_STEPS:-500}" EVAL_EVERY="${OVERFIT_EVAL_EVERY:-50}" \
HIDDEN_DIM="${HIDDEN_DIM:-768}" DEPTH="${DEPTH:-4}" \
LAMBDA_RGB=0 LAMBDA_LPIPS=0 EVAL_LPIPS=1 \
OUTPUT_NAME="${QUICK_NAMESPACE}/det_k1_n1" \
bash "${SCRIPT_DIR}/22_probe_vggt_generation.sh"
barrier

OVERFIT_DIR="${SCALE_ROOT}/${QUICK_NAMESPACE}/det_k1_n1"
if [[ "${NODE_RANK}" -eq 0 ]]; then
  "${PYTHON_BIN}" "${PROJECT}/check_single_target_probe.py" \
    --metrics "${OVERFIT_DIR}/metrics.jsonl" --stage overfit \
    --max_latent_mse "${OVERFIT_MAX_MSE:-0.01}" \
    --min_motion_cosine "${OVERFIT_MIN_MOTION_COSINE:-0.90}" \
    --min_norm_ratio "${MIN_NORM_RATIO:-0.8}" \
    --max_norm_ratio "${MAX_NORM_RATIO:-1.2}" \
    --output "${OVERFIT_DIR}/decision.json"
fi
barrier

MODE=train GEN_MODE=deterministic R7_NAMESPACE="${R7_NAMESPACE}" \
TARGET_INDEX=1 MAX_SAMPLES=16 EVAL_SAMPLES=16 \
MAX_STEPS="${N16_STEPS:-1000}" EVAL_EVERY="${N16_EVAL_EVERY:-100}" \
HIDDEN_DIM="${HIDDEN_DIM:-768}" DEPTH="${DEPTH:-4}" \
LAMBDA_RGB=0 LAMBDA_LPIPS=0 EVAL_LPIPS=1 \
OUTPUT_NAME="${QUICK_NAMESPACE}/det_k1_n16" \
bash "${SCRIPT_DIR}/22_probe_vggt_generation.sh"
barrier

N16_DIR="${SCALE_ROOT}/${QUICK_NAMESPACE}/det_k1_n16"
if [[ "${NODE_RANK}" -eq 0 ]]; then
  "${PYTHON_BIN}" "${PROJECT}/check_single_target_probe.py" \
    --metrics "${N16_DIR}/metrics.jsonl" --stage heldout \
    --max_latent_mse "${N16_MAX_MSE:-0.50}" \
    --min_motion_cosine "${N16_MIN_MOTION_COSINE:-0.10}" \
    --min_norm_ratio "${MIN_NORM_RATIO:-0.8}" \
    --max_norm_ratio "${MAX_NORM_RATIO:-1.2}" \
    --min_psnr_gain "${N16_MIN_PSNR_GAIN:-0.0}" \
    --min_lpips_gain "${N16_MIN_LPIPS_GAIN:-0.0}" \
    --output "${N16_DIR}/decision.json"
  cat <<EOF
Quick ladder passed smoke, manifold runtime, one-pair overfit, and held-out n16.
Review:
  ${SCALE_ROOT}/${QUICK_NAMESPACE}/manifold/summary.json
  ${OVERFIT_DIR}/decision.json
  ${N16_DIR}/decision.json
Only then launch MAX_SAMPLES=256 manually; the ladder never starts it blindly.
EOF
fi
barrier
