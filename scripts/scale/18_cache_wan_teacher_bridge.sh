#!/bin/bash
# Build a small frozen Wan-VAE/native-patch teacher cache for bridge diagnosis.
# The teacher is never used by production inference.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/spatialvid.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"

configure_modelarts_distributed
require_scale_cluster
require_output_url
[[ "${NODE_RANK}" -eq 0 ]] || exit 0
ensure_spatialvid_subset_splits

WAN_T2V_13B_DIR="${WAN_T2V_13B_DIR:-${VGGAE_REF_ROOT}/Wan2.1-T2V-1.3B}"
WAN_VAE_CKPT="${WAN_VAE_CKPT:-${WAN_T2V_13B_DIR}/Wan2.1_VAE.pth}"
R7_NAMESPACE="${R7_NAMESPACE:-r7_t2_c192_v3}"
R7_CKPT="${R7_CKPT:-${SCALE_ROOT}/${R7_NAMESPACE}/joint/checkpoint_best.pt}"
R7_CKPT_URL="${R7_CKPT_URL:-${SCALE_REMOTE_ROOT}/${R7_NAMESPACE}/joint/checkpoint_best.pt}"
R7_CKPT_MIRROR_URL="${R7_CKPT_MIRROR_URL:-${SCALE_MIRROR_ROOT}/${R7_NAMESPACE}/joint/checkpoint_best.pt}"
TEACHER_OUTPUT="${WAN_TEACHER_OUTPUT:-${PERSISTENT_OBS_ROOT}/teacher_cache/r7_wan_native_1k_v1}"
require_dir "${WAN_T2V_13B_DIR}" "Wan2.1-T2V-1.3B"
require_file "${WAN_VAE_CKPT}" "Wan VAE checkpoint"
ensure_local_checkpoint "${R7_CKPT}" "${R7_CKPT_URL}" \
  "R7 teacher checkpoint" "${R7_CKPT_MIRROR_URL}"

PYTHONPATH="${PROJECT}" "${PYTHON_BIN}" \
  "${PROJECT}/cache_wan_teacher_bridge.py" \
  --csv "${SPATIALVID_TRAIN_10K_CSV}" \
  --video_root "${SPATIALVID_VIDEO_ROOT}" \
  --wan_ckpt_dir "${WAN_T2V_13B_DIR}" \
  --wan_vae_ckpt "${WAN_VAE_CKPT}" \
  --r7_ckpt "${R7_CKPT}" --encoder_ckpt "${STREAMVGGT_CKPT}" \
  --output_dir "${TEACHER_OUTPUT}" \
  --max_videos "${MAX_VIDEOS:-1024}" \
  --samples_per_shard "${SAMPLES_PER_SHARD:-128}" \
  --num_workers "${NUM_WORKERS:-2}" --dtype bf16

printf 'Wan teacher cache: %s\n' "${TEACHER_OUTPUT}"
