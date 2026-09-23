#!/bin/bash
# Three nodes x 8 Ascend: RGB-only static/dynamic RAE -> image/video flow.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"
export PYTHONPATH="${PROJECT}${PYTHONPATH:+:${PYTHONPATH}}"
export MASTER_PORT="${MASTER_PORT:-29951}" HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-7200}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
configure_modelarts_distributed
[[ "${NNODES}" == 3 && "${NUM_NPUS}" == 8 ]] || { echo 'This budget/model is configured for 3 nodes x 8 NPUs'; exit 2; }
SCENE_NAMESPACE="${SCENE_NAMESPACE:-scene_rae_c256_15day_v1}"
[[ "${SCENE_NAMESPACE}" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
SCENE_STAGE="${SCENE_STAGE:-all}"
[[ "${SCENE_STAGE}" =~ ^(all|rehearsal|ae|cache|image|video)$ ]] || exit 2
OUT="${SCALE_ROOT}/${SCENE_NAMESPACE}"
PRIMARY="${SCALE_REMOTE_ROOT}/${SCENE_NAMESPACE}"
MIRROR="${SCALE_MIRROR_ROOT}/${SCENE_NAMESPACE}"
LOGS="${OUT}/launcher/node${NODE_RANK}"
mkdir -p "${LOGS}" "${LOCAL_CACHE_ROOT}/tmp"
export TMPDIR="${LOCAL_CACHE_ROOT}/tmp"
export MOX_VIDEO_CACHE_GB="${SCENE_RGB_CACHE_GB:-200}"
export MOX_VIDEO_CACHE_DIR="${LOCAL_CACHE_ROOT}/cache/scene_rgb"
R7_CKPT="${R7_CKPT:-${SCALE_ROOT}/r7_t2_c192_v2/joint/checkpoint_best.pt}"
IO="${PROJECT}/scripts/window_run_io.py"
# Only the launcher log directory is watched. Model/cache payloads and their
# commit receipts are published synchronously by their owning training rank.
"${PYTHON_BIN}" "${IO}" publish --source "${LOGS}" --root "${PRIMARY}/launcher/node${NODE_RANK}" \
  --root "${MIRROR}/launcher/node${NODE_RANK}" --watch &
SYNC_PID=$!
finish() {
  local code=$?
  trap - EXIT
  kill "${SYNC_PID}" 2>/dev/null || true
  wait "${SYNC_PID}" 2>/dev/null || true
  "${PYTHON_BIN}" - "${LOGS}" "${code}" <<'PY'
import sys,time
from pathlib import Path
from scripts.window_run_io import atomic
atomic(Path(sys.argv[1])/'exit.json',dict(exit_code=int(sys.argv[2]),time=time.time()))
PY
  "${PYTHON_BIN}" "${IO}" publish --source "${LOGS}" --root "${PRIMARY}/launcher/node${NODE_RANK}" \
    --root "${MIRROR}/launcher/node${NODE_RANK}" || true
  exit "${code}"
}
trap finish EXIT
exec > >(tee -a "${LOGS}/pipeline.log") 2>&1
ensure_local_checkpoint "${R7_CKPT}" "${SCALE_REMOTE_ROOT}/r7_t2_c192_v2/joint/checkpoint_best.pt" \
  'spatial R7 warm-start' "${SCALE_MIRROR_ROOT}/r7_t2_c192_v2/joint/checkpoint_best.pt"
ensure_local_checkpoint "${STREAMVGGT_CKPT}" "${STREAMVGGT_URL:-}" 'StreamVGGT' "${STREAMVGGT_MIRROR_URL:-}"
CONFIG="${SCENE_CONFIG:-${PROJECT}/configs/scene_15day_v1.json}"
COHORT="${SCENE_COHORT:-${PROJECT}/configs/scene_cohort_v1.json}"
# Reuse the existing full-HQ text bank; new WAI images have no invented captions.
# SCENE_TEXT_ROOT=none deliberately selects image-only/unconditional training.
TEXT="${SCENE_TEXT_ROOT:-${PERSISTENT_OBS_ROOT}/text_embeddings/fullhq_umt5_256_v1}"
TEXT_ARGS=()
[[ "${TEXT}" == none ]] || TEXT_ARGS=(--text "${TEXT}")
ARGS=(--config "${CONFIG}" --cohort "${COHORT}" --encoder "${STREAMVGGT_CKPT}" --r7 "${R7_CKPT}" "${TEXT_ARGS[@]}")
if [[ "${SCENE_STAGE}" == all || "${SCENE_STAGE}" == rehearsal ]]; then
  # Same full model, RGB shape and 24-rank topology as production. Two updates
  # per trainer, real save/load of optimizer/RNG, cache dual write/read, sampling.
  # Poor pictures/PSNR never prevent automatic progression to the full run.
  for stage in ae cache image video; do
    STAGE_LOG_FILE="" run_torchrun "${PROJECT}/train_scene_pipeline.py" "${ARGS[@]}" --stage "${stage}" --smoke \
      --output "${OUT}/rehearsal" --root "${PRIMARY}/rehearsal" --root "${MIRROR}/rehearsal"
  done
  [[ "${SCENE_STAGE}" != rehearsal ]] || exit 0
fi
for stage in ae cache image video; do
  if [[ "${SCENE_STAGE}" == all || "${SCENE_STAGE}" == "${stage}" ]]; then
    STAGE_LOG_FILE="" run_torchrun "${PROJECT}/train_scene_pipeline.py" "${ARGS[@]}" --stage "${stage}" \
      --output "${OUT}/production" --root "${PRIMARY}/production" --root "${MIRROR}/production"
  fi
done
echo "Scene RAE pipeline finished: ${MIRROR}/production/video/"
