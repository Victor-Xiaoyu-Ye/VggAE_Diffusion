#!/bin/bash
# Three nodes x8, 48h quality warm-start. No AE retraining or train RGB encoding.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"
source "${SCRIPT_DIR}/../lib/job_cleanup.sh"
export PYTHONPATH="${PROJECT}${PYTHONPATH:+:${PYTHONPATH}}"
export MASTER_PORT="${MASTER_PORT:-29953}" HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-7200}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
configure_modelarts_distributed
[[ "${NNODES}" == 3 && "${NUM_NPUS}" == 8 ]] || {
  echo 'Scene prefix quality run requires 3 nodes x 8 NPUs.' >&2; exit 2;
}
SOURCE_NAMESPACE="${SCENE_SOURCE_NAMESPACE:-scene_rae_c256_15day_v1}"
NAMESPACE="${SCENE_PREFIX_NAMESPACE:-scene_street_prefix4_48h_v1}"
for name in "${SOURCE_NAMESPACE}" "${NAMESPACE}"; do
  [[ "${name}" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid namespace'; exit 2; }
done
[[ "${SOURCE_NAMESPACE}" != "${NAMESPACE}" ]] || { echo 'Source must remain read-only'; exit 2; }
STAGE="${SCENE_PREFIX_STAGE:-all}"
[[ "${STAGE}" =~ ^(all|preflight|train)$ ]] || exit 2
OUT="${SCALE_ROOT}/${NAMESPACE}"
PRIMARY="${SCALE_REMOTE_ROOT}/${NAMESPACE}"
MIRROR="${SCALE_MIRROR_ROOT}/${NAMESPACE}"
LOGS="${OUT}/launcher/node${NODE_RANK}"
mkdir -p "${LOGS}" "${LOCAL_CACHE_ROOT}/tmp"
export TMPDIR="${LOCAL_CACHE_ROOT}/tmp"
export MOX_VIDEO_CACHE_DIR="${LOCAL_CACHE_ROOT}/cache/scene_prefix_eval"
export MOX_VIDEO_CACHE_GB="${SCENE_RGB_CACHE_GB:-20}"
IO="${PROJECT}/scripts/window_run_io.py"
SYNC_PID="" IDLE_GUARD_PID="" TEE_PID=""
PREFIX_STARTED=$(date +%s)
write_exit() {
  "${PYTHON_BIN}" - "${LOGS}" "$1" "$2" <<'PY'
import sys,time
from pathlib import Path
from scripts.window_run_io import atomic
atomic(Path(sys.argv[1])/'exit.json',dict(exit_code=int(sys.argv[2]),
    publication_failed=sys.argv[3]=='failed',time=time.time()))
PY
}
publish_logs() {
  "${PYTHON_BIN}" "${IO}" publish --source "${LOGS}" \
    --root "${PRIMARY}/launcher/node${NODE_RANK}" --root "${MIRROR}/launcher/node${NODE_RANK}"
}
finish() {
  local code=$? publication_code
  trap - EXIT INT TERM
  set +e
  echo "[launcher cleanup] node=${NODE_RANK}, task_exit_code=${code}"
  stop_job_child "${IDLE_GUARD_PID}" 'NPU idle guard' || { [[ "${code}" != 0 ]] || code=70; }
  stop_job_child "${SYNC_PID}" 'log publisher' || { [[ "${code}" != 0 ]] || code=70; }
  echo "Scene prefix launcher finished: node=${NODE_RANK}, exit_code=${code}"
  exec 1>&3 2>&4
  [[ -z "${TEE_PID}" ]] || wait "${TEE_PID}" 2>/dev/null
  write_exit "${code}" pending
  publication_code=$?
  if [[ "${publication_code}" == 0 ]]; then
    publish_logs
    publication_code=$?
  fi
  if [[ "${publication_code}" != 0 ]]; then
    [[ "${code}" != 0 ]] || code=73
    write_exit "${code}" failed
    publish_logs
  fi
  exit "${code}"
}
exec 3>&1 4>&2
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
exec > >(tee -a "${LOGS}/pipeline.log") 2>&1
TEE_PID=$!
"${PYTHON_BIN}" "${IO}" publish --source "${LOGS}" \
  --root "${PRIMARY}/launcher/node${NODE_RANK}" --root "${MIRROR}/launcher/node${NODE_RANK}" --watch &
SYNC_PID=$!
if [[ "${SCENE_NPU_IDLE_GUARD:-1}" == 1 ]]; then
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "${PYTHON_BIN}" "${PROJECT}/scripts/npu_idle_guard.py" \
    --parent-pid "$$" --devices "${NUM_NPUS}" --output "${LOGS}/npu_idle_guard" \
    --threshold "${SCENE_NPU_IDLE_THRESHOLD:-2}" --idle-seconds "${SCENE_NPU_IDLE_SECONDS:-600}" \
    --poll "${SCENE_NPU_IDLE_POLL:-30}" --period "${SCENE_NPU_IDLE_PERIOD:-60}" \
    --burst-seconds "${SCENE_NPU_IDLE_BURST:-10}" --matrix-size "${SCENE_NPU_IDLE_MATRIX:-2048}" &
  IDLE_GUARD_PID=$!
fi
# Used only for prefix-contract verification and additional heldout RGB encoding.
# No historical R7 checkpoint, text encoder, Wan generator or raw train data.
if [[ "${SCENE_PREFIX_NEEDS_ENCODER:-1}" == 1 ]]; then
  ensure_local_checkpoint "${STREAMVGGT_CKPT}" "${STREAMVGGT_URL:-}" 'StreamVGGT' "${STREAMVGGT_MIRROR_URL:-}"
fi
SOURCE_SUFFIX="${SCENE_PREFIX_SOURCE_SUFFIX-production}"
ENTRYPOINT="${SCENE_PREFIX_ENTRYPOINT:-train_scene_prefix.py}"
ARGS=(--config "${SCENE_PREFIX_CONFIG:-${PROJECT}/configs/scene_prefix_48h_v1.json}"
  --cohort "${PROJECT}/configs/scene_cohort_v1.json"
  --selection "${PROJECT}/configs/scene_street_ids_v1.json"
  --encoder "${STREAMVGGT_CKPT}"
  --source-local "${SCALE_ROOT}/${SOURCE_NAMESPACE}${SOURCE_SUFFIX:+/${SOURCE_SUFFIX}}"
  --source-root "${SCALE_REMOTE_ROOT}/${SOURCE_NAMESPACE}${SOURCE_SUFFIX:+/${SOURCE_SUFFIX}}"
  --source-root "${SCALE_MIRROR_ROOT}/${SOURCE_NAMESPACE}${SOURCE_SUFFIX:+/${SOURCE_SUFFIX}}"
  --output "${OUT}" --root "${PRIMARY}" --root "${MIRROR}")
echo "Prefix quality run: ${SOURCE_NAMESPACE} -> ${NAMESPACE}; entry=${ENTRYPOINT}"
echo 'Observed frames0..3; generated frames4..8; no text or future camera conditions.'
if [[ "${STAGE}" == all || "${STAGE}" == preflight ]]; then
  STAGE_LOG_FILE="" run_torchrun "${PROJECT}/${ENTRYPOINT}" "${ARGS[@]}" \
    --stage preflight --startup-seconds "$(( $(date +%s)-PREFIX_STARTED ))"
  PREFIX_STARTED=$(date +%s)
fi
if [[ "${STAGE}" == all || "${STAGE}" == train ]]; then
  # A genuinely new process restores optimizer/FP32 EMA/RNG/cursors after the
  # full-shape preflight. Poor sample quality never blocks this next phase.
  STAGE_LOG_FILE="" run_torchrun "${PROJECT}/${ENTRYPOINT}" "${ARGS[@]}" \
    --stage train --startup-seconds "$(( $(date +%s)-PREFIX_STARTED ))"
fi
echo "Scene prefix run complete: ${MIRROR}/eval/"
