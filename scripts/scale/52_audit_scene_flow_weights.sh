#!/bin/bash
# Read-only raw/EMA replay from stage51 caches and checkpoints, on 1+ nodes.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"
export PYTHONPATH="${PROJECT}${PYTHONPATH:+:${PYTHONPATH}}"
export MASTER_PORT="${MASTER_PORT:-29952}" HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-7200}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
configure_modelarts_distributed
[[ "${NNODES}" =~ ^[1-9][0-9]*$ && "${NUM_NPUS}" == 8 ]] || {
  echo 'Scene replay requires at least one node with 8 NPUs per node.' >&2; exit 2;
}
SCENE_SOURCE_NAMESPACE="${SCENE_SOURCE_NAMESPACE:-scene_rae_c256_15day_v1}"
SCENE_AUDIT_NAMESPACE="${SCENE_AUDIT_NAMESPACE:-scene_rae_raw_ema_audit_v1}"
for namespace in "${SCENE_SOURCE_NAMESPACE}" "${SCENE_AUDIT_NAMESPACE}"; do
  [[ "${namespace}" =~ ^[a-zA-Z0-9_-]+$ ]] || {
    echo "Invalid scene namespace: ${namespace}" >&2; exit 2;
  }
done
[[ "${SCENE_SOURCE_NAMESPACE}" != "${SCENE_AUDIT_NAMESPACE}" ]] || {
  echo 'Source and audit namespaces must differ; source training artifacts are read-only.' >&2; exit 2;
}
PROFILE="${SCENE_AUDIT_PROFILE:-standard}"
[[ "${PROFILE}" =~ ^(smoke|standard)$ ]] || {
  echo 'SCENE_AUDIT_PROFILE must be smoke or standard.' >&2; exit 2;
}
OUT="${SCALE_ROOT}/${SCENE_AUDIT_NAMESPACE}"
PRIMARY="${SCALE_REMOTE_ROOT}/${SCENE_AUDIT_NAMESPACE}"
MIRROR="${SCALE_MIRROR_ROOT}/${SCENE_AUDIT_NAMESPACE}"
LOGS="${OUT}/launcher/node${NODE_RANK}"
mkdir -p "${LOGS}" "${LOCAL_CACHE_ROOT}/tmp"
export TMPDIR="${LOCAL_CACHE_ROOT}/tmp"
IO="${PROJECT}/scripts/window_run_io.py"
SYNC_PID=""
IDLE_GUARD_PID=""
TEE_PID=""

stop_child() {
  local target=$1 active_pid
  [[ -n "${target}" ]] || return 0
  # Never signal an old helper PID that may have been reused by another process.
  for active_pid in $(jobs -pr); do
    if [[ "${active_pid}" == "${target}" ]]; then
      kill "${target}" 2>/dev/null || true
    fi
  done
  wait "${target}" 2>/dev/null || true
}
write_exit() {
  "${PYTHON_BIN}" - "${LOGS}" "$1" "$2" <<'PY'
import sys, time
from pathlib import Path
from scripts.window_run_io import atomic
atomic(Path(sys.argv[1]) / 'exit.json', dict(
    exit_code=int(sys.argv[2]), publication_failed=sys.argv[3] == 'failed',
    publication_receipt='publication_status.json', time=time.time()))
PY
}
publish_logs() {
  "${PYTHON_BIN}" "${IO}" publish --source "${LOGS}" \
    --root "${PRIMARY}/launcher/node${NODE_RANK}" \
    --root "${MIRROR}/launcher/node${NODE_RANK}"
}
finish() {
  local code=$? publication_code
  trap - EXIT INT TERM
  set +e
  # Wait for the guard's terminal status before the final dual publication.
  stop_child "${IDLE_GUARD_PID}"
  stop_child "${SYNC_PID}"
  echo "Scene replay launcher finished: node=${NODE_RANK}, exit_code=${code}"
  # Close and drain tee so the final pipeline tail is part of the snapshot.
  exec 1>&3 2>&4
  [[ -z "${TEE_PID}" ]] || wait "${TEE_PID}" 2>/dev/null
  write_exit "${code}" pending
  publication_code=$?
  if [[ "${publication_code}" == 0 ]]; then
    publish_logs
    publication_code=$?
  fi
  if [[ "${publication_code}" != 0 ]]; then
    echo "Final launcher publication failed on node ${NODE_RANK} (status ${publication_code})." >&2
    [[ "${code}" != 0 ]] || code=73
    write_exit "${code}" failed
    # Best effort to deliver the failure record to the still-available root.
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
  --root "${PRIMARY}/launcher/node${NODE_RANK}" \
  --root "${MIRROR}/launcher/node${NODE_RANK}" --watch &
SYNC_PID=$!
# Covers checkpoint transfers and CPU deserialization without allocating HBM
# until a low-utilization pulse is needed. This work is excluded from DI metrics.
if [[ "${SCENE_NPU_IDLE_GUARD:-1}" == 1 ]]; then
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "${PYTHON_BIN}" "${PROJECT}/scripts/npu_idle_guard.py" \
    --parent-pid "$$" --devices "${NUM_NPUS}" --output "${LOGS}/npu_idle_guard" \
    --threshold "${SCENE_NPU_IDLE_THRESHOLD:-2}" \
    --idle-seconds "${SCENE_NPU_IDLE_SECONDS:-600}" \
    --poll "${SCENE_NPU_IDLE_POLL:-30}" --period "${SCENE_NPU_IDLE_PERIOD:-60}" \
    --burst-seconds "${SCENE_NPU_IDLE_BURST:-10}" \
    --matrix-size "${SCENE_NPU_IDLE_MATRIX:-2048}" &
  IDLE_GUARD_PID=$!
  echo "NPU idle guard started: pid=${IDLE_GUARD_PID}; logs=${LOGS}/npu_idle_guard"
fi
echo "Scene raw/EMA audit: source=${SCENE_SOURCE_NAMESPACE}/production, output=${SCENE_AUDIT_NAMESPACE}"
echo "Cluster: nodes=${NNODES}, NPUs_per_node=${NUM_NPUS}, profile=${PROFILE}"
# The Python coordinator stages eval latents/decoder and each checkpoint once
# per node. No R7, encoder, text encoder or raw RGB dataset is needed for replay.
STAGE_LOG_FILE="" run_torchrun "${PROJECT}/audit_scene_flow_weights.py" \
  --source-local "${SCALE_ROOT}/${SCENE_SOURCE_NAMESPACE}/production" \
  --source-root "${SCALE_REMOTE_ROOT}/${SCENE_SOURCE_NAMESPACE}/production" \
  --source-root "${SCALE_MIRROR_ROOT}/${SCENE_SOURCE_NAMESPACE}/production" \
  --output "${OUT}" --root "${PRIMARY}" --root "${MIRROR}" \
  --profile "${PROFILE}" \
  --sample-steps "${SCENE_AUDIT_STEPS:-32}" \
  --sample-method "${SCENE_AUDIT_METHOD:-euler}" \
  --cases-per-domain "${SCENE_AUDIT_CASES_PER_DOMAIN:-0}" \
  --max-cases "${SCENE_AUDIT_MAX_CASES:-0}"
echo "Scene raw/EMA replay completed: ${MIRROR}/"
