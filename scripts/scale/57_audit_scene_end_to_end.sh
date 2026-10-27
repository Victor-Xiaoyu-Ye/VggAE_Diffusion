#!/bin/bash
# Offline, read-only full-chain diagnosis; three independent node workers.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"
source "${SCRIPT_DIR}/../lib/job_cleanup.sh"
export PYTHONPATH="${PROJECT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MASTER_PORT="${MASTER_PORT:-29957}"
configure_modelarts_distributed
[[ "${NNODES}" == 3 && "${NUM_NPUS}" == 8 && "${NODE_RANK}" =~ ^[0-2]$ ]] || {
  echo 'Stage57 uses three nodes x eight allocated NPUs, one diagnostic worker on NPU0 per node.' >&2; exit 2;
}
[[ "${MASTER_PORT}" =~ ^[0-9]+$ && "${MASTER_PORT}" -ge 1024 && "${MASTER_PORT}" -le 65533 ]] || {
  echo 'MASTER_PORT must be in 1024..65533; work/merge use the next two ports.' >&2; exit 2;
}
STAGE="${SCENE_AUDIT_STAGE:-all}"
[[ "${STAGE}" == all || "${STAGE}" == dry ]] || { echo 'SCENE_AUDIT_STAGE must be all or dry.' >&2; exit 2; }
NAMESPACE="${SCENE_AUDIT_NAMESPACE:-scene_end_to_end_audit_v1}"
SOURCE_NAMESPACE="${SCENE_AUDIT_SOURCE_NAMESPACE:-scene_street_prefix4_48h_v1}"
for name in "${NAMESPACE}" "${SOURCE_NAMESPACE}"; do
  [[ "${name}" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid namespace.' >&2; exit 2; }
done
[[ "${NAMESPACE}" != "${SOURCE_NAMESPACE}" ]] || { echo 'Source must remain read-only.' >&2; exit 2; }
CONFIG="${SCENE_AUDIT_CONFIG:-${PROJECT}/configs/scene_end_to_end_audit_v1.json}"
[[ -f "${CONFIG}" ]] || { echo "Missing config: ${CONFIG}" >&2; exit 2; }
OUT="${SCALE_ROOT}/${NAMESPACE}"
PRIMARY="${SCALE_REMOTE_ROOT}/${NAMESPACE}"
MIRROR="${SCALE_MIRROR_ROOT}/${NAMESPACE}"
SOURCE_LOCAL="${SCENE_AUDIT_SOURCE_LOCAL:-${SCALE_ROOT}/${SOURCE_NAMESPACE}}"
SOURCE_PRIMARY="${SCENE_AUDIT_SOURCE_PRIMARY:-${SCALE_REMOTE_ROOT}/${SOURCE_NAMESPACE}}"
SOURCE_MIRROR="${SCENE_AUDIT_SOURCE_MIRROR:-${SCALE_MIRROR_ROOT}/${SOURCE_NAMESPACE}}"
TIMEOUT="${SCENE_AUDIT_TIMEOUT_SECONDS:-86400}"
[[ "${TIMEOUT}" =~ ^[1-9][0-9]*$ ]] || { echo 'Diagnostic timeout must be a positive integer.' >&2; exit 2; }
if [[ "${STAGE}" == dry ]]; then
  echo "Stage57 dry plan: config=${CONFIG}; source=${SOURCE_LOCAL}"
  echo 'Three independent work processes: NODE_RANK partitions cases, local NPU0 only; no HCCL or training.'
  echo "Existing Stream checkpoint=${STREAMVGGT_CKPT}; no public model download."
  echo "Worker coordinator port=$((MASTER_PORT+1)); leader merge port=$((MASTER_PORT+2)); timeout=${TIMEOUT}s"
  echo "Outputs=${OUT}; primary=${PRIMARY}; mirror=${MIRROR}"
  echo 'Idle guard covers all eight devices per node, including CPU/IO and waiting periods.'
  exit 0
fi
LOGS="${OUT}/launcher/node${NODE_RANK}"
mkdir -p "${LOGS}" "${LOCAL_CACHE_ROOT}/tmp"
export TMPDIR="${LOCAL_CACHE_ROOT}/tmp"
export MOX_VIDEO_CACHE_DIR="${LOCAL_CACHE_ROOT}/cache/scene_end_to_end_rgb"
export MOX_VIDEO_CACHE_GB="${SCENE_AUDIT_RGB_CACHE_GB:-12}"
IO="${PROJECT}/scripts/window_run_io.py"
SYNC_PID="" IDLE_GUARD_PID="" TEE_PID=""
write_exit() {
  "${PYTHON_BIN}" - "${LOGS}" "$1" "$2" <<'PY'
import sys,time
from pathlib import Path
from scripts.window_run_io import atomic
atomic(Path(sys.argv[1])/'exit.json',dict(exit_code=int(sys.argv[2]),
    publication_failed=sys.argv[3]=='failed',time=time.time()))
PY
}
write_phase() {
  "${PYTHON_BIN}" - "${LOGS}" "$1" "$2" "${NODE_RANK}" <<'PY'
import sys,time
from pathlib import Path
from scripts.window_run_io import atomic
atomic(Path(sys.argv[1])/('phase_'+sys.argv[2]+'.json'),dict(phase=sys.argv[2],
    exit_code=int(sys.argv[3]),node=int(sys.argv[4]),time=time.time()))
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
  echo "Scene full-chain audit finished: node=${NODE_RANK}, exit_code=${code}"
  exec 1>&3 2>&4
  [[ -z "${TEE_PID}" ]] || wait "${TEE_PID}" 2>/dev/null
  write_exit "${code}" pending
  publication_code=$?
  if [[ "${publication_code}" == 0 ]]; then publish_logs; publication_code=$?; fi
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
EFFECTIVE_CONFIG="${LOGS}/effective_config.json"
"${PYTHON_BIN}" - "${CONFIG}" "${EFFECTIVE_CONFIG}" <<'PY'
import json,sys
from pathlib import Path
from scripts.window_run_io import atomic
cfg=json.loads(Path(sys.argv[1]).read_text(encoding='utf8'))
target=Path(sys.argv[2])
if target.exists() and json.loads(target.read_text(encoding='utf8'))!=cfg:
    raise SystemExit('Audit config changed in this namespace; select a fresh SCENE_AUDIT_NAMESPACE.')
atomic(target,cfg)
PY
for source in "${STREAMVGGT_URL:-}" "${STREAMVGGT_MIRROR_URL:-}"; do
  [[ "${source}" != http://* && "${source}" != https://* ]] || {
    echo 'Use existing local/OBS StreamVGGT weights; public downloads are not allowed in this launcher.' >&2; exit 2;
  }
done
STAGING_CODE=0
# Library helper can call exit: isolate it so the runner can still report an
# unavailable teacher and execute the independent cached AE/DiT diagnostics.
(ensure_local_checkpoint "${STREAMVGGT_CKPT}" "${STREAMVGGT_URL:-}" \
  'full-head StreamVGGT diagnostic teacher' "${STREAMVGGT_MIRROR_URL:-}") || STAGING_CODE=$?
write_phase checkpoint "${STAGING_CODE}"
ARGS=(--config "${EFFECTIVE_CONFIG}" --output "${OUT}" --root "${PRIMARY}" --root "${MIRROR}"
  --source-local "${SOURCE_LOCAL}" --source-root "${SOURCE_PRIMARY}" --source-root "${SOURCE_MIRROR}"
  --encoder "${STREAMVGGT_CKPT}" --device npu:0)
echo "Stage57 work: node=${NODE_RANK}/${NNODES}, NPU0, config=${EFFECTIVE_CONFIG}"
echo 'Component anomalies are saved by the runner; captured component failures must not stop unrelated branches.'
echo 'DI_throughput excludes idle-guard pulses. No optimizer or training stage is invoked.'
WORK_CODE=0
MASTER_PORT=$((MASTER_PORT+1)) OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
  PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}" \
  "${PYTHON_BIN}" -u "${PROJECT}/scripts/coordinated_scene_controls.py" --timeout "${TIMEOUT}" -- \
  "${PYTHON_BIN}" -u "${PROJECT}/audit_scene_end_to_end.py" "${ARGS[@]}" --stage work || WORK_CODE=$?
write_phase work "${WORK_CODE}"
MERGE_CODE=0
# Preserve partial diagnostics even after a worker-process failure. Ports are
# distinct, so a previous coordinator's TCPStore cannot supply stale results.
MASTER_PORT=$((MASTER_PORT+2)) AUDIT_TIMEOUT_SECONDS="${TIMEOUT}" \
  "${PYTHON_BIN}" -u "${PROJECT}/scripts/coordinated_diagnostic.py" \
  "${PYTHON_BIN}" -u "${PROJECT}/audit_scene_end_to_end.py" "${ARGS[@]}" --stage merge || MERGE_CODE=$?
write_phase merge "${MERGE_CODE}"
FINAL_CODE=0
for code in "${STAGING_CODE}" "${WORK_CODE}" "${MERGE_CODE}"; do
  if [[ "${code}" != 0 ]]; then FINAL_CODE="${code}"; break; fi
done
echo "Stage57 collected: checkpoint=${STAGING_CODE}, work=${WORK_CODE}, merge=${MERGE_CODE}; ${MIRROR}"
exit "${FINAL_CODE}"
