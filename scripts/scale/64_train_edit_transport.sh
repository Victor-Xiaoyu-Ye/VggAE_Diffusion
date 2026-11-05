#!/bin/bash
# Train an edit-transport module, then produce three matched native VACE outputs.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"
source "${SCRIPT_DIR}/../lib/job_cleanup.sh"
export PYTHONPATH="${PROJECT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MASTER_PORT="${MASTER_PORT:-30500}" HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-7200}"
configure_modelarts_distributed
STAGE="${LEARNED_STAGE:-all}"
[[ "${STAGE}" =~ ^(all|smoke|prepare|train|predict|render|summarize|dry)$ ]] || exit 2
[[ "${NNODES}" == 3 && "${NUM_NPUS}" == 8 && "${NODE_RANK}" =~ ^[0-2]$ ]] || {
  echo 'Stage64 uses three nodes x eight NPUs.'; exit 2;
}
[[ "${MASTER_PORT}" =~ ^[0-9]+$ && "${MASTER_PORT}" -ge 1024 && "${MASTER_PORT}" -le 65400 ]] || exit 2
BASE_PORT="${MASTER_PORT}"
NAMESPACE="${LEARNED_NAMESPACE:-edit_transport_learned_v1}"
[[ "${STAGE}" != smoke ]] || NAMESPACE="${NAMESPACE}_smoke"
[[ "${NAMESPACE}" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
CONFIG="${LEARNED_CONFIG:-${PROJECT}/configs/edit_transport_learned_v1.json}"
BANK_NAMESPACE="${LEARNED_BANK_NAMESPACE:-vace_edit_pilot_v1}"
REFERENCE_NAMESPACE="${LEARNED_REFERENCE_NAMESPACE:-edit_reference_video_v1}"
[[ "${BANK_NAMESPACE}" =~ ^[a-zA-Z0-9_-]+$ && "${REFERENCE_NAMESPACE}" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
BANK="${LEARNED_BANK_LOCAL:-${LOCAL_CACHE_ROOT}/cache_latents/${BANK_NAMESPACE}}"
BANK_PRIMARY="${LEARNED_BANK_PRIMARY:-${SCALE_REMOTE_ROOT}/${BANK_NAMESPACE}/cache_latents}"
BANK_MIRROR="${LEARNED_BANK_MIRROR:-${PERSISTENT_OBS_ROOT}/cache_latents/${BANK_NAMESPACE}}"
REFERENCES="${LEARNED_REFERENCE_LOCAL:-${SCALE_ROOT}/${REFERENCE_NAMESPACE}}"
REFERENCE_PRIMARY="${LEARNED_REFERENCE_PRIMARY:-${SCALE_REMOTE_ROOT}/${REFERENCE_NAMESPACE}}"
REFERENCE_MIRROR="${LEARNED_REFERENCE_MIRROR:-${SCALE_MIRROR_ROOT}/${REFERENCE_NAMESPACE}}"
OUT="${SCALE_ROOT}/${NAMESPACE}"
PRIMARY="${SCALE_REMOTE_ROOT}/${NAMESPACE}"
MIRROR="${SCALE_MIRROR_ROOT}/${NAMESPACE}"
MODEL="${EDIT_MODEL:-${VGGAE_REF_ROOT}/Wan2.1-VACE-14B}"
export EDIT_RAE="${EDIT_RAE:-${LOCAL_CACHE_ROOT}/editing_teacher/scene_c256_best_joint.pt}"
if [[ "${STAGE}" == dry ]]; then
  echo "Stage64 ${NAMESPACE}: read-only Stage60 bank=${BANK_NAMESPACE}; anchors=${REFERENCE_NAMESPACE}."
  echo 'Source and legitimate anchor -> trainable RAE decoder residual -> Native / RGB writeback / learned RAE writeback.'
  echo 'Frozen Stream/RAE/VACE14B; existing cluster staging only, no internet installation or new weights.'
  echo "Outputs=${PRIMARY}; mirror=${MIRROR}"
  exit 0
fi
LOGS="${OUT}/launcher/node${NODE_RANK}"
mkdir -p "${LOGS}" "${LOCAL_CACHE_ROOT}/tmp"
export TMPDIR="${LOCAL_CACHE_ROOT}/tmp" ASCEND_PROCESS_LOG_PATH="${LOGS}/npu"
IO="${PROJECT}/scripts/window_run_io.py"
SYNC_PID='' IDLE_GUARD_PID='' TEE_PID=''
publish_logs() {
  "${PYTHON_BIN}" "${IO}" publish --source "${LOGS}" \
    --root "${PRIMARY}/launcher/node${NODE_RANK}" --root "${MIRROR}/launcher/node${NODE_RANK}"
}
finish() {
  local code=$?
  trap - EXIT INT TERM
  set +e
  stop_job_child "${IDLE_GUARD_PID}" 'NPU idle guard' || { [[ "${code}" != 0 ]] || code=70; }
  stop_job_child "${SYNC_PID}" 'log publisher' || { [[ "${code}" != 0 ]] || code=70; }
  echo "Stage64 node=${NODE_RANK} exit=${code}"
  exec 1>&3 2>&4
  [[ -z "${TEE_PID}" ]] || wait "${TEE_PID}" 2>/dev/null
  "${PYTHON_BIN}" - "${LOGS}/exit.json" "${code}" <<'PY'
import sys,time
from pathlib import Path
from scripts.window_run_io import atomic
atomic(Path(sys.argv[1]),dict(exit_code=int(sys.argv[2]),time=time.time()))
PY
  publish_logs || { [[ "${code}" != 0 ]] || code=73; }
  exit "${code}"
}
exec 3>&1 4>&2
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
exec > >(tee -a "${LOGS}/pipeline.log") 2>&1
TEE_PID=$!
echo "Stage64 startup stage=${STAGE} node=${NODE_RANK}/${NNODES} namespace=${NAMESPACE}"
"${PYTHON_BIN}" "${IO}" publish --source "${LOGS}" \
  --root "${PRIMARY}/launcher/node${NODE_RANK}" --root "${MIRROR}/launcher/node${NODE_RANK}" --watch &
SYNC_PID=$!
if [[ "${SCENE_NPU_IDLE_GUARD:-1}" == 1 ]]; then
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "${PYTHON_BIN}" "${PROJECT}/scripts/npu_idle_guard.py" --parent-pid "$$" --devices "${NUM_NPUS}" \
    --output "${LOGS}/npu_idle_guard" --threshold "${SCENE_NPU_IDLE_THRESHOLD:-2}" \
    --idle-seconds "${SCENE_NPU_IDLE_SECONDS:-600}" --poll 30 --period 60 --burst-seconds 10 --matrix-size 2048 &
  IDLE_GUARD_PID=$!
fi
EFFECTIVE="${LOGS}/effective_config.json"
"${PYTHON_BIN}" - "${CONFIG}" "${EFFECTIVE}" "${STAGE}" <<'PY'
import json,sys
from pathlib import Path
from scripts.window_run_io import atomic
target=Path(sys.argv[2]);cfg=json.loads(Path(sys.argv[1]).read_text())
if sys.argv[3]=='smoke':
    cfg.update(sample_steps=4,train_pairs=48)
    cfg['training'].update(max_steps=2,min_steps=0,eval_every=1,save_every=1,
        monitor_cases=2,log_every=1,patience=4)
if target.exists():
    previous=json.loads(target.read_text())
    if sys.argv[3] in ('prepare','train','predict','render','summarize'):
        cfg=previous
    elif previous!=cfg:
        raise SystemExit('Config changed; choose a fresh LEARNED_NAMESPACE.')
atomic(target,cfg)
PY
coordinate() {
  local offset=$1; shift
  MASTER_PORT=$((BASE_PORT+offset)) OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
    PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}" \
    "${PYTHON_BIN}" -u "${PROJECT}/scripts/coordinated_scene_controls.py" \
    --timeout "${LEARNED_TIMEOUT_SECONDS:-172800}" --join-timeout "${EDIT_JOIN_TIMEOUT_SECONDS:-3600}" \
    --phase "stage64_${offset}" --run-id "${NAMESPACE}:${STAGE}" \
    --status-file "${LOGS}/coordinator_phase_${offset}.json" -- "$@"
}
ARGS=("${PROJECT}/train_edit_transport.py" --config "${EFFECTIVE}" --output "${OUT}" \
  --root "${PRIMARY}" --root "${MIRROR}" --data "${BANK}" \
  --data-root "${BANK_PRIMARY}" --data-root "${BANK_MIRROR}" \
  --references "${REFERENCES}" --reference-root "${REFERENCE_PRIMARY}" --reference-root "${REFERENCE_MIRROR}" \
  --rae "${EDIT_RAE}" --encoder "${STREAMVGGT_CKPT}" --model "${MODEL}")
distributed_phase() {
  local offset=$1 phase=$2
  coordinate "${offset}" "${TORCHRUN_BIN}" "--nnodes=${NNODES}" "--node_rank=${NODE_RANK}" \
    "--nproc_per_node=${NUM_NPUS}" "--master_addr=${MASTER_ADDR}" \
    "--master_port=$((BASE_PORT+offset+60))" -- "${ARGS[@]}" --phase "${phase}"
}
FINAL_CODE=0
TEACHER_CODE=0
if [[ "${STAGE}" =~ ^(all|smoke|prepare|train|predict)$ ]]; then
  coordinate 2 bash "${PROJECT}/scripts/stage_editing_teacher.sh" || TEACHER_CODE=$?
  [[ "${TEACHER_CODE}" == 0 ]] || FINAL_CODE="${TEACHER_CODE}"
fi
if [[ "${STAGE}" =~ ^(all|smoke|prepare)$ && "${TEACHER_CODE}" == 0 ]]; then
  distributed_phase 3 prepare || FINAL_CODE=$?
fi
if [[ "${STAGE}" =~ ^(all|smoke|train)$ && "${TEACHER_CODE}" == 0 ]]; then
  distributed_phase 10 train || FINAL_CODE=$?
fi
if [[ "${STAGE}" =~ ^(all|smoke|predict)$ && "${TEACHER_CODE}" == 0 ]]; then
  distributed_phase 11 predict || FINAL_CODE=$?
fi
# Preserve usable RGB/native comparisons even if preparation/training/prediction failed.
if [[ "${STAGE}" =~ ^(all|smoke|render)$ ]]; then
  distributed_phase 12 render || FINAL_CODE=$?
fi
if [[ "${STAGE}" =~ ^(all|smoke|summarize)$ ]]; then
  coordinate 20 "${PYTHON_BIN}" "${ARGS[@]}" --phase summarize || FINAL_CODE=$?
fi
echo "Stage64 finished ${MIRROR}; exit=${FINAL_CODE}"
exit "${FINAL_CODE}"
