#!/bin/bash
# Frozen Stage65 models, decoder-calibrated joint state feedback, matched RGB control.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"
source "${SCRIPT_DIR}/../lib/job_cleanup.sh"
export PYTHONPATH="${PROJECT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MASTER_PORT="${MASTER_PORT:-30700}" HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-7200}"
configure_modelarts_distributed
STAGE="${FEEDBACK_STAGE:-all}"
[[ "${STAGE}" =~ ^(all|smoke|prepare|render|summarize|dry)$ ]] || exit 2
[[ "${NNODES}" == 3 && "${NUM_NPUS}" == 8 && "${NODE_RANK}" =~ ^[0-2]$ ]] || {
  echo 'Stage66 uses three nodes x eight NPUs.'; exit 2;
}
[[ "${MASTER_PORT}" =~ ^[0-9]+$ && "${MASTER_PORT}" -ge 1024 && "${MASTER_PORT}" -le 65400 ]] || exit 2
BASE_PORT="${MASTER_PORT}"
NAMESPACE="${FEEDBACK_NAMESPACE:-edit_state_feedback_v1}"
[[ "${STAGE}" != smoke ]] || NAMESPACE="${NAMESPACE}_smoke"
[[ "${NAMESPACE}" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
CONFIG="${FEEDBACK_CONFIG:-${DEFAULT_FEEDBACK_CONFIG:-${PROJECT}/configs/edit_state_feedback_v1.json}}"
CONFIG_EXPLICIT="${FEEDBACK_CONFIG+x}"
STATE_NAMESPACE="${FEEDBACK_STATE_NAMESPACE:-vace_edit_state_v1}"
TRANSPORT_NAMESPACE="${FEEDBACK_TRANSPORT_NAMESPACE:-edit_transport_learned_v1}"
[[ "${STATE_NAMESPACE}" =~ ^[a-zA-Z0-9_-]+$ && "${TRANSPORT_NAMESPACE}" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
STATES="${FEEDBACK_STATE_LOCAL:-${SCALE_ROOT}/${STATE_NAMESPACE}}"
STATE_PRIMARY="${FEEDBACK_STATE_PRIMARY:-${SCALE_REMOTE_ROOT}/${STATE_NAMESPACE}}"
STATE_MIRROR="${FEEDBACK_STATE_MIRROR:-${SCALE_MIRROR_ROOT}/${STATE_NAMESPACE}}"
TRANSPORT="${FEEDBACK_TRANSPORT_LOCAL:-${SCALE_ROOT}/${TRANSPORT_NAMESPACE}}"
TRANSPORT_PRIMARY="${FEEDBACK_TRANSPORT_PRIMARY:-${SCALE_REMOTE_ROOT}/${TRANSPORT_NAMESPACE}}"
TRANSPORT_MIRROR="${FEEDBACK_TRANSPORT_MIRROR:-${SCALE_MIRROR_ROOT}/${TRANSPORT_NAMESPACE}}"
OUT="${SCALE_ROOT}/${NAMESPACE}"
PRIMARY="${SCALE_REMOTE_ROOT}/${NAMESPACE}"
MIRROR="${SCALE_MIRROR_ROOT}/${NAMESPACE}"
MODEL="${EDIT_MODEL:-${VGGAE_REF_ROOT}/Wan2.1-VACE-14B}"
export EDIT_RAE="${EDIT_RAE:-${LOCAL_CACHE_ROOT}/editing_teacher/scene_c256_best_joint.pt}"
read -r -a RENDER_ARMS <<< "${FEEDBACK_ARMS:-${DEFAULT_FEEDBACK_ARMS:-native rgb_base rae_base rgb_joint rae_joint}}"
[[ "${#RENDER_ARMS[@]}" -gt 0 ]] || exit 2
declare -A SEEN_ARMS=()
for arm in "${RENDER_ARMS[@]}"; do
  [[ "${arm}" =~ ^(native|rgb_base|rae_base|rgb_joint|rae_joint|rgb_sync|rae_sync)$ ]] || {
    echo "Invalid FEEDBACK_ARMS: ${arm}"; exit 2;
  }
  [[ -z "${SEEN_ARMS[${arm}]+present}" ]] || { echo "Duplicate FEEDBACK_ARMS: ${arm}"; exit 2; }
  SEEN_ARMS["${arm}"]=1
done
if [[ "${STAGE}" == dry ]]; then
  echo "Stage66 ${NAMESPACE}: read-only Stage65=${STATE_NAMESPACE}; source states=${TRANSPORT_NAMESPACE}."
  echo 'Frozen selected RGB/RAE models; joint decoder calibration reads current VACE prediction and writes real denoising state.'
  echo "Arms=${RENDER_ARMS[*]}; no optimizer training, internet installation or new model weights."
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
  echo "Stage66 node=${NODE_RANK} exit=${code}"
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
echo "Stage66 startup stage=${STAGE} node=${NODE_RANK}/${NNODES} namespace=${NAMESPACE}"
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
"${PYTHON_BIN}" - "${CONFIG}" "${EFFECTIVE}" "${STAGE}" "${OUT}" "${PRIMARY}" "${MIRROR}" "${CONFIG_EXPLICIT}" <<'PY'
import json,sys
from pathlib import Path
from scripts.window_run_io import atomic
target=Path(sys.argv[2]);cfg=json.loads(Path(sys.argv[1]).read_text())
if sys.argv[3]=='smoke':
    cfg.update(cases=3,sample_steps=6)
    cfg['solver'].update(sweeps=1,steps=1)
previous=json.loads(target.read_text()) if target.exists() else None
if sys.argv[3] in ('prepare','render','summarize'):
    from utils.scene_run import ArtifactStore
    manifest=ArtifactStore(sys.argv[4],sys.argv[5:7]).json('manifest.json')
    recorded=manifest['config'] if manifest is not None else previous
    if manifest is not None and previous is not None and previous!=recorded:
        raise SystemExit('Local effective config differs from saved manifest; use the matching FEEDBACK_NAMESPACE/config.')
    if recorded is not None:
        if sys.argv[7] and cfg!=recorded:
            raise SystemExit('Explicit FEEDBACK_CONFIG differs from saved run config; choose a fresh FEEDBACK_NAMESPACE.')
        cfg=recorded
elif previous is not None and previous!=cfg:
    raise SystemExit('Config changed; choose a fresh FEEDBACK_NAMESPACE.')
atomic(target,cfg)
PY
coordinate() {
  local offset=$1; shift
  MASTER_PORT=$((BASE_PORT+offset)) OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
    PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}" \
    "${PYTHON_BIN}" -u "${PROJECT}/scripts/coordinated_scene_controls.py" \
    --timeout "${FEEDBACK_TIMEOUT_SECONDS:-172800}" --join-timeout "${EDIT_JOIN_TIMEOUT_SECONDS:-3600}" \
    --phase "stage66_${offset}" --run-id "${NAMESPACE}:${STAGE}" \
    --status-file "${LOGS}/coordinator_phase_${offset}.json" -- "$@"
}
ARGS=("${PROJECT}/refine_vace_edit_state.py" --config "${EFFECTIVE}" --output "${OUT}" \
  --root "${PRIMARY}" --root "${MIRROR}" \
  --states "${STATES}" --state-root "${STATE_PRIMARY}" --state-root "${STATE_MIRROR}" \
  --transport "${TRANSPORT}" --transport-root "${TRANSPORT_PRIMARY}" --transport-root "${TRANSPORT_MIRROR}" \
  --rae "${EDIT_RAE}" --encoder "${STREAMVGGT_CKPT}" --model "${MODEL}")
distributed_phase() {
  local offset=$1 phase=$2; shift 2
  coordinate "${offset}" "${TORCHRUN_BIN}" "--nnodes=${NNODES}" "--node_rank=${NODE_RANK}" \
    "--nproc_per_node=${NUM_NPUS}" "--master_addr=${MASTER_ADDR}" \
    "--master_port=$((BASE_PORT+offset+60))" -- "${ARGS[@]}" --phase "${phase}" "$@"
}
FINAL_CODE=0
TEACHER_CODE=0
if [[ "${STAGE}" =~ ^(all|smoke|prepare)$ ]]; then
  coordinate 2 bash "${PROJECT}/scripts/stage_editing_teacher.sh" || TEACHER_CODE=$?
  [[ "${TEACHER_CODE}" == 0 ]] || FINAL_CODE="${TEACHER_CODE}"
fi
if [[ "${STAGE}" =~ ^(all|smoke|prepare)$ && "${TEACHER_CODE}" == 0 ]]; then
  distributed_phase 3 prepare || FINAL_CODE=$?
fi
# Preparation/one arm failure must not suppress independent cached baselines.
# Every arm uses a fresh process so large models are not duplicated across arms.
if [[ "${STAGE}" =~ ^(all|smoke|render)$ ]]; then
  for arm in "${RENDER_ARMS[@]}"; do
    case "${arm}" in
      native) offset=20 ;;
      rgb_base) offset=21 ;;
      rae_base) offset=22 ;;
      rgb_joint) offset=23 ;;
      rae_joint) offset=24 ;;
      rgb_sync) offset=25 ;;
      rae_sync) offset=26 ;;
    esac
    distributed_phase "${offset}" render --arm "${arm}" || FINAL_CODE=$?
  done
fi
if [[ "${STAGE}" =~ ^(all|smoke|summarize)$ ]]; then
  # Keep CPU media/serialization crashes diagnosable without changing the
  # generation code signature (completed cases remain resumable).
  PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 \
    coordinate 30 "${PYTHON_BIN}" "${ARGS[@]}" --phase summarize || FINAL_CODE=$?
fi
echo "Stage66 finished ${MIRROR}; exit=${FINAL_CODE}"
exit "${FINAL_CODE}"
