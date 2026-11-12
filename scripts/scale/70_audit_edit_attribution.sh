#!/bin/bash
# One coordinated 3x8 job. Stage69 is read-only; no training or new anchors.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"
source "${SCRIPT_DIR}/../lib/job_cleanup.sh"
export PYTHONPATH="${PROJECT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MASTER_PORT="${MASTER_PORT:-31400}" HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-7200}"
configure_modelarts_distributed
STAGE="${ATTR_STAGE:-all}"
[[ "${STAGE}" =~ ^(all|smoke|resume|evaluate|summarize|dry)$ ]] || exit 2
[[ "${NNODES}" == 3 && "${NUM_NPUS}" == 8 && "${NODE_RANK}" =~ ^[0-2]$ ]] || exit 2
[[ "${MASTER_PORT}" =~ ^[0-9]+$ && "${MASTER_PORT}" -ge 1024 && "${MASTER_PORT}" -le 65000 ]] || exit 2
BASE_PORT="${MASTER_PORT}"
NAMESPACE="${ATTR_NAMESPACE:-editing_attribution_v1}"
[[ "${STAGE}" != smoke ]] || NAMESPACE="${NAMESPACE}_smoke"
PARENT_NAMESPACE="${ATTR_PARENT_NAMESPACE:-editing_paper_evidence_v1}"
STATE_NAMESPACE="${ATTR_STATE_NAMESPACE:-vace_edit_state_v1}"
BANK_NAMESPACE="${ATTR_BANK_NAMESPACE:-vace_edit_pilot_v1}"
for name in "${NAMESPACE}" "${PARENT_NAMESPACE}" "${STATE_NAMESPACE}" "${BANK_NAMESPACE}"; do
  [[ "${name}" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
done
[[ "${NAMESPACE}" != "${PARENT_NAMESPACE}" && "${NAMESPACE}" != "${STATE_NAMESPACE}" && "${NAMESPACE}" != "${BANK_NAMESPACE}" ]] || exit 2
OUT="${SCALE_ROOT}/${NAMESPACE}"
PRIMARY="${SCALE_REMOTE_ROOT}/${NAMESPACE}"
MIRROR="${SCALE_MIRROR_ROOT}/${NAMESPACE}"
CONFIG="${ATTR_CONFIG:-${PROJECT}/configs/editing_attribution_v1.json}"
MODEL="${EDIT_MODEL:-${VGGAE_REF_ROOT}/Wan2.1-VACE-14B}"
export EDIT_RAE="${EDIT_RAE:-${LOCAL_CACHE_ROOT}/editing_teacher/scene_c256_best_joint.pt}"
if [[ "${STAGE}" == dry ]]; then
  echo "Stage70 3x8: fixed Stage69 subset; calibration attribution then same-SFT RGB/VAE/RAE operators."
  echo "Parent=${PARENT_NAMESPACE}; output=${PRIMARY}; mirror=${MIRROR}; no training."
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
  echo "Stage70 node=${NODE_RANK} exit=${code}"
  exec 1>&3 2>&4
  [[ -z "${TEE_PID}" ]] || wait "${TEE_PID}" 2>/dev/null
  publish_logs || { [[ "${code}" != 0 ]] || code=73; }
  exit "${code}"
}
exec 3>&1 4>&2
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
exec > >(tee -a "${LOGS}/pipeline.log") 2>&1
TEE_PID=$!
echo "Stage70 startup stage=${STAGE} node=${NODE_RANK}/${NNODES} namespace=${NAMESPACE}"
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
cfg=json.loads(Path(sys.argv[1]).read_text())
if sys.argv[3]=='smoke':
    cfg.update(smoke=True,sample_steps=6)
    cfg['feedback']['solver'].update(sweeps=1,steps=1)
    cfg['vae_calibration']['steps']=1
target=Path(sys.argv[2])
if sys.argv[3] in ('resume','evaluate','summarize') and target.exists():
    # Continue the already frozen local run, including a smoke namespace.
    cfg=json.loads(target.read_text())
if target.exists() and json.loads(target.read_text())!=cfg:
    raise SystemExit('Config differs from local run; select a fresh ATTR_NAMESPACE')
atomic(target,cfg)
PY
coordinate() {
  local offset=$1; shift
  MASTER_PORT=$((BASE_PORT+offset)) OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
    PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}" \
    "${PYTHON_BIN}" -u "${PROJECT}/scripts/coordinated_scene_controls.py" \
    --timeout "${ATTR_TIMEOUT_SECONDS:-172800}" --join-timeout "${EDIT_JOIN_TIMEOUT_SECONDS:-7200}" \
    --phase "stage70_${offset}" --run-id "${NAMESPACE}:${STAGE}" \
    --status-file "${LOGS}/coordinator_phase_${offset}.json" -- "$@"
}
ARGS=("${PROJECT}/audit_edit_attribution.py" --config "${EFFECTIVE}" --output "${OUT}"
  --root "${PRIMARY}" --root "${MIRROR}"
  --parent "${SCALE_ROOT}/${PARENT_NAMESPACE}" --parent-root "${SCALE_REMOTE_ROOT}/${PARENT_NAMESPACE}" --parent-root "${SCALE_MIRROR_ROOT}/${PARENT_NAMESPACE}"
  --states "${SCALE_ROOT}/${STATE_NAMESPACE}" --state-root "${SCALE_REMOTE_ROOT}/${STATE_NAMESPACE}" --state-root "${SCALE_MIRROR_ROOT}/${STATE_NAMESPACE}"
  --bank "${LOCAL_CACHE_ROOT}/cache_latents/${BANK_NAMESPACE}"
  --bank-root "${SCALE_REMOTE_ROOT}/${BANK_NAMESPACE}/cache_latents" --bank-root "${PERSISTENT_OBS_ROOT}/cache_latents/${BANK_NAMESPACE}"
  --data "${EDIT_DATA_LOCAL:-${LOCAL_CACHE_ROOT}/editing/Senorita-pilot}"
  --source "${EDIT_DATA_SOURCE:-${PERSISTENT_OBS_ROOT}/dataset/Senorita-pilot}"
  --rae "${EDIT_RAE}" --encoder "${STREAMVGGT_CKPT}" --model "${MODEL}")
distributed_phase() {
  local offset=$1 phase=$2; shift 2
  coordinate "${offset}" "${TORCHRUN_BIN}" "--nnodes=${NNODES}" "--node_rank=${NODE_RANK}" \
    "--nproc_per_node=${NUM_NPUS}" "--master_addr=${MASTER_ADDR}" "--master_port=$((BASE_PORT+offset+100))" \
    -- "${ARGS[@]}" --phase "${phase}" "$@"
}
cpu_phase() { VGGAE_CPU_ONLY=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 "$@"; }
FINAL_CODE=0
if [[ "${STAGE}" =~ ^(all|smoke|resume)$ ]]; then
  cpu_phase coordinate 1 "${PYTHON_BIN}" "${ARGS[@]}" --phase index || FINAL_CODE=$?
  coordinate 2 bash "${PROJECT}/scripts/stage_editing_teacher.sh" || FINAL_CODE=$?
  distributed_phase 3 prepare || FINAL_CODE=$?
  # P0 runs first; VAE preparation failure cannot prevent its output.
  ARMS=(cal_on_online_on cal_on_online_off cal_off_online_on cal_off_online_off cal_off_magnitude cal_off_fixed_bridge_on cal_off_fixed_bridge_off cal_on_fixed_bridge_on cal_on_fixed_bridge_off)
  offset=10
  for arm in "${ARMS[@]}"; do
    distributed_phase "${offset}" render --arm "${arm}" || FINAL_CODE=$?
    offset=$((offset+1))
  done
  distributed_phase 25 prepare_vae || FINAL_CODE=$?
  offset=30
  for arm in op_sft op_rgb op_rgb_smooth op_vae op_rae; do
    distributed_phase "${offset}" render --arm "${arm}" || FINAL_CODE=$?
    offset=$((offset+1))
  done
fi
if [[ "${STAGE}" =~ ^(all|smoke|resume|evaluate)$ ]]; then
  cpu_phase distributed_phase 50 evaluate || FINAL_CODE=$?
fi
cpu_phase coordinate 60 "${PYTHON_BIN}" "${ARGS[@]}" --phase summarize || FINAL_CODE=$?
echo "Stage70 finished ${MIRROR}; exit=${FINAL_CODE}"
exit "${FINAL_CODE}"
