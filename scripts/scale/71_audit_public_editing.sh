#!/bin/bash
# One 3x8 job: existing trained systems, public gray inputs, evaluation only.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"
source "${SCRIPT_DIR}/../lib/job_cleanup.sh"
export PYTHONPATH="${PROJECT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MASTER_PORT="${MASTER_PORT:-31500}" HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-7200}"
configure_modelarts_distributed
STAGE="${PUBLIC_STAGE:-all}"
[[ "${STAGE}" =~ ^(all|smoke|resume|evaluate|summarize|audit|dry)$ ]] || exit 2
[[ "${NNODES}" == 3 && "${NUM_NPUS}" == 8 && "${NODE_RANK}" =~ ^[0-2]$ ]] || {
  echo 'Stage71 requires three nodes x eight NPUs.'; exit 2;
}
[[ "${MASTER_PORT}" =~ ^[0-9]+$ && "${MASTER_PORT}" -ge 1024 && "${MASTER_PORT}" -le 65000 ]] || exit 2
BASE_PORT="${MASTER_PORT}"
NAMESPACE="${PUBLIC_NAMESPACE:-editing_public_colorization_v1}"
[[ "${STAGE}" != smoke ]] || NAMESPACE="${NAMESPACE}_smoke"
PARENT_NAMESPACE="${PUBLIC_PARENT_NAMESPACE:-editing_paper_evidence_v1}"
STATE_NAMESPACE="${PUBLIC_STATE_NAMESPACE:-vace_edit_state_v1}"
BANK_NAMESPACE="${PUBLIC_BANK_NAMESPACE:-vace_edit_pilot_v1}"
TRANSPORT_NAMESPACE="${PUBLIC_TRANSPORT_NAMESPACE:-edit_transport_learned_v1}"
for name in "${NAMESPACE}" "${PARENT_NAMESPACE}" "${STATE_NAMESPACE}" "${BANK_NAMESPACE}" "${TRANSPORT_NAMESPACE}"; do
  [[ "${name}" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
done
for name in "${PARENT_NAMESPACE}" "${STATE_NAMESPACE}" "${BANK_NAMESPACE}" "${TRANSPORT_NAMESPACE}"; do
  [[ "${NAMESPACE}" != "${name}" ]] || exit 2
done
OUT="${SCALE_ROOT}/${NAMESPACE}"
PRIMARY="${SCALE_REMOTE_ROOT}/${NAMESPACE}"
MIRROR="${SCALE_MIRROR_ROOT}/${NAMESPACE}"
CONFIG="${PUBLIC_CONFIG:-${PROJECT}/configs/editing_public_colorization_v1.json}"
MODEL="${EDIT_MODEL:-${VGGAE_REF_ROOT}/Wan2.1-VACE-14B}"
export EDIT_RAE="${EDIT_RAE:-${LOCAL_CACHE_ROOT}/editing_teacher/scene_c256_best_joint.pt}"
if [[ "${STAGE}" == dry ]]; then
  echo 'Stage71 3x8: official V2V/gray, <=20 source-screened cases; Native/SFT/RGB/RAE; no training.'
  echo "Parent=${PARENT_NAMESPACE}; output=${PRIMARY}; mirror=${MIRROR}"
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
  echo "Stage71 node=${NODE_RANK} exit=${code}"
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
echo "Stage71 startup stage=${STAGE} node=${NODE_RANK}/${NNODES} namespace=${NAMESPACE}"
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
VGGAE_CPU_ONLY=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 "${PYTHON_BIN}" - \
  "${CONFIG}" "${EFFECTIVE}" "${STAGE}" "${OUT}" "${PRIMARY}" "${MIRROR}" <<'PY'
import json,sys
from pathlib import Path
from scripts.window_run_io import atomic
from utils.scene_run import ArtifactStore
cfg=json.loads(Path(sys.argv[1]).read_text())
if sys.argv[3]=='smoke':
    cfg.update(smoke=True,cases=3,sample_steps=6)
    cfg['feedback']['solver'].update(sweeps=1,steps=1)
target=Path(sys.argv[2])
if sys.argv[3] in ('resume','evaluate','summarize'):
    old=ArtifactStore(sys.argv[4],sys.argv[5:7]).json('manifest.json',True)
    cfg=old['config']  # A fresh node also restores the exact durable smoke/full contract.
if target.exists() and json.loads(target.read_text())!=cfg:
    raise SystemExit('Config differs from local run; select a fresh PUBLIC_NAMESPACE')
atomic(target,cfg)
PY
coordinate() {
  local offset=$1; shift
  MASTER_PORT=$((BASE_PORT+offset)) OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
    PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}" \
    "${PYTHON_BIN}" -u "${PROJECT}/scripts/coordinated_scene_controls.py" \
    --timeout "${PUBLIC_TIMEOUT_SECONDS:-86400}" --join-timeout "${EDIT_JOIN_TIMEOUT_SECONDS:-7200}" \
    --phase "stage71_${offset}" --run-id "${NAMESPACE}:${STAGE}" \
    --status-file "${LOGS}/coordinator_phase_${offset}.json" -- "$@"
}
ARGS=("${PROJECT}/audit_public_editing.py" --config "${EFFECTIVE}" --output "${OUT}"
  --root "${PRIMARY}" --root "${MIRROR}"
  --parent "${SCALE_ROOT}/${PARENT_NAMESPACE}" --parent-root "${SCALE_REMOTE_ROOT}/${PARENT_NAMESPACE}" --parent-root "${SCALE_MIRROR_ROOT}/${PARENT_NAMESPACE}"
  --states "${SCALE_ROOT}/${STATE_NAMESPACE}" --state-root "${SCALE_REMOTE_ROOT}/${STATE_NAMESPACE}" --state-root "${SCALE_MIRROR_ROOT}/${STATE_NAMESPACE}"
  --transport "${SCALE_ROOT}/${TRANSPORT_NAMESPACE}" --transport-root "${SCALE_REMOTE_ROOT}/${TRANSPORT_NAMESPACE}" --transport-root "${SCALE_MIRROR_ROOT}/${TRANSPORT_NAMESPACE}"
  --bank "${LOCAL_CACHE_ROOT}/cache_latents/${BANK_NAMESPACE}"
  --bank-root "${SCALE_REMOTE_ROOT}/${BANK_NAMESPACE}/cache_latents" --bank-root "${PERSISTENT_OBS_ROOT}/cache_latents/${BANK_NAMESPACE}"
  --benchmark "${PUBLIC_BENCHMARK:-${VGGAE_REF_ROOT}/VACE-Benchmark}"
  --benchmark-source "${PUBLIC_BENCHMARK_SOURCE:-${PERSISTENT_OBS_ROOT}/vggae_ref/VACE-Benchmark}"
  --rae "${EDIT_RAE}" --encoder "${STREAMVGGT_CKPT}" --model "${MODEL}")
distributed_phase() {
  local offset=$1 phase=$2; shift 2
  coordinate "${offset}" "${TORCHRUN_BIN}" "--nnodes=${NNODES}" "--node_rank=${NODE_RANK}" \
    "--nproc_per_node=${NUM_NPUS}" "--master_addr=${MASTER_ADDR}" "--master_port=$((BASE_PORT+offset+100))" \
    -- "${ARGS[@]}" --phase "${phase}" "$@"
}
cpu_phase() { VGGAE_CPU_ONLY=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 "$@"; }
FINAL_CODE=0
if [[ "${STAGE}" == audit ]]; then
  cpu_phase coordinate 1 "${PYTHON_BIN}" "${ARGS[@]}" --phase audit
  exit 0
fi
if [[ "${STAGE}" =~ ^(all|smoke|resume)$ ]]; then
  cpu_phase coordinate 1 "${PYTHON_BIN}" "${ARGS[@]}" --phase index
  cpu_phase distributed_phase 2 sources
  cpu_phase coordinate 3 "${PYTHON_BIN}" "${ARGS[@]}" --phase select
  distributed_phase 4 text || FINAL_CODE=$?
  distributed_phase 5 anchors || FINAL_CODE=$?
  cpu_phase distributed_phase 6 shared || FINAL_CODE=$?
  distributed_phase 7 cache || FINAL_CODE=$?
  cpu_phase coordinate 8 "${PYTHON_BIN}" "${ARGS[@]}" --phase weights || FINAL_CODE=$?
  # These three arms never require RAE setup to succeed.
  distributed_phase 10 render --arm native || FINAL_CODE=$?
  distributed_phase 11 render --arm sft || FINAL_CODE=$?
  distributed_phase 12 render --arm rgb_full || FINAL_CODE=$?
  coordinate 15 bash "${PROJECT}/scripts/stage_editing_teacher.sh" || FINAL_CODE=$?
  distributed_phase 16 prepare || FINAL_CODE=$?
  distributed_phase 17 render --arm rae_full || FINAL_CODE=$?
fi
if [[ "${STAGE}" =~ ^(all|smoke|resume|evaluate)$ ]]; then
  cpu_phase distributed_phase 20 evaluate || FINAL_CODE=$?
fi
cpu_phase coordinate 30 "${PYTHON_BIN}" "${ARGS[@]}" --phase summarize || FINAL_CODE=$?
echo "Stage71 finished ${MIRROR}; exit=${FINAL_CODE}"
exit "${FINAL_CODE}"
