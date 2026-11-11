#!/bin/bash
# Three-node paper evidence: frozen models + one matched no-state SFT.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"
source "${SCRIPT_DIR}/../lib/job_cleanup.sh"
export PYTHONPATH="${PROJECT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MASTER_PORT="${MASTER_PORT:-31000}" HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-7200}"
configure_modelarts_distributed
STAGE="${PAPER_STAGE:-all}"
[[ "${STAGE}" =~ ^(all|smoke|inputs|sft|render|evaluate|summarize|repair|dry)$ ]] || exit 2
[[ "${NNODES}" == 3 && "${NUM_NPUS}" == 8 && "${NODE_RANK}" =~ ^[0-2]$ ]] || {
  echo 'Stage69 requires three nodes x eight NPUs.'; exit 2;
}
[[ "${MASTER_PORT}" =~ ^[0-9]+$ && "${MASTER_PORT}" -ge 1024 && "${MASTER_PORT}" -le 65300 ]] || exit 2
BASE_PORT="${MASTER_PORT}"
NAMESPACE="${PAPER_NAMESPACE:-editing_paper_evidence_v1}"
[[ "${STAGE}" != smoke ]] || NAMESPACE="${NAMESPACE}_smoke"
[[ "${NAMESPACE}" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
CONFIG="${PAPER_CONFIG:-${PROJECT}/configs/editing_paper_evidence_v1.json}"
CONFIG_EXPLICIT="${PAPER_CONFIG+x}"
STATE_NAMESPACE="${PAPER_STATE_NAMESPACE:-vace_edit_state_v1}"
TRANSPORT_NAMESPACE="${PAPER_TRANSPORT_NAMESPACE:-edit_transport_learned_v1}"
BANK_NAMESPACE="${PAPER_BANK_NAMESPACE:-vace_edit_pilot_v1}"
for name in "${STATE_NAMESPACE}" "${TRANSPORT_NAMESPACE}" "${BANK_NAMESPACE}"; do
  [[ "${name}" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
done
OUT="${SCALE_ROOT}/${NAMESPACE}"
PRIMARY="${SCALE_REMOTE_ROOT}/${NAMESPACE}"
MIRROR="${SCALE_MIRROR_ROOT}/${NAMESPACE}"
MODEL="${EDIT_MODEL:-${VGGAE_REF_ROOT}/Wan2.1-VACE-14B}"
export EDIT_RAE="${EDIT_RAE:-${LOCAL_CACHE_ROOT}/editing_teacher/scene_c256_best_joint.pt}"
ARMS=(native native_reference sft rgb_base rae_base rgb_full rae_full
  rae_reinject rae_no_temporal rae_no_graph rae_shuffle_graph rae_no_reliability
  rae_bridge_only rae_velocity_only rae_zero_edit rae_no_state native_reference_60)
if [[ -n "${PAPER_ARM:-}" ]]; then
  [[ " ${ARMS[*]} " == *" ${PAPER_ARM} "* && "${STAGE}" == render ]] || exit 2
  ARMS=("${PAPER_ARM}")
fi
if [[ "${STAGE}" == dry ]]; then
  echo "Stage69: 3x8; only matched reference SFT is trained; frozen Stage65 RGB/RAE."
  echo "Reserved sources <=128; subset <=32; 7 main controls + 10 ablations + 3 seed repeats."
  echo "Outputs=${PRIMARY}; mirror=${MIRROR}; namespace=${NAMESPACE}"
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
  echo "Stage69 node=${NODE_RANK} exit=${code}"
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
echo "Stage69 startup stage=${STAGE} node=${NODE_RANK}/${NNODES} namespace=${NAMESPACE}"
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
    cfg.update(smoke=True,cases=3,sample_steps=6)
    cfg['feedback']['solver'].update(sweeps=1,steps=1)
previous=json.loads(target.read_text()) if target.exists() else None
if sys.argv[3] not in ('all','smoke'):
    from utils.scene_run import ArtifactStore
    manifest=ArtifactStore(sys.argv[4],sys.argv[5:7]).json('manifest.json')
    recorded=manifest['config'] if manifest is not None else previous
    if recorded is not None:
        if sys.argv[7] and cfg!=recorded:raise SystemExit('Explicit config differs from frozen run')
        cfg=recorded
elif previous is not None and previous!=cfg:
    raise SystemExit('Config changed; choose a fresh PAPER_NAMESPACE')
atomic(target,cfg)
PY
coordinate() {
  local offset=$1; shift
  MASTER_PORT=$((BASE_PORT+offset)) OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
    PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}" \
    "${PYTHON_BIN}" -u "${PROJECT}/scripts/coordinated_scene_controls.py" \
    --timeout "${PAPER_TIMEOUT_SECONDS:-345600}" --join-timeout "${EDIT_JOIN_TIMEOUT_SECONDS:-7200}" \
    --phase "stage69_${offset}" --run-id "${NAMESPACE}:${STAGE}" \
    --status-file "${LOGS}/coordinator_phase_${offset}.json" -- "$@"
}
ARGS=("${PROJECT}/paper_editing_suite.py" --config "${EFFECTIVE}" --output "${OUT}"
  --root "${PRIMARY}" --root "${MIRROR}"
  --bank "${LOCAL_CACHE_ROOT}/cache_latents/${BANK_NAMESPACE}"
  --bank-root "${SCALE_REMOTE_ROOT}/${BANK_NAMESPACE}/cache_latents"
  --bank-root "${PERSISTENT_OBS_ROOT}/cache_latents/${BANK_NAMESPACE}"
  --states "${SCALE_ROOT}/${STATE_NAMESPACE}" --state-root "${SCALE_REMOTE_ROOT}/${STATE_NAMESPACE}" --state-root "${SCALE_MIRROR_ROOT}/${STATE_NAMESPACE}"
  --transport "${SCALE_ROOT}/${TRANSPORT_NAMESPACE}" --transport-root "${SCALE_REMOTE_ROOT}/${TRANSPORT_NAMESPACE}" --transport-root "${SCALE_MIRROR_ROOT}/${TRANSPORT_NAMESPACE}"
  --data "${EDIT_DATA_LOCAL:-${LOCAL_CACHE_ROOT}/editing/Senorita-pilot}"
  --source "${EDIT_DATA_SOURCE:-${PERSISTENT_OBS_ROOT}/dataset/Senorita-pilot}"
  --rae "${EDIT_RAE}" --encoder "${STREAMVGGT_CKPT}" --model "${MODEL}")
distributed_phase() {
  local offset=$1 phase=$2; shift 2
  coordinate "${offset}" "${TORCHRUN_BIN}" "--nnodes=${NNODES}" "--node_rank=${NODE_RANK}" \
    "--nproc_per_node=${NUM_NPUS}" "--master_addr=${MASTER_ADDR}" \
    "--master_port=$((BASE_PORT+offset+100))" -- "${ARGS[@]}" --phase "${phase}" "$@"
}
cpu_phase() {
  VGGAE_CPU_ONLY=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 "$@"
}
FINAL_CODE=0
if [[ "${STAGE}" == repair ]]; then
  # A new ModelArts job may have an empty local cache. Restore only assets/data;
  # all source selection, prepared states, SFT and completed samples stay frozen.
  cpu_phase coordinate 1 "${PYTHON_BIN}" "${ARGS[@]}" --phase repair_inputs || FINAL_CODE=$?
  coordinate 5 bash "${PROJECT}/scripts/stage_editing_teacher.sh" || FINAL_CODE=$?
fi
if [[ "${STAGE}" =~ ^(all|smoke|inputs)$ ]]; then
  cpu_phase coordinate 1 "${PYTHON_BIN}" "${ARGS[@]}" --phase index || FINAL_CODE=$?
  cpu_phase distributed_phase 2 screen || FINAL_CODE=$?
  cpu_phase distributed_phase 3 sources || FINAL_CODE=$?
  cpu_phase coordinate 4 "${PYTHON_BIN}" "${ARGS[@]}" --phase select || FINAL_CODE=$?
  coordinate 5 bash "${PROJECT}/scripts/stage_editing_teacher.sh" || FINAL_CODE=$?
  distributed_phase 6 text || FINAL_CODE=$?
  distributed_phase 7 anchors || FINAL_CODE=$?
  distributed_phase 8 prepare || FINAL_CODE=$?
  distributed_phase 9 cache || FINAL_CODE=$?
fi
if [[ "${STAGE}" =~ ^(all|smoke|sft)$ ]]; then
  distributed_phase 10 sft || FINAL_CODE=$?
fi
# Independent cached arms continue after another arm fails. No quality gate.
if [[ "${STAGE}" =~ ^(all|smoke|render|repair)$ ]]; then
  cpu_phase coordinate 11 "${PYTHON_BIN}" "${ARGS[@]}" --phase weights || FINAL_CODE=$?
  RENDER_EXTRA=()
  [[ "${STAGE}" != repair ]] || RENDER_EXTRA=(--missing-only)
  offset=20
  for arm in "${ARMS[@]}"; do
    distributed_phase "${offset}" render --arm "${arm}" "${RENDER_EXTRA[@]}" || FINAL_CODE=$?
    offset=$((offset+1))
  done
fi
if [[ "${STAGE}" =~ ^(all|smoke|evaluate|repair)$ ]]; then
  # CPU/Gloo avoids reloading Ascend for small independent evaluation models.
  cpu_phase distributed_phase 60 evaluate || FINAL_CODE=$?
fi
if [[ "${STAGE}" =~ ^(all|smoke|evaluate|summarize|repair)$ ]]; then
  cpu_phase coordinate 61 "${PYTHON_BIN}" "${ARGS[@]}" --phase summarize || FINAL_CODE=$?
fi
echo "Stage69 finished ${MIRROR}; exit=${FINAL_CODE}"
exit "${FINAL_CODE}"
