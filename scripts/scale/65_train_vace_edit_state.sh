#!/bin/bash
# Train VACE state conditioning with matched RGB and RAE arms; retain native outputs.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"
source "${SCRIPT_DIR}/../lib/job_cleanup.sh"
export PYTHONPATH="${PROJECT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MASTER_PORT="${MASTER_PORT:-30600}" HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-7200}"
configure_modelarts_distributed
STAGE="${STATE_STAGE:-all}"
[[ "${STAGE}" =~ ^(all|smoke|states|cache|train|render|summarize|dry)$ ]] || exit 2
[[ "${NNODES}" == 3 && "${NUM_NPUS}" == 8 && "${NODE_RANK}" =~ ^[0-2]$ ]] || {
  echo 'Stage65 uses three nodes x eight NPUs.'; exit 2;
}
[[ "${MASTER_PORT}" =~ ^[0-9]+$ && "${MASTER_PORT}" -ge 1024 && "${MASTER_PORT}" -le 65400 ]] || exit 2
BASE_PORT="${MASTER_PORT}"
NAMESPACE="${STATE_NAMESPACE:-vace_edit_state_v1}"
[[ "${STAGE}" != smoke ]] || NAMESPACE="${NAMESPACE}_smoke"
[[ "${NAMESPACE}" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
CONFIG="${STATE_CONFIG:-${PROJECT}/configs/vace_edit_state_v1.json}"
CONFIG_EXPLICIT="${STATE_CONFIG+x}"
BANK_NAMESPACE="${STATE_BANK_NAMESPACE:-vace_edit_pilot_v1}"
TRANSPORT_NAMESPACE="${STATE_TRANSPORT_NAMESPACE:-edit_transport_learned_v1}"
REFERENCE_NAMESPACE="${STATE_REFERENCE_NAMESPACE:-edit_reference_video_v1}"
[[ "${BANK_NAMESPACE}" =~ ^[a-zA-Z0-9_-]+$ && "${TRANSPORT_NAMESPACE}" =~ ^[a-zA-Z0-9_-]+$ && "${REFERENCE_NAMESPACE}" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
BANK="${STATE_BANK_LOCAL:-${LOCAL_CACHE_ROOT}/cache_latents/${BANK_NAMESPACE}}"
BANK_PRIMARY="${STATE_BANK_PRIMARY:-${SCALE_REMOTE_ROOT}/${BANK_NAMESPACE}/cache_latents}"
BANK_MIRROR="${STATE_BANK_MIRROR:-${PERSISTENT_OBS_ROOT}/cache_latents/${BANK_NAMESPACE}}"
TRANSPORT="${STATE_TRANSPORT_LOCAL:-${SCALE_ROOT}/${TRANSPORT_NAMESPACE}}"
TRANSPORT_PRIMARY="${STATE_TRANSPORT_PRIMARY:-${SCALE_REMOTE_ROOT}/${TRANSPORT_NAMESPACE}}"
TRANSPORT_MIRROR="${STATE_TRANSPORT_MIRROR:-${SCALE_MIRROR_ROOT}/${TRANSPORT_NAMESPACE}}"
REFERENCES="${STATE_REFERENCE_LOCAL:-${SCALE_ROOT}/${REFERENCE_NAMESPACE}}"
REFERENCE_PRIMARY="${STATE_REFERENCE_PRIMARY:-${SCALE_REMOTE_ROOT}/${REFERENCE_NAMESPACE}}"
REFERENCE_MIRROR="${STATE_REFERENCE_MIRROR:-${SCALE_MIRROR_ROOT}/${REFERENCE_NAMESPACE}}"
OUT="${SCALE_ROOT}/${NAMESPACE}"
PRIMARY="${SCALE_REMOTE_ROOT}/${NAMESPACE}"
MIRROR="${SCALE_MIRROR_ROOT}/${NAMESPACE}"
MODEL="${EDIT_MODEL:-${VGGAE_REF_ROOT}/Wan2.1-VACE-14B}"
export EDIT_RAE="${EDIT_RAE:-${LOCAL_CACHE_ROOT}/editing_teacher/scene_c256_best_joint.pt}"
if [[ "${STAGE}" == dry ]]; then
  echo "Stage65 ${NAMESPACE}: read-only bank=${BANK_NAMESPACE}; transport=${TRANSPORT_NAMESPACE}; anchors=${REFERENCE_NAMESPACE}."
  echo 'Shared source-flow reference -> trainable RGB/RAE state bridge in VACE -> Native / RGB-conditioned / RAE-conditioned video.'
  echo 'Frozen Stream/RAE and native base weights; train LoRA plus bridge. Existing staging, no internet installation or new weights.'
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
  echo "Stage65 node=${NODE_RANK} exit=${code}"
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
echo "Stage65 startup stage=${STAGE} node=${NODE_RANK}/${NNODES} namespace=${NAMESPACE}"
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
    cfg.update(sample_steps=4,train_pairs=48)
    cfg['training'].update(max_steps=2,min_steps=0,eval_every=1,save_every=1,
        monitor_cases=2,log_every=1,patience=4,preview_every=0)
previous=json.loads(target.read_text()) if target.exists() else None
if sys.argv[3] in ('states','cache','train','render','summarize'):
    from utils.scene_run import ArtifactStore
    manifest=ArtifactStore(sys.argv[4],sys.argv[5:7]).json('manifest.json')
    recorded=manifest['config'] if manifest is not None else previous
    if manifest is not None and previous is not None and previous!=recorded:
        raise SystemExit('Local effective config differs from saved manifest; use the matching STATE_NAMESPACE/config.')
    if recorded is not None:
        if sys.argv[7] and cfg!=recorded:
            raise SystemExit('Explicit STATE_CONFIG differs from saved run config; choose a fresh STATE_NAMESPACE.')
        cfg=recorded
elif previous is not None and previous!=cfg:
    raise SystemExit('Config changed; choose a fresh STATE_NAMESPACE.')
atomic(target,cfg)
PY
coordinate() {
  local offset=$1; shift
  MASTER_PORT=$((BASE_PORT+offset)) OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
    PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}" \
    "${PYTHON_BIN}" -u "${PROJECT}/scripts/coordinated_scene_controls.py" \
    --timeout "${STATE_TIMEOUT_SECONDS:-172800}" --join-timeout "${EDIT_JOIN_TIMEOUT_SECONDS:-3600}" \
    --phase "stage65_${offset}" --run-id "${NAMESPACE}:${STAGE}" \
    --status-file "${LOGS}/coordinator_phase_${offset}.json" -- "$@"
}
ARGS=("${PROJECT}/train_vace_edit_state.py" --config "${EFFECTIVE}" --output "${OUT}" \
  --root "${PRIMARY}" --root "${MIRROR}" --data "${BANK}" \
  --data-root "${BANK_PRIMARY}" --data-root "${BANK_MIRROR}" \
  --transport "${TRANSPORT}" --transport-root "${TRANSPORT_PRIMARY}" --transport-root "${TRANSPORT_MIRROR}" \
  --references "${REFERENCES}" --reference-root "${REFERENCE_PRIMARY}" --reference-root "${REFERENCE_MIRROR}" \
  --rae "${EDIT_RAE}" --encoder "${STREAMVGGT_CKPT}" --model "${MODEL}")
distributed_phase() {
  local offset=$1 phase=$2; shift 2
  coordinate "${offset}" "${TORCHRUN_BIN}" "--nnodes=${NNODES}" "--node_rank=${NODE_RANK}" \
    "--nproc_per_node=${NUM_NPUS}" "--master_addr=${MASTER_ADDR}" \
    "--master_port=$((BASE_PORT+offset+60))" -- "${ARGS[@]}" --phase "${phase}" "$@"
}
FINAL_CODE=0
TEACHER_CODE=0
read -r -a TRAIN_ARMS <<< "${STATE_ARMS:-rgb rae}"
[[ "${#TRAIN_ARMS[@]}" -gt 0 ]] || exit 2
for arm in "${TRAIN_ARMS[@]}"; do
  [[ "${arm}" == rgb || "${arm}" == rae ]] || { echo "Invalid STATE_ARMS: ${arm}"; exit 2; }
done
if [[ "${STAGE}" =~ ^(all|smoke|states)$ ]]; then
  coordinate 2 bash "${PROJECT}/scripts/stage_editing_teacher.sh" || TEACHER_CODE=$?
  [[ "${TEACHER_CODE}" == 0 ]] || FINAL_CODE="${TEACHER_CODE}"
fi
if [[ "${STAGE}" =~ ^(all|smoke|states)$ && "${TEACHER_CODE}" == 0 ]]; then
  distributed_phase 3 states || FINAL_CODE=$?
fi
# Each phase is a fresh process: teacher/encoder, VAE and VACE14B do not coexist.
if [[ "${STAGE}" =~ ^(all|smoke|cache)$ ]]; then
  distributed_phase 4 cache || FINAL_CODE=$?
fi
if [[ "${STAGE}" =~ ^(all|smoke|train)$ ]]; then
  for arm in "${TRAIN_ARMS[@]}"; do
    offset=10; [[ "${arm}" != rae ]] || offset=11
    distributed_phase "${offset}" train --arm "${arm}" || FINAL_CODE=$?
  done
fi
# A failed arm must not prevent an independent baseline or final status report.
if [[ "${STAGE}" =~ ^(all|smoke|render)$ ]]; then
  distributed_phase 20 render --arm native || FINAL_CODE=$?
  distributed_phase 21 render --arm rgb || FINAL_CODE=$?
  distributed_phase 22 render --arm rae || FINAL_CODE=$?
fi
if [[ "${STAGE}" =~ ^(all|smoke|summarize)$ ]]; then
  coordinate 30 "${PYTHON_BIN}" "${ARGS[@]}" --phase summarize || FINAL_CODE=$?
fi
echo "Stage65 finished ${MIRROR}; exit=${FINAL_CODE}"
exit "${FINAL_CODE}"
