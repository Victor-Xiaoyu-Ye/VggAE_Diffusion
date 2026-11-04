#!/bin/bash
# Native VACE -> matched instruction SFT / edited-target RAE supervision.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"
source "${SCRIPT_DIR}/../lib/job_cleanup.sh"
export PYTHONPATH="${PROJECT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MASTER_PORT="${MASTER_PORT:-29960}" HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-7200}"
configure_modelarts_distributed
STAGE="${EDIT_STAGE:-all}"
[[ "${STAGE}" =~ ^(all|smoke|prepare|native|train|evaluate|sync|dry)$ ]] || { echo 'Invalid EDIT_STAGE'; exit 2; }
[[ "${NNODES}" == 3 && "${NUM_NPUS}" == 8 && "${NODE_RANK}" =~ ^[0-2]$ ]] || {
  echo 'Use three nodes x eight NPUs, with NODE_RANK=0/1/2.'; exit 2;
}
[[ "${MASTER_PORT}" =~ ^[0-9]+$ && "${MASTER_PORT}" -ge 1024 && "${MASTER_PORT}" -le 65400 ]] || exit 2
BASE_PORT="${MASTER_PORT}"
NAMESPACE="${EDIT_NAMESPACE:-vace_edit_pilot_v1}"
[[ "${STAGE}" != smoke ]] || NAMESPACE="${NAMESPACE}_smoke"
[[ "${NAMESPACE}" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid EDIT_NAMESPACE'; exit 2; }
MODEL="${EDIT_MODEL:-${VGGAE_REF_ROOT}/Wan2.1-VACE-14B}"
BENCHMARK="${EDIT_BENCHMARK:-${VGGAE_REF_ROOT}/VACE-Benchmark}"
SOURCE="${EDIT_DATA_SOURCE:-${PERSISTENT_OBS_ROOT}/dataset/Senorita-pilot}"
DATA="${EDIT_DATA_LOCAL:-${LOCAL_CACHE_ROOT}/editing/Senorita-pilot}"
OUT="${SCALE_ROOT}/${NAMESPACE}"
PRIMARY="${SCALE_REMOTE_ROOT}/${NAMESPACE}"
MIRROR="${SCALE_MIRROR_ROOT}/${NAMESPACE}"
BANK_NAMESPACE="${EDIT_BANK_NAMESPACE:-${NAMESPACE}}"
[[ "${BANK_NAMESPACE}" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
REUSE_BANK="${EDIT_REUSE_BANK:-0}"
[[ "${REUSE_BANK}" =~ ^[01]$ ]] || exit 2
[[ "${BANK_NAMESPACE}" == "${NAMESPACE}" || "${REUSE_BANK}" == 1 ]] || { echo 'Different bank namespace requires read-only EDIT_REUSE_BANK=1'; exit 2; }
BANK="${LOCAL_CACHE_ROOT}/cache_latents/${BANK_NAMESPACE}"
BANK_PRIMARY="${SCALE_REMOTE_ROOT}/${BANK_NAMESPACE}/cache_latents"
BANK_MIRROR="${PERSISTENT_OBS_ROOT}/cache_latents/${BANK_NAMESPACE}"
RAE="${EDIT_RAE:-${LOCAL_CACHE_ROOT}/editing_teacher/scene_c256_best_joint.pt}"
CONFIG="${EDIT_CONFIG:-${PROJECT}/configs/vace_edit_pilot_v1.json}"
ARMS="${EDIT_ARMS:-sft rae}"
SEEN=' '
NEEDS_TEACHER=0
for arm in ${ARMS}; do
  [[ "${arm}" =~ ^(sft|region_sft|rae|rae_balanced|rae_detached)$ && "${SEEN}" != *" ${arm} "* ]] || exit 2
  SEEN="${SEEN}${arm} "
  [[ "${arm}" != rae* ]] || NEEDS_TEACHER=1
done
if [[ "${STAGE}" == dry ]]; then
  echo "Stage60: ${MODEL}; source=${SOURCE}; benchmark=${BENCHMARK}"
  echo "24-way native cache -> native previews -> 2-step/resume -> arms=${ARMS}."
  echo "Outputs=${PRIMARY}; mirror=${MIRROR}; durable cache=${BANK_MIRROR}."
  echo "Reuse bank=${REUSE_BANK}; config=${CONFIG}; source-only controls=${EDIT_CONTROLS:-none}."
  echo 'Native VAE remains the output path. SAM2 is available but not required for the first all-white-mask instruction pilot.'
  exit 0
fi
LOGS="${OUT}/launcher/node${NODE_RANK}"
mkdir -p "${LOGS}" "${LOCAL_CACHE_ROOT}/tmp"
export TMPDIR="${LOCAL_CACHE_ROOT}/tmp"
export ASCEND_PROCESS_LOG_PATH="${LOGS}/npu"
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
  echo "Stage60 node=${NODE_RANK} exit=${code}"
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
echo "Stage60 startup: stage=${STAGE} namespace=${NAMESPACE} node=${NODE_RANK}/${NNODES} host=${HOSTNAME:-unknown} master=${MASTER_ADDR}:${BASE_PORT} VC_TASK_INDEX=${VC_TASK_INDEX:-unset}"
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
    cfg.update(train_pairs=48,dev_pairs=4,test_pairs=8,benchmark_cases=2,steps=3,
        save_every=1,eval_every=3,eval_cases=2,sample_steps=4,log_every=1,rae_warmup=0)
    if cfg.get('gradient_probe_every'): cfg['gradient_probe_every']=1
target=Path(sys.argv[2])
if target.exists() and json.loads(target.read_text())!=cfg:
    raise SystemExit('Config changed for an existing namespace. Select a new EDIT_NAMESPACE.')
atomic(target,cfg)
PY
coordinate() {
  local offset=$1; shift
  MASTER_PORT=$((BASE_PORT+offset)) OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
    PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}" \
    "${PYTHON_BIN}" -u "${PROJECT}/scripts/coordinated_scene_controls.py" \
    --timeout "${EDIT_TIMEOUT_SECONDS:-604800}" --phase "stage60_${offset}" \
    --join-timeout "${EDIT_JOIN_TIMEOUT_SECONDS:-3600}" --run-id "${NAMESPACE}:${STAGE}" \
    --status-file "${LOGS}/coordinator_phase_${offset}.json" -- "$@"
}
distributed_phase() {
  local offset=$1; shift
  coordinate "${offset}" "${TORCHRUN_BIN}" "--nnodes=${NNODES}" "--node_rank=${NODE_RANK}" \
    "--nproc_per_node=${NUM_NPUS}" "--master_addr=${MASTER_ADDR}" \
    "--master_port=$((BASE_PORT+offset+60))" -- "$@"
}
PREP=(--config "${EFFECTIVE}" --model "${MODEL}" --benchmark "${BENCHMARK}" \
  --data "${DATA}" --output "${BANK}" --root "${BANK_PRIMARY}" --root "${BANK_MIRROR}")
TRAIN=(--config "${EFFECTIVE}" --model "${MODEL}" --data "${BANK}" \
  --data-root "${BANK_PRIMARY}" --data-root "${BANK_MIRROR}")
[[ -z "${EDIT_CONTROLS:-}" ]] || TRAIN+=(--controls "${EDIT_CONTROLS}")
if [[ "${REUSE_BANK}" == 1 && "${STAGE}" != sync ]]; then
  coordinate 1 "${PYTHON_BIN}" "${PROJECT}/scripts/check_editing_bank.py" --bank "${BANK}" \
    --root "${BANK_PRIMARY}" --root "${BANK_MIRROR}"
elif [[ "${STAGE}" =~ ^(all|smoke|prepare)$ ]]; then
  coordinate 1 "${PYTHON_BIN}" "${PROJECT}/prepare_vace_edit.py" --stage index \
    --source "${SOURCE}" "${PREP[@]}"
  distributed_phase 2 "${PROJECT}/prepare_vace_edit.py" --stage cache "${PREP[@]}"
fi
FINAL_CODE=0
if [[ "${STAGE}" =~ ^(all|smoke|native)$ ]]; then
  distributed_phase 3 "${PROJECT}/train_vace_edit.py" --arm native "${TRAIN[@]}" \
    --output "${OUT}/native" --root "${PRIMARY}/native" --root "${MIRROR}/native" || FINAL_CODE=$?
fi
if [[ "${STAGE}" =~ ^(all|smoke|train|evaluate)$ ]]; then
  TEACHER_CODE=0
  if [[ "${NEEDS_TEACHER}" == 1 ]]; then
    # All nodes must agree on teacher setup failure, so ordinary SFT can still
    # finish and publish its results if the optional historical teacher is absent.
    export EDIT_RAE="${RAE}"
    coordinate 4 bash "${PROJECT}/scripts/stage_editing_teacher.sh" || TEACHER_CODE=$?
    if [[ "${TEACHER_CODE}" == 0 ]]; then
      if [[ "${REUSE_BANK}" == 1 ]]; then
        coordinate 5 "${PYTHON_BIN}" "${PROJECT}/scripts/check_editing_bank.py" --bank "${BANK}" \
          --root "${BANK_PRIMARY}" --root "${BANK_MIRROR}" --require-teacher || TEACHER_CODE=$?
      else
        distributed_phase 5 "${PROJECT}/prepare_vace_edit.py" --stage teacher "${PREP[@]}" \
          --rae "${RAE}" --encoder "${STREAMVGGT_CKPT}" || TEACHER_CODE=$?
      fi
    fi
    [[ "${TEACHER_CODE}" == 0 ]] || FINAL_CODE="${TEACHER_CODE}"
  fi
  FAILED=' '
  OFFSET=10
  # Complete all short backward/checkpoint/new-process tests before long arms.
  LIMITS=(2 0)
  EVAL_ARGS=()
  if [[ "${STAGE}" == evaluate ]]; then
    [[ -n "${EDIT_EVAL_CHECKPOINT:-}" ]] || { echo 'evaluate requires EDIT_EVAL_CHECKPOINT'; exit 2; }
    LIMITS=(0); EVAL_ARGS=(--eval-checkpoint "${EDIT_EVAL_CHECKPOINT}")
  fi
  for limit in "${LIMITS[@]}"; do
    for arm in ${ARMS}; do
      OFFSET=$((OFFSET+1))
      [[ "${FAILED}" != *" ${arm} "* ]] || continue
      [[ "${arm}" != rae* || "${TEACHER_CODE}" == 0 ]] || continue
      CODE=0
      distributed_phase "${OFFSET}" "${PROJECT}/train_vace_edit.py" "${TRAIN[@]}" \
        --arm "${arm}" --rae "${RAE}" --stop-after "${limit}" "${EVAL_ARGS[@]}" --output "${OUT}/${arm}" \
        --root "${PRIMARY}/${arm}" --root "${MIRROR}/${arm}" || CODE=$?
      if [[ "${CODE}" != 0 ]]; then FAILED="${FAILED}${arm} "; FINAL_CODE="${CODE}"; fi
    done
  done
fi
if [[ "${STAGE}" =~ ^(all|smoke|native|train|evaluate|sync)$ ]]; then
  # Run after every independent arm, so a failed training arm never prevents
  # media repair for the successful ones. Includes previews from older jobs.
  SYNC_ARMS="native ${ARMS}"
  [[ "${STAGE}" != native ]] || SYNC_ARMS=native
  coordinate 40 "${PYTHON_BIN}" "${PROJECT}/scripts/finalize_vace_edit.py" \
    --output "${OUT}" --root "${PRIMARY}" --root "${MIRROR}" --arms ${SYNC_ARMS} || FINAL_CODE=$?
fi
echo "Stage60 finished: ${MIRROR}; failed arms=${FAILED:-none}; exit=${FINAL_CODE}"
exit "${FINAL_CODE}"
