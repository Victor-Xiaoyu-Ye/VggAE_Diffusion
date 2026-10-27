#!/bin/bash
# Matched future-control / observed-reference pilot, 3 nodes x 8 Ascend NPUs.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"
source "${SCRIPT_DIR}/../lib/job_cleanup.sh"
export PYTHONPATH="${PROJECT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MASTER_PORT="${MASTER_PORT:-29956}" HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-7200}"
configure_modelarts_distributed
[[ "${NNODES}" == 3 && "${NUM_NPUS}" == 8 ]] || {
  echo 'Stage56 requires three nodes x eight NPUs.' >&2; exit 2;
}
[[ "${MASTER_PORT}" =~ ^[0-9]+$ && "${MASTER_PORT}" -ge 1024 && "${MASTER_PORT}" -le 65531 ]] || {
  echo 'MASTER_PORT must be in 1024..65531 (three preparation ports follow it).' >&2; exit 2;
}
STAGE="${SCENE_CONTROL_STAGE:-all}"
[[ "${STAGE}" =~ ^(all|smoke|prepare|preflight|train|dry)$ ]] || {
  echo 'SCENE_CONTROL_STAGE must be all, smoke, prepare, preflight, train or dry.' >&2; exit 2;
}
SOURCE_NAMESPACE="${SCENE_CONTROL_SOURCE_NAMESPACE:-scene_street_prefix4_48h_v1}"
BASE_NAMESPACE="${SCENE_CONTROL_NAMESPACE:-scene_street_control_reference_v1}"
for name in "${SOURCE_NAMESPACE}" "${BASE_NAMESPACE}"; do
  [[ "${name}" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid namespace' >&2; exit 2; }
done
[[ "${SOURCE_NAMESPACE}" != "${BASE_NAMESPACE}" ]] || { echo 'Source must remain read-only' >&2; exit 2; }
[[ "${STAGE}" != smoke ]] || BASE_NAMESPACE="${BASE_NAMESPACE}_smoke"
ARMS="${SCENE_CONTROL_ARMS:-baseline motion reference}"
[[ -n "${ARMS// /}" ]] || { echo 'At least one arm is required' >&2; exit 2; }
for arm in ${ARMS}; do
  [[ "${arm}" =~ ^(baseline|camera|motion|reference)$ ]] || { echo "Invalid arm: ${arm}" >&2; exit 2; }
done
CONFIG="${SCENE_CONTROL_CONFIG:-${PROJECT}/configs/scene_control_reference_v1.json}"
[[ -f "${CONFIG}" ]] || { echo "Missing config: ${CONFIG}" >&2; exit 2; }
OUT="${SCALE_ROOT}/${BASE_NAMESPACE}"
PRIMARY="${SCALE_REMOTE_ROOT}/${BASE_NAMESPACE}"
MIRROR="${SCALE_MIRROR_ROOT}/${BASE_NAMESPACE}"
CONTROLS="${SCALE_ROOT}/${BASE_NAMESPACE}_controls"
CONTROLS_PRIMARY="${SCALE_REMOTE_ROOT}/${BASE_NAMESPACE}_controls"
CONTROLS_MIRROR="${SCALE_MIRROR_ROOT}/${BASE_NAMESPACE}_controls"
SOURCE_LOCAL="${SCENE_CONTROL_SOURCE_LOCAL:-${SCALE_ROOT}/${SOURCE_NAMESPACE}}"
SOURCE_PRIMARY="${SCENE_CONTROL_SOURCE_PRIMARY:-${SCALE_REMOTE_ROOT}/${SOURCE_NAMESPACE}}"
SOURCE_MIRROR="${SCENE_CONTROL_SOURCE_MIRROR:-${SCALE_MIRROR_ROOT}/${SOURCE_NAMESPACE}}"
if [[ "${STAGE}" == dry ]]; then
  echo "Stage56 dry plan: config=${CONFIG}; source=${SOURCE_LOCAL}"
  echo "Preparation: CPU plan -> 24 independent raw StreamVGGT teacher workers -> merge"
  echo "Training: arms=${ARMS}; each uses two-step preflight then a fresh-process resume"
  echo "Controls=${CONTROLS_MIRROR}; output prefix=${MIRROR}; arm suffixes=${ARMS}"
  echo 'Future pseudo-camera/point trajectories are explicit controls; appearance is observed-only.'
  exit 0
fi
LOGS="${OUT}/launcher/node${NODE_RANK}"
mkdir -p "${LOGS}" "${LOCAL_CACHE_ROOT}/tmp"
export TMPDIR="${LOCAL_CACHE_ROOT}/tmp"
export MOX_VIDEO_CACHE_DIR="${LOCAL_CACHE_ROOT}/cache/scene_control_rgb"
export MOX_VIDEO_CACHE_GB="${SCENE_CONTROL_RGB_CACHE_GB:-40}"
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
  echo "Scene control launcher finished: node=${NODE_RANK}, exit_code=${code}"
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

# The exact effective JSON is reused by preparation and every arm. Smoke is a
# separate namespace and never leaves a 24-window manifest in the full pilot.
EFFECTIVE_CONFIG="${LOGS}/effective_config.json"
"${PYTHON_BIN}" - "${CONFIG}" "${EFFECTIVE_CONFIG}" "${STAGE}" <<'PY'
import json,sys
from pathlib import Path
from scripts.window_run_io import atomic
cfg=json.loads(Path(sys.argv[1]).read_text(encoding='utf8'))
if sys.argv[3]=='smoke':
    cfg.update(max_steps=3,dev_cases=4,monitor_cases=4,sample_steps=4,
               eval_every_steps=2,eval_seeds=[101],log_every=1,convergence=None)
    cfg['dataset']=dict(cfg['dataset'],train_windows=24,eval_cases=4,source_shard_limit=8)
    cfg['notes']=[*cfg.get('notes',[]),'Separate full-shape three-update smoke; not a quality experiment.']
old=Path(sys.argv[2])
if old.exists() and json.loads(old.read_text(encoding='utf8'))!=cfg:
    raise SystemExit('Effective config changed in this namespace; select a fresh SCENE_CONTROL_NAMESPACE.')
atomic(old,cfg)
PY
SETUP_TEXT=$("${PYTHON_BIN}" - "${EFFECTIVE_CONFIG}" <<'PY'
import json,sys
from pathlib import Path
c=json.loads(Path(sys.argv[1]).read_text(encoding='utf8')); t=c['teacher']; d=c['dataset']
for value in (d['train_windows'],d['eval_cases'],c['history'],c['seed'],
              t['size'],t['grid'],t['query_stride'],t['query_margin'],t['query_chunk_size'],
              d.get('source_shard_limit',0)):
    print(int(value))
PY
)
mapfile -t SETUP <<< "${SETUP_TEXT}"
[[ "${#SETUP[@]}" == 10 ]] || { echo 'Invalid effective setup fields' >&2; exit 2; }
PREP_ARGS=(--source-local "${SOURCE_LOCAL}" --source-root "${SOURCE_PRIMARY}" --source-root "${SOURCE_MIRROR}"
  --local "${CONTROLS}" --root "${CONTROLS_PRIMARY}" --root "${CONTROLS_MIRROR}"
  --cohort "${PROJECT}/configs/scene_cohort_v1.json" --selection "${PROJECT}/configs/scene_street_ids_v1.json"
  --train-windows "${SETUP[0]}" --eval-cases "${SETUP[1]}" --history "${SETUP[2]}" --seed "${SETUP[3]}"
  --size "${SETUP[4]}" --grid "${SETUP[5]}" --query-stride "${SETUP[6]}"
  --query-margin "${SETUP[7]}" --query-chunk-size "${SETUP[8]}" --workers "${WORLD_SIZE}"
  --source-shard-limit "${SETUP[9]}"
  --teacher-checkpoint "${STREAMVGGT_CKPT}" --device npu
  --precision "${SCENE_CONTROL_PRECISION:-bfloat16}" --raw-cache "${MOX_VIDEO_CACHE_DIR}")
if [[ "${STAGE}" =~ ^(all|smoke|prepare)$ ]]; then
  for source in "${STREAMVGGT_URL:-}" "${STREAMVGGT_MIRROR_URL:-}"; do
    [[ "${source}" != http://* && "${source}" != https://* ]] || {
      echo 'Use existing local/OBS StreamVGGT weights; this cluster route has no public download.' >&2; exit 2;
    }
  done
  ensure_local_checkpoint "${STREAMVGGT_CKPT}" "${STREAMVGGT_URL:-}" 'full-head StreamVGGT control teacher' "${STREAMVGGT_MIRROR_URL:-}"
  PREP_TIMEOUT="${SCENE_CONTROL_PREP_TIMEOUT_SECONDS:-172800}"
  # Each coordinator owns a distinct port; no HCCL group waits during CPU/IO.
  MASTER_PORT=$((MASTER_PORT+1)) AUDIT_TIMEOUT_SECONDS="${PREP_TIMEOUT}" \
    "${PYTHON_BIN}" -u "${PROJECT}/scripts/coordinated_diagnostic.py" \
    "${PYTHON_BIN}" -u "${PROJECT}/scripts/prepare_scene_controls.py" "${PREP_ARGS[@]}" --stage plan
  # End torchrun options before the script: Python 3.11 otherwise expands the
  # child's --local as an ambiguous abbreviation of torchrun's --local-* flags.
  MASTER_PORT=$((MASTER_PORT+2)) OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
    PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}" \
    "${PYTHON_BIN}" -u "${PROJECT}/scripts/coordinated_scene_controls.py" --timeout "${PREP_TIMEOUT}" -- \
    "${TORCHRUN_BIN}" "--nnodes=${NNODES}" "--node_rank=${NODE_RANK}" "--nproc_per_node=${NUM_NPUS}" \
    "--master_addr=${MASTER_ADDR}" "--master_port=${MASTER_PORT}" \
    -- "${PROJECT}/scripts/prepare_scene_controls.py" "${PREP_ARGS[@]}" --stage work
  MASTER_PORT=$((MASTER_PORT+3)) AUDIT_TIMEOUT_SECONDS="${PREP_TIMEOUT}" \
    "${PYTHON_BIN}" -u "${PROJECT}/scripts/coordinated_diagnostic.py" \
    "${PYTHON_BIN}" -u "${PROJECT}/scripts/prepare_scene_controls.py" "${PREP_ARGS[@]}" --stage merge
fi
if [[ "${STAGE}" != prepare ]]; then
  for arm in ${ARMS}; do
    export SCENE_CONTROL_ARM="${arm}"
    ARGS=(--config "${EFFECTIVE_CONFIG}" --cohort "${PROJECT}/configs/scene_cohort_v1.json"
      --selection "${PROJECT}/configs/scene_street_ids_v1.json" --encoder "${STREAMVGGT_CKPT}"
      --source-local "${SOURCE_LOCAL}" --source-root "${SOURCE_PRIMARY}" --source-root "${SOURCE_MIRROR}"
      --controls-local "${CONTROLS}" --controls-root "${CONTROLS_PRIMARY}" --controls-root "${CONTROLS_MIRROR}"
      --output "${SCALE_ROOT}/${BASE_NAMESPACE}_${arm}"
      --root "${SCALE_REMOTE_ROOT}/${BASE_NAMESPACE}_${arm}" --root "${SCALE_MIRROR_ROOT}/${BASE_NAMESPACE}_${arm}")
    echo "Starting arm=${arm}; explicit future pseudo-controls, observed appearance only."
    STARTED=$(date +%s)
    if [[ "${STAGE}" != train ]]; then
      STAGE_LOG_FILE="" run_torchrun "${PROJECT}/train_scene_control.py" "${ARGS[@]}" \
        --stage preflight --startup-seconds "$(( $(date +%s)-STARTED ))"
      STARTED=$(date +%s)
    fi
    if [[ "${STAGE}" != preflight ]]; then
      STAGE_LOG_FILE="" run_torchrun "${PROJECT}/train_scene_control.py" "${ARGS[@]}" \
        --stage train --startup-seconds "$(( $(date +%s)-STARTED ))"
    fi
  done
fi
echo "Stage56 complete: ${MIRROR}; controls=${CONTROLS_MIRROR}; arms=${ARMS}"
