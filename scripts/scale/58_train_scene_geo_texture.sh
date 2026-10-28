#!/bin/bash
# Geometry consistency + matched future-geometry conditional generation pilot.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"
source "${SCRIPT_DIR}/../lib/job_cleanup.sh"
export PYTHONPATH="${PROJECT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MASTER_PORT="${MASTER_PORT:-29958}" HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-7200}"
configure_modelarts_distributed
[[ "${NNODES}" == 3 && "${NUM_NPUS}" == 8 && "${NODE_RANK}" =~ ^[0-2]$ ]] || {
  echo 'Stage58 requires three nodes x eight allocated NPUs.' >&2; exit 2;
}
[[ "${MASTER_PORT}" =~ ^[0-9]+$ && "${MASTER_PORT}" -ge 1024 && "${MASTER_PORT}" -le 65522 ]] || {
  echo 'MASTER_PORT must be in 1024..65522; independent phase ports use offsets through +13.' >&2; exit 2;
}
BASE_PORT="${MASTER_PORT}"
STAGE="${SCENE_GEO_STAGE:-all}"
[[ "${STAGE}" =~ ^(all|smoke|audit|preflight|train|dry)$ ]] || {
  echo 'SCENE_GEO_STAGE must be all, smoke, audit, preflight, train or dry.' >&2; exit 2;
}
BASE_NAMESPACE="${SCENE_GEO_NAMESPACE:-scene_geo_texture_pilot_v1}"
SOURCE_NAMESPACE="${SCENE_GEO_SOURCE_NAMESPACE:-scene_street_prefix4_48h_v1}"
for name in "${BASE_NAMESPACE}" "${SOURCE_NAMESPACE}"; do
  [[ "${name}" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid namespace.' >&2; exit 2; }
done
[[ "${STAGE}" != smoke ]] || BASE_NAMESPACE="${BASE_NAMESPACE}_smoke"
for suffix in '' _audit _data _baseline _oracle_geo; do
  [[ "${BASE_NAMESPACE}${suffix}" != "${SOURCE_NAMESPACE}" ]] || {
    echo 'Output namespaces must not overlap the source run.' >&2; exit 2;
  }
done
ARMS="${SCENE_GEO_ARMS:-baseline oracle_geo}"
[[ -n "${ARMS// /}" ]] || { echo 'At least one arm is required.' >&2; exit 2; }
SEEN=' '
for arm in ${ARMS}; do
  [[ "${arm}" =~ ^(baseline|oracle_geo)$ ]] || { echo "Invalid arm: ${arm}" >&2; exit 2; }
  [[ "${SEEN}" != *" ${arm} "* ]] || { echo "Duplicate arm: ${arm}" >&2; exit 2; }
  SEEN="${SEEN}${arm} "
done
CONFIG="${SCENE_GEO_CONFIG:-${PROJECT}/configs/scene_geo_texture_pilot_v1.json}"
[[ -f "${CONFIG}" ]] || { echo "Missing config: ${CONFIG}" >&2; exit 2; }
OUT="${SCALE_ROOT}/${BASE_NAMESPACE}"
PRIMARY="${SCALE_REMOTE_ROOT}/${BASE_NAMESPACE}"
MIRROR="${SCALE_MIRROR_ROOT}/${BASE_NAMESPACE}"
SOURCE_LOCAL="${SCENE_GEO_SOURCE_LOCAL:-${SCALE_ROOT}/${SOURCE_NAMESPACE}}"
SOURCE_PRIMARY="${SCENE_GEO_SOURCE_PRIMARY:-${SCALE_REMOTE_ROOT}/${SOURCE_NAMESPACE}}"
SOURCE_MIRROR="${SCENE_GEO_SOURCE_MIRROR:-${SCALE_MIRROR_ROOT}/${SOURCE_NAMESPACE}}"
DATA="${SCALE_ROOT}/${BASE_NAMESPACE}_data"
DATA_PRIMARY="${SCALE_REMOTE_ROOT}/${BASE_NAMESPACE}_data"
DATA_MIRROR="${SCALE_MIRROR_ROOT}/${BASE_NAMESPACE}_data"
AUDIT="${SCALE_ROOT}/${BASE_NAMESPACE}_audit"
AUDIT_PRIMARY="${SCALE_REMOTE_ROOT}/${BASE_NAMESPACE}_audit"
AUDIT_MIRROR="${SCALE_MIRROR_ROOT}/${BASE_NAMESPACE}_audit"
TIMEOUT="${SCENE_GEO_TIMEOUT_SECONDS:-172800}"
[[ "${TIMEOUT}" =~ ^[1-9][0-9]*$ ]] || { echo 'Phase timeout must be a positive integer.' >&2; exit 2; }
if [[ "${STAGE}" == dry ]]; then
  echo "Stage58 dry plan: config=${CONFIG}; source=${SOURCE_LOCAL}"
  echo "Audit=${AUDIT_MIRROR}; one NPU0 worker/node; no HCCL."
  echo "Shared compact data=${DATA_MIRROR}; CPU leader preparation, no HCCL."
  echo "Training arms=${ARMS}; each uses 24 ranks, two-update preflight and a fresh-process resume."
  echo "Outputs=${MIRROR}_{baseline,oracle_geo}; fixed-window matched update budget."
  echo 'Oracle future geometry is unavailable during ordinary inference and may carry appearance information.'
  echo 'Independent audit failures are reported without blocking training; no new public downloads.'
  echo 'Idle guard covers all eight cards/node throughout staging, CPU work and peer waits.'
  exit 0
fi
LOGS="${OUT}/launcher/node${NODE_RANK}"
mkdir -p "${LOGS}" "${LOCAL_CACHE_ROOT}/tmp"
export TMPDIR="${LOCAL_CACHE_ROOT}/tmp"
export MOX_VIDEO_CACHE_DIR="${LOCAL_CACHE_ROOT}/cache/scene_geo_texture_rgb"
export MOX_VIDEO_CACHE_GB="${SCENE_GEO_RGB_CACHE_GB:-12}"
IO="${PROJECT}/scripts/window_run_io.py"
SYNC_PID='' IDLE_GUARD_PID='' TEE_PID=''
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
  echo "Scene geometry pilot finished: node=${NODE_RANK}, exit_code=${code}"
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
"${PYTHON_BIN}" - "${CONFIG}" "${EFFECTIVE_CONFIG}" "${STAGE}" <<'PY'
import json,sys
from pathlib import Path
from scripts.window_run_io import atomic
cfg=json.loads(Path(sys.argv[1]).read_text(encoding='utf8'))
if sys.argv[3]=='smoke':
    cfg.update(max_steps=3,dev_cases=4,monitor_cases=4,sample_steps=4,
               save_every_steps=2,eval_every_steps=2,eval_seeds=[101],log_every=1,convergence=None,
               cases=[0,4],teacher_cases=[0,4])
    cfg['dataset']=dict(cfg['dataset'],candidate_windows=24,train_windows=24,eval_cases=4,source_shard_limit=8)
    cfg['notes']=[*cfg.get('notes',[]),'Separate three-update smoke; no quality certification.']
target=Path(sys.argv[2])
if target.exists() and json.loads(target.read_text(encoding='utf8'))!=cfg:
    raise SystemExit('Effective config changed in this namespace; choose a fresh SCENE_GEO_NAMESPACE.')
atomic(target,cfg)
PY
FINAL_CODE=0
remember_failure() {
  [[ "$1" == 0 || "${FINAL_CODE}" != 0 ]] || FINAL_CODE="$1"
  return 0
}
COMMON=(--config "${EFFECTIVE_CONFIG}" --source-local "${SOURCE_LOCAL}"
  --source-root "${SOURCE_PRIMARY}" --source-root "${SOURCE_MIRROR}")
if [[ "${STAGE}" =~ ^(all|smoke|audit)$ ]]; then
  STAGING_CODE=0
  for source in "${STREAMVGGT_URL:-}" "${STREAMVGGT_MIRROR_URL:-}"; do
    if [[ "${source}" == http://* || "${source}" == https://* ]]; then
      echo 'Ignoring public teacher download URL; use staged local/OBS weights for the audit.' >&2
      STAGING_CODE=2
    fi
  done
  if [[ "${STAGING_CODE}" == 0 ]]; then
    # The helper may exit; keep it isolated from unrelated cached training.
    (ensure_local_checkpoint "${STREAMVGGT_CKPT}" "${STREAMVGGT_URL:-}" \
      'existing StreamVGGT geometry audit teacher' "${STREAMVGGT_MIRROR_URL:-}") || STAGING_CODE=$?
  fi
  write_phase checkpoint "${STAGING_CODE}"
  remember_failure "${STAGING_CODE}"
  AUDIT_ARGS=("${COMMON[@]}" --output "${AUDIT}" --root "${AUDIT_PRIMARY}" --root "${AUDIT_MIRROR}"
    --encoder "${STREAMVGGT_CKPT}" --device npu:0 --node-rank "${NODE_RANK}" --nodes "${NNODES}")
  WORK_CODE=0
  MASTER_PORT=$((BASE_PORT+1)) OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
    PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}" \
    "${PYTHON_BIN}" -u "${PROJECT}/scripts/coordinated_scene_controls.py" --timeout "${TIMEOUT}" -- \
    "${PYTHON_BIN}" -u "${PROJECT}/audit_scene_geometry_consistency.py" "${AUDIT_ARGS[@]}" --stage work || WORK_CODE=$?
  write_phase audit_work "${WORK_CODE}"
  remember_failure "${WORK_CODE}"
  MERGE_CODE=0
  MASTER_PORT=$((BASE_PORT+2)) AUDIT_TIMEOUT_SECONDS="${TIMEOUT}" \
    "${PYTHON_BIN}" -u "${PROJECT}/scripts/coordinated_diagnostic.py" \
    "${PYTHON_BIN}" -u "${PROJECT}/audit_scene_geometry_consistency.py" "${AUDIT_ARGS[@]}" --stage merge || MERGE_CODE=$?
  write_phase audit_merge "${MERGE_CODE}"
  remember_failure "${MERGE_CODE}"
fi
if [[ "${STAGE}" != audit ]]; then
  PREP_CODE=0
  PREP_ARGS=("${COMMON[@]}" --cohort "${PROJECT}/configs/scene_cohort_v1.json"
    --selection "${PROJECT}/configs/scene_street_ids_v1.json"
    --output "${DATA}" --root "${DATA_PRIMARY}" --root "${DATA_MIRROR}" --stage prepare)
  MASTER_PORT=$((BASE_PORT+3)) AUDIT_TIMEOUT_SECONDS="${TIMEOUT}" \
    "${PYTHON_BIN}" -u "${PROJECT}/scripts/coordinated_diagnostic.py" \
    "${PYTHON_BIN}" -u "${PROJECT}/train_scene_geo_texture.py" "${PREP_ARGS[@]}" || PREP_CODE=$?
  write_phase prepare "${PREP_CODE}"
  remember_failure "${PREP_CODE}"
  if [[ "${PREP_CODE}" == 0 ]]; then
    for arm in ${ARMS}; do
      export SCENE_GEO_ARM="${arm}"
      ARM_OFFSET=0
      [[ "${arm}" != oracle_geo ]] || ARM_OFFSET=2
      TRAIN_ARGS=("${COMMON[@]}" --cohort "${PROJECT}/configs/scene_cohort_v1.json"
        --selection "${PROJECT}/configs/scene_street_ids_v1.json"
        --data-local "${DATA}" --data-root "${DATA_PRIMARY}" --data-root "${DATA_MIRROR}"
        --output "${SCALE_ROOT}/${BASE_NAMESPACE}_${arm}"
        --root "${SCALE_REMOTE_ROOT}/${BASE_NAMESPACE}_${arm}" --root "${SCALE_MIRROR_ROOT}/${BASE_NAMESPACE}_${arm}")
      ARM_CODE=0
      for phase in preflight train; do
        [[ "${STAGE}" != train || "${phase}" != preflight ]] || continue
        [[ "${STAGE}" != preflight || "${phase}" != train ]] || continue
        [[ "${ARM_CODE}" == 0 ]] || break
        PHASE_OFFSET=0
        [[ "${phase}" != train ]] || PHASE_OFFSET=1
        STARTED=$(date +%s)
        echo "Starting ${arm}/${phase}; oracle future geometry is not an ordinary inference input."
        # Coordinator ports differ from torchrun rendezvous and from every
        # other phase. All nodes learn failures before advancing to the next arm.
        MASTER_PORT=$((BASE_PORT+4+ARM_OFFSET+PHASE_OFFSET)) OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
          PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}" \
          "${PYTHON_BIN}" -u "${PROJECT}/scripts/coordinated_scene_controls.py" --timeout "${TIMEOUT}" -- \
          "${TORCHRUN_BIN}" "--nnodes=${NNODES}" "--node_rank=${NODE_RANK}" "--nproc_per_node=${NUM_NPUS}" \
          "--master_addr=${MASTER_ADDR}" "--master_port=$((BASE_PORT+10+ARM_OFFSET+PHASE_OFFSET))" \
          -- "${PROJECT}/train_scene_geo_texture.py" "${TRAIN_ARGS[@]}" --stage "${phase}" \
          --startup-seconds "$(( $(date +%s)-STARTED ))" || ARM_CODE=$?
        write_phase "${arm}_${phase}" "${ARM_CODE}"
        remember_failure "${ARM_CODE}"
      done
    done
  else
    echo 'Shared input preparation failed; training arms skipped. Independent audit results are retained.' >&2
  fi
fi
echo "Stage58 collected: stage=${STAGE}; exit=${FINAL_CODE}; outputs=${MIRROR}; arms=${ARMS}"
exit "${FINAL_CODE}"
