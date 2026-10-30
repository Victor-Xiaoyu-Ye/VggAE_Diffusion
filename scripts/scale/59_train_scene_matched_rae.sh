#!/bin/bash
# Matched frozen feature sources -> video RAE -> from-scratch prefix diffusion.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"
source "${SCRIPT_DIR}/../lib/job_cleanup.sh"
export PYTHONPATH="${PROJECT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

# Invoked under the per-node coordinator. Failures propagate to every node
# before any distributed phase is started; never download public weights.
if [[ "${1:-}" == --assets ]]; then
  case "${2:-}" in
    encoder)
      for source in "${STREAMVGGT_URL:-}" "${STREAMVGGT_MIRROR_URL:-}"; do
        [[ "${source}" != http://* && "${source}" != https://* ]] || {
          echo 'Use the existing local/OBS StreamVGGT checkpoint.' >&2; exit 2;
        }
      done
      ensure_local_checkpoint "${STREAMVGGT_CKPT}" "${STREAMVGGT_URL:-}" \
        'existing StreamVGGT feature encoder' "${STREAMVGGT_MIRROR_URL:-}" ;;
    perceptual)
      VGG_FILE="${TORCH_HOME:-${XDG_CACHE_HOME:-${HOME}/.cache}/torch}/hub/checkpoints/vgg16-397923af.pth"
      for source in "${SCENE_MATCHED_VGG_URL:-}" "${SCENE_MATCHED_VGG_MIRROR_URL:-}"; do
        [[ "${source}" != http://* && "${source}" != https://* ]] || {
          echo 'LPIPS VGG assets must be staged offline from local/OBS storage.' >&2; exit 2;
        }
      done
      ensure_local_checkpoint "${VGG_FILE}" "${SCENE_MATCHED_VGG_URL:-}" \
        'offline torchvision VGG16 for LPIPS (preserving TORCH_HOME)' "${SCENE_MATCHED_VGG_MIRROR_URL:-}"
      "${PYTHON_BIN}" - --check-lpips-assets <<'PY'
import importlib.util
from pathlib import Path
spec = importlib.util.find_spec('lpips')
roots = list(spec.submodule_search_locations or []) if spec else []
if not any((Path(root)/'weights/v0.1/vgg.pth').is_file() for root in roots):
    raise SystemExit('Missing bundled LPIPS weights/v0.1/vgg.pth. Stage the existing LPIPS package offline before codec training.')
print('[offline assets] VGG16 and LPIPS calibration weights found; no public download.')
PY
      ;;
    *) echo 'Unknown internal asset phase.' >&2; exit 2 ;;
  esac
  exit 0
fi

export MASTER_PORT="${MASTER_PORT:-29959}" HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-7200}"
configure_modelarts_distributed
[[ "${NNODES}" == 3 && "${NUM_NPUS}" == 8 && "${NODE_RANK}" =~ ^[0-2]$ ]] || {
  echo 'Stage59 requires three nodes x eight allocated NPUs.' >&2; exit 2;
}
[[ "${MASTER_PORT}" =~ ^[0-9]+$ && "${MASTER_PORT}" -ge 1024 && "${MASTER_PORT}" -le 65441 ]] || {
  echo 'MASTER_PORT must be in 1024..65441 (phase offsets through +94).' >&2; exit 2;
}
BASE_PORT="${MASTER_PORT}"
STAGE="${SCENE_MATCHED_STAGE:-all}"
[[ "${STAGE}" =~ ^(all|smoke|prepare|audit|train|dry)$ ]] || {
  echo 'SCENE_MATCHED_STAGE must be all, smoke, prepare, audit, train or dry.' >&2; exit 2;
}
BASE_NAMESPACE="${SCENE_MATCHED_NAMESPACE:-scene_matched_rae_v1}"
SOURCE_NAMESPACE="${SCENE_MATCHED_SOURCE_NAMESPACE:-scene_geo_texture_pilot_v1_data}"
GENERATOR_NAMESPACE="${SCENE_MATCHED_GENERATOR_NAMESPACE:-scene_street_prefix4_48h_v1}"
CODEC_NAMESPACE="${SCENE_MATCHED_CODEC_NAMESPACE:-scene_rae_c256_15day_v1}"
for name in "${BASE_NAMESPACE}" "${SOURCE_NAMESPACE}" "${GENERATOR_NAMESPACE}" "${CODEC_NAMESPACE}"; do
  [[ "${name}" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid namespace.' >&2; exit 2; }
done
[[ "${STAGE}" != smoke ]] || BASE_NAMESPACE="${BASE_NAMESPACE}_smoke"
for suffix in '' _data _dino _stream _source_audit; do
  [[ "${BASE_NAMESPACE}${suffix}" != "${SOURCE_NAMESPACE}" && "${BASE_NAMESPACE}${suffix}" != "${GENERATOR_NAMESPACE}" && "${BASE_NAMESPACE}${suffix}" != "${CODEC_NAMESPACE}" ]] || {
    echo 'Output namespaces must not overlap source input.' >&2; exit 2;
  }
done
ARMS="${SCENE_MATCHED_ARMS:-dino stream}"
[[ -n "${ARMS// /}" ]] || { echo 'At least one arm is required.' >&2; exit 2; }
SEEN=' '
for arm in ${ARMS}; do
  [[ "${arm}" =~ ^(dino|stream)$ && "${SEEN}" != *" ${arm} "* ]] || {
    echo "Invalid or duplicate arm: ${arm}" >&2; exit 2;
  }
  SEEN="${SEEN}${arm} "
done
CONFIG="${SCENE_MATCHED_CONFIG:-${PROJECT}/configs/scene_matched_rae_v1.json}"
COHORT="${SCENE_MATCHED_COHORT:-${PROJECT}/configs/scene_cohort_v1.json}"
[[ -f "${CONFIG}" ]] || { echo "Missing config: ${CONFIG}" >&2; exit 2; }
OUT="${SCALE_ROOT}/${BASE_NAMESPACE}"
PRIMARY="${SCALE_REMOTE_ROOT}/${BASE_NAMESPACE}"
MIRROR="${SCALE_MIRROR_ROOT}/${BASE_NAMESPACE}"
DATA="${SCALE_ROOT}/${BASE_NAMESPACE}_data"
DATA_PRIMARY="${SCALE_REMOTE_ROOT}/${BASE_NAMESPACE}_data"
DATA_MIRROR="${SCALE_MIRROR_ROOT}/${BASE_NAMESPACE}_data"
SOURCE_LOCAL="${SCENE_MATCHED_SOURCE_LOCAL:-${SCALE_ROOT}/${SOURCE_NAMESPACE}}"
SOURCE_PRIMARY="${SCENE_MATCHED_SOURCE_PRIMARY:-${SCALE_REMOTE_ROOT}/${SOURCE_NAMESPACE}}"
SOURCE_MIRROR="${SCENE_MATCHED_SOURCE_MIRROR:-${SCALE_MIRROR_ROOT}/${SOURCE_NAMESPACE}}"
GENERATOR_LOCAL="${SCENE_MATCHED_GENERATOR_LOCAL:-${SCALE_ROOT}/${GENERATOR_NAMESPACE}}"
GENERATOR_PRIMARY="${SCENE_MATCHED_GENERATOR_PRIMARY:-${SCALE_REMOTE_ROOT}/${GENERATOR_NAMESPACE}}"
GENERATOR_MIRROR="${SCENE_MATCHED_GENERATOR_MIRROR:-${SCALE_MIRROR_ROOT}/${GENERATOR_NAMESPACE}}"
CODEC_LOCAL="${SCENE_MATCHED_CODEC_LOCAL:-${SCALE_ROOT}/${CODEC_NAMESPACE}/production}"
CODEC_PRIMARY="${SCENE_MATCHED_CODEC_PRIMARY:-${SCALE_REMOTE_ROOT}/${CODEC_NAMESPACE}/production}"
CODEC_MIRROR="${SCENE_MATCHED_CODEC_MIRROR:-${SCALE_MIRROR_ROOT}/${CODEC_NAMESPACE}/production}"
TIMEOUT="${SCENE_MATCHED_TIMEOUT_SECONDS:-604800}"
[[ "${TIMEOUT}" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid phase timeout.' >&2; exit 2; }
if [[ "${STAGE}" == dry ]]; then
  echo "Stage59 matched pilot: config=${CONFIG}; source=${SOURCE_LOCAL}"
  echo "Prepare paired RGB/features on 24 ranks -> ${DATA_MIRROR}."
  echo "Existing source train/dev free generation -> ${MIRROR}_source_audit (original 518px)."
  echo "Arms=${ARMS}; each: codec2 -> codec resume -> latent cache -> flow2 -> flow resume."
  echo "Outputs=${MIRROR}_{dino,stream}; fixed train and development free generation."
  echo 'Same frozen checkpoint; DINOv2 versus Stream features; not a published V-RAE reproduction.'
  echo 'Independent arms preserve checkpoints/media; all eight cards/node guarded throughout staging and waits.'
  exit 0
fi

LOGS="${OUT}/launcher/node${NODE_RANK}"
mkdir -p "${LOGS}" "${LOCAL_CACHE_ROOT}/tmp"
export TMPDIR="${LOCAL_CACHE_ROOT}/tmp"
export MOX_VIDEO_CACHE_DIR="${LOCAL_CACHE_ROOT}/cache/scene_matched_rgb"
export MOX_VIDEO_CACHE_GB="${SCENE_MATCHED_RGB_CACHE_GB:-12}"
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
  stop_job_child "${IDLE_GUARD_PID}" 'NPU idle guard' || { [[ "${code}" != 0 ]] || code=70; }
  stop_job_child "${SYNC_PID}" 'log publisher' || { [[ "${code}" != 0 ]] || code=70; }
  echo "Scene matched pilot finished: node=${NODE_RANK}, exit_code=${code}"
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
    cfg.update(train_windows=24,dev_cases=4,train_eval_cases=4,monitor_cases=4,
               sample_steps=4,eval_seeds=[101],save_every_steps=2,eval_every_steps=2,log_every=1)
    cfg['codec']=dict(cfg['codec'],steps=3,accum=1)
    cfg['flow_training']=dict(cfg['flow_training'],steps=3,accum=1)
    cfg['notes']=[*cfg.get('notes',[]),'Separate three-update smoke with new-process checkpoint recovery; no quality claim.']
target=Path(sys.argv[2])
if target.exists() and json.loads(target.read_text(encoding='utf8'))!=cfg:
    raise SystemExit('Effective config changed; choose a fresh SCENE_MATCHED_NAMESPACE.')
atomic(target,cfg)
PY
FINAL_CODE=0
remember_failure() { [[ "$1" == 0 || "${FINAL_CODE}" != 0 ]] || FINAL_CODE="$1"; return 0; }
coordinate() {
  local offset=$1
  shift
  MASTER_PORT=$((BASE_PORT+offset)) OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
    PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}" \
    "${PYTHON_BIN}" -u "${PROJECT}/scripts/coordinated_scene_controls.py" --timeout "${TIMEOUT}" -- "$@"
}
distributed_phase() {
  local offset=$1
  shift
  coordinate "${offset}" "${TORCHRUN_BIN}" "--nnodes=${NNODES}" "--node_rank=${NODE_RANK}" \
    "--nproc_per_node=${NUM_NPUS}" "--master_addr=${MASTER_ADDR}" \
    "--master_port=$((BASE_PORT+offset+60))" -- "$@"
}
PREP_CODE=0
if [[ "${STAGE}" =~ ^(all|smoke|prepare)$ ]]; then
  coordinate 1 bash "${BASH_SOURCE[0]}" --assets encoder || PREP_CODE=$?
  write_phase encoder_assets "${PREP_CODE}"
  remember_failure "${PREP_CODE}"
  if [[ "${PREP_CODE}" == 0 ]]; then
    distributed_phase 2 "${PROJECT}/prepare_scene_matched.py" --stage prepare --config "${EFFECTIVE_CONFIG}" \
      --cohort "${COHORT}" --encoder "${STREAMVGGT_CKPT}" \
      --source-local "${SOURCE_LOCAL}" --source-root "${SOURCE_PRIMARY}" --source-root "${SOURCE_MIRROR}" \
      --output "${DATA}" --root "${DATA_PRIMARY}" --root "${DATA_MIRROR}" || PREP_CODE=$?
    write_phase prepare "${PREP_CODE}"
    remember_failure "${PREP_CODE}"
  fi
fi
if [[ "${STAGE}" =~ ^(all|smoke|audit)$ && "${PREP_CODE}" == 0 ]]; then
  AUDIT_CODE=0
  distributed_phase 4 "${PROJECT}/prepare_scene_matched.py" --stage audit --config "${EFFECTIVE_CONFIG}" \
    --cohort "${COHORT}" \
    --bank-local "${DATA}" --bank-root "${DATA_PRIMARY}" --bank-root "${DATA_MIRROR}" \
    --source-local "${SOURCE_LOCAL}" --source-root "${SOURCE_PRIMARY}" --source-root "${SOURCE_MIRROR}" \
    --generator-local "${GENERATOR_LOCAL}" --generator-root "${GENERATOR_PRIMARY}" --generator-root "${GENERATOR_MIRROR}" \
    --codec-local "${CODEC_LOCAL}" --codec-root "${CODEC_PRIMARY}" --codec-root "${CODEC_MIRROR}" \
    --output "${SCALE_ROOT}/${BASE_NAMESPACE}_source_audit" \
    --root "${PRIMARY}_source_audit" --root "${MIRROR}_source_audit" || AUDIT_CODE=$?
  write_phase source_audit "${AUDIT_CODE}"
  remember_failure "${AUDIT_CODE}"
fi
if [[ "${STAGE}" =~ ^(all|smoke|train)$ && "${PREP_CODE}" == 0 ]]; then
  ASSET_CODE=0
  coordinate 3 bash "${BASH_SOURCE[0]}" --assets perceptual || ASSET_CODE=$?
  write_phase perceptual_assets "${ASSET_CODE}"
  remember_failure "${ASSET_CODE}"
  if [[ "${ASSET_CODE}" == 0 ]]; then
    for arm in ${ARMS}; do
      ARM_OFFSET=10
      [[ "${arm}" != stream ]] || ARM_OFFSET=30
      TRAIN_ARGS=(--config "${EFFECTIVE_CONFIG}" --arm "${arm}"
        --data-local "${DATA}" --data-root "${DATA_PRIMARY}" --data-root "${DATA_MIRROR}"
        --output "${SCALE_ROOT}/${BASE_NAMESPACE}_${arm}"
        --root "${SCALE_REMOTE_ROOT}/${BASE_NAMESPACE}_${arm}" --root "${SCALE_MIRROR_ROOT}/${BASE_NAMESPACE}_${arm}")
      ARM_CODE=0
      for phase in codec_preflight codec cache flow_preflight flow; do
        [[ "${ARM_CODE}" == 0 ]] || break
        case "${phase}" in
          codec_preflight) PHASE_OFFSET=0; EXTRA=(--stage codec --stop-after 2) ;;
          codec) PHASE_OFFSET=1; EXTRA=(--stage codec) ;;
          cache) PHASE_OFFSET=2; EXTRA=(--stage cache) ;;
          flow_preflight) PHASE_OFFSET=3; EXTRA=(--stage flow --stop-after 2) ;;
          flow) PHASE_OFFSET=4; EXTRA=(--stage flow) ;;
        esac
        echo "Starting ${arm}/${phase}; new process, same declared initialization/data/target protocol."
        distributed_phase "$((ARM_OFFSET+PHASE_OFFSET))" "${PROJECT}/train_scene_matched.py" \
          "${TRAIN_ARGS[@]}" "${EXTRA[@]}" || ARM_CODE=$?
        write_phase "${arm}_${phase}" "${ARM_CODE}"
        remember_failure "${ARM_CODE}"
      done
    done
  fi
elif [[ "${PREP_CODE}" != 0 ]]; then
  echo 'Shared feature preparation failed; dependent training skipped, completed artifacts retained.' >&2
fi
echo "Stage59 collected: stage=${STAGE}; exit=${FINAL_CODE}; outputs=${MIRROR}; arms=${ARMS}"
exit "${FINAL_CODE}"
