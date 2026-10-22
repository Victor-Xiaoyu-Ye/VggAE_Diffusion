#!/bin/bash
# Offline full-head geometry teacher audit of saved RAW/AE/GENERATED previews.
# One independent inference worker per node; no training or HCCL collectives.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"
source "${SCRIPT_DIR}/../lib/job_cleanup.sh"
export PYTHONPATH="${PROJECT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
configure_modelarts_distributed
[[ "${NNODES}" =~ ^[1-9][0-9]*$ && "${NUM_NPUS}" =~ ^[1-9][0-9]*$ &&
   "${NODE_RANK}" =~ ^[0-9]+$ && "${NODE_RANK}" -lt "${NNODES}" ]] || {
  echo 'Invalid NNODES, NUM_NPUS or NODE_RANK for independent trackmask workers.' >&2; exit 2;
}
SOURCE_NAMESPACE="${SCENE_TRACKMASK_SOURCE_NAMESPACE:-scene_street_motion_converge_v1_control}"
TEACHER_KIND="${SCENE_TRACKMASK_TEACHER_KIND:-streamvggt}"
case "${TEACHER_KIND}" in
  streamvggt) DEFAULT_NAMESPACE='scene_trackmask_stream_stage54_v1' ;;
  vggt) DEFAULT_NAMESPACE='scene_trackmask_stage54_v1' ;;
  *) echo 'SCENE_TRACKMASK_TEACHER_KIND must be streamvggt or vggt.' >&2; exit 2 ;;
esac
NAMESPACE="${SCENE_TRACKMASK_NAMESPACE:-${DEFAULT_NAMESPACE}}"
for namespace in "${SOURCE_NAMESPACE}" "${NAMESPACE}"; do
  [[ "${namespace}" =~ ^[a-zA-Z0-9_-]+$ ]] || {
    echo "Invalid scene namespace: ${namespace}" >&2; exit 2;
  }
done
[[ "${SOURCE_NAMESPACE}" != "${NAMESPACE}" ]] || {
  echo 'Source and audit namespaces must differ; source artifacts are read-only.' >&2; exit 2;
}
OUT="${SCALE_ROOT}/${NAMESPACE}"
PRIMARY="${SCALE_REMOTE_ROOT}/${NAMESPACE}"
MIRROR="${SCALE_MIRROR_ROOT}/${NAMESPACE}"
LOGS="${OUT}/launcher/node${NODE_RANK}"
mkdir -p "${LOGS}" "${LOCAL_CACHE_ROOT}/tmp"
export TMPDIR="${LOCAL_CACHE_ROOT}/tmp"
IO="${PROJECT}/scripts/window_run_io.py"
SYNC_PID=""
IDLE_GUARD_PID=""
TEE_PID=""

write_exit() {
  "${PYTHON_BIN}" - "${LOGS}" "$1" "$2" <<'PY'
import sys, time
from pathlib import Path
from scripts.window_run_io import atomic
atomic(Path(sys.argv[1]) / 'exit.json', dict(
    exit_code=int(sys.argv[2]), publication_failed=sys.argv[3] == 'failed',
    publication_receipt='publication_status.json', time=time.time()))
PY
}
publish_logs() {
  "${PYTHON_BIN}" "${IO}" publish --source "${LOGS}" \
    --root "${PRIMARY}/launcher/node${NODE_RANK}" \
    --root "${MIRROR}/launcher/node${NODE_RANK}"
}
finish() {
  local code=$? publication_code
  trap - EXIT INT TERM
  set +e
  echo "[launcher cleanup] node=${NODE_RANK}, task_exit_code=${code}"
  stop_job_child "${IDLE_GUARD_PID}" 'NPU idle guard' || { [[ "${code}" != 0 ]] || code=70; }
  stop_job_child "${SYNC_PID}" 'log publisher' || { [[ "${code}" != 0 ]] || code=70; }
  echo "Scene trackmask launcher finished: node=${NODE_RANK}, exit_code=${code}"
  exec 1>&3 2>&4
  [[ -z "${TEE_PID}" ]] || wait "${TEE_PID}" 2>/dev/null
  write_exit "${code}" pending
  publication_code=$?
  if [[ "${publication_code}" == 0 ]]; then
    publish_logs
    publication_code=$?
  fi
  if [[ "${publication_code}" != 0 ]]; then
    echo "Final launcher publication failed on node ${NODE_RANK} (status ${publication_code})." >&2
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
  --root "${PRIMARY}/launcher/node${NODE_RANK}" \
  --root "${MIRROR}/launcher/node${NODE_RANK}" --watch &
SYNC_PID=$!
# Starts before checkpoint staging and covers all allocated devices. Guard
# pulses are infrastructure work and must not count toward DI_throughput.
if [[ "${SCENE_NPU_IDLE_GUARD:-1}" == 1 ]]; then
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "${PYTHON_BIN}" "${PROJECT}/scripts/npu_idle_guard.py" \
    --parent-pid "$$" --devices "${NUM_NPUS}" --output "${LOGS}/npu_idle_guard" \
    --threshold "${SCENE_NPU_IDLE_THRESHOLD:-2}" \
    --idle-seconds "${SCENE_NPU_IDLE_SECONDS:-600}" \
    --poll "${SCENE_NPU_IDLE_POLL:-30}" --period "${SCENE_NPU_IDLE_PERIOD:-60}" \
    --burst-seconds "${SCENE_NPU_IDLE_BURST:-10}" \
    --matrix-size "${SCENE_NPU_IDLE_MATRIX:-2048}" &
  IDLE_GUARD_PID=$!
  echo "NPU idle guard started: pid=${IDLE_GUARD_PID}; logs=${LOGS}/npu_idle_guard"
fi

TEACHER_ARGS=(--teacher-kind "${TEACHER_KIND}")
if [[ "${TEACHER_KIND}" == streamvggt ]]; then
  # Reuse the production checkpoint, restoring its trained task heads in the
  # separate audit adapter. The Python loader checks full-head coverage: an
  # aggregator-only checkpoint must never silently use random task heads.
  TEACHER_CHECKPOINT="${STREAMVGGT_CKPT}"
  [[ -z "${SCENE_TRACKMASK_TEACHER_CODE:-}" ]] || \
    TEACHER_ARGS+=(--teacher-code "${SCENE_TRACKMASK_TEACHER_CODE}")
  for source in "${STREAMVGGT_URL:-}" "${STREAMVGGT_MIRROR_URL:-}"; do
    [[ "${source}" != http://* && "${source}" != https://* ]] || {
      echo 'StreamVGGT sources must be existing local/OBS/S3 assets; this backend never downloads public weights.' >&2
      exit 2
    }
  done
  if [[ "${VGGT_TRACKMASK_DOWNLOAD_OFFICIAL:-0}" != 0 ]]; then
    echo 'Ignoring VGGT_TRACKMASK_DOWNLOAD_OFFICIAL: StreamVGGT reuses the existing checkpoint and never invokes the public downloader.'
  fi
  ensure_local_checkpoint "${TEACHER_CHECKPOINT}" "${STREAMVGGT_URL:-}" \
    'full-head StreamVGGT tracking teacher' "${STREAMVGGT_MIRROR_URL:-}"
else
  TEACHER_CODE="${VGGT_TRACKMASK_CODE:-${PROJECT}/third_party/vggt_trackmask}"
  TEACHER_CHECKPOINT="${VGGT_TRACKMASK_CHECKPOINT:-${VGGAE_REF_ROOT}/VGGT-1B/model.safetensors}"
  [[ -f "${TEACHER_CODE}/vggt/models/vggt.py" && -f "${TEACHER_CODE}/vggt/heads/track_head.py" ]] || {
    echo "Missing full-head VGGT code at ${TEACHER_CODE}; set VGGT_TRACKMASK_CODE." >&2
    exit 2
  }
  TEACHER_ARGS+=(--teacher-code "${TEACHER_CODE}")
  # Original VGGT remains an explicit alternative. The cluster is offline;
  # the opt-in public downloader is only for separately connected machines.
  DOWNLOAD_OFFICIAL="${VGGT_TRACKMASK_DOWNLOAD_OFFICIAL:-0}"
  [[ "${DOWNLOAD_OFFICIAL}" == 0 || "${DOWNLOAD_OFFICIAL}" == 1 ]] || {
    echo 'VGGT_TRACKMASK_DOWNLOAD_OFFICIAL must be 0 or 1.' >&2; exit 2;
  }
  if [[ "${DOWNLOAD_OFFICIAL}" == 1 ]]; then
    [[ "${TEACHER_CHECKPOINT}" == *.safetensors ]] || {
      echo 'Pinned official download requires a .safetensors destination.' >&2; exit 2;
    }
    echo 'Official VGGT preparation opted in: up to 5,026,367,224 network bytes per node when not already verified.'
    "${PYTHON_BIN}" "${PROJECT}/scripts/prepare_trackmask_teacher.py" \
      --destination "${TEACHER_CHECKPOINT}"
  else
    for source in "${VGGT_TRACKMASK_CHECKPOINT_URL:-}" "${VGGT_TRACKMASK_CHECKPOINT_MIRROR_URL:-}"; do
      [[ "${source}" != http://* && "${source}" != https://* ]] || {
        echo 'Use the official opt-in downloader or stage the checkpoint first; source URLs accept local/OBS/S3 paths.' >&2
        exit 2
      }
    done
    ensure_local_checkpoint "${TEACHER_CHECKPOINT}" "${VGGT_TRACKMASK_CHECKPOINT_URL:-}" \
      'full-head VGGT tracking teacher' "${VGGT_TRACKMASK_CHECKPOINT_MIRROR_URL:-}"
  fi
fi

CASES="${SCENE_TRACKMASK_CASES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63}"
echo "Scene trackmask audit: teacher=${TEACHER_KIND}, source=${SOURCE_NAMESPACE}, output=${NAMESPACE}"
echo "Cluster: nodes=${NNODES}, worker_node=${NODE_RANK}, one teacher worker per node; allocated NPUs=${NUM_NPUS}"
echo 'Input is a decoded lossy preview; ROI metrics are distinct from original training metrics. No model training.'
echo 'Worker DI_throughput measures useful teacher inference; idle-guard work is excluded.'
"${PYTHON_BIN}" "${PROJECT}/audit_scene_trackmask.py" \
  --source-local "${SCENE_TRACKMASK_SOURCE_LOCAL:-${SCALE_ROOT}/${SOURCE_NAMESPACE}}" \
  --source-root "${SCENE_TRACKMASK_SOURCE_PRIMARY:-${SCALE_REMOTE_ROOT}/${SOURCE_NAMESPACE}}" \
  --source-root "${SCENE_TRACKMASK_SOURCE_MIRROR:-${SCALE_MIRROR_ROOT}/${SOURCE_NAMESPACE}}" \
  --output "${OUT}" --root "${PRIMARY}" --root "${MIRROR}" \
  "${TEACHER_ARGS[@]}" --teacher-checkpoint "${TEACHER_CHECKPOINT}" \
  --device "${SCENE_TRACKMASK_DEVICE:-npu:0}" \
  --precision "${SCENE_TRACKMASK_PRECISION:-bfloat16}" \
  --query-stride "${SCENE_TRACKMASK_QUERY_STRIDE:-21}" --allocated-devices "${NUM_NPUS}" \
  --node-rank "${NODE_RANK}" --nodes "${NNODES}" --cases "${CASES}" \
  --seed "${SCENE_TRACKMASK_SEED:-101}" --step "${SCENE_TRACKMASK_STEP:-6000}" \
  --weights "${SCENE_TRACKMASK_WEIGHTS:-ema}"
echo "Scene trackmask worker completed: node=${NODE_RANK}; ${MIRROR}/"
