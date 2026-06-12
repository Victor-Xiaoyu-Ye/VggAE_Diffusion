#!/bin/bash

configure_modelarts_distributed() {
  NUM_NPUS=${NUM_NPUS:-${LOCAL_WORLD_SIZE:-8}}
  NNODES=${NNODES:-${VC_WORKER_NUM:-1}}
  NODE_RANK=${NODE_RANK:-${VC_TASK_INDEX:-0}}
  if [[ -z "${MASTER_ADDR:-}" ]]; then
    if [[ -n "${VC_WORKER_HOSTS:-}" ]]; then
      MASTER_ADDR=$(printf '%s' "${VC_WORKER_HOSTS}" | cut -d',' -f1)
    else
      MASTER_ADDR=127.0.0.1
    fi
  fi
  MASTER_PORT=${MASTER_PORT:-29600}
  WORLD_SIZE=$((NNODES * NUM_NPUS))
  export NUM_NPUS NNODES NODE_RANK
  export MASTER_ADDR MASTER_PORT WORLD_SIZE
}

require_output_url() {
  if [[ -z "${OUTPUT_URL:-}" ]]; then
    echo "OUTPUT_URL is required for persistent ModelArts outputs." >&2
    exit 1
  fi
}

remote_stage_dir() {
  local stage=$1
  printf '%s/%s' "${REMOTE_RUN_ROOT%/}" "${stage}"
}

stage_resume_checkpoint() {
  local source=$1
  local local_path=$2
  if [[ -z "${source}" ]]; then
    return
  fi
  case "${source}" in
    obs://*|s3://*)
      mkdir -p "$(dirname "${local_path}")"
      "${PYTHON_BIN}" "${PROJECT}/scripts/moxing_transfer.py" \
        "${source}" "${local_path}"
      printf '%s' "${local_path}"
      ;;
    *)
      printf '%s' "${source}"
      ;;
  esac
}

ensure_local_checkpoint() {
  local local_path=$1
  local remote_path=$2
  local label=$3
  if [[ -s "${local_path}" ]]; then
    return
  fi
  if [[ -z "${remote_path}" ]]; then
    echo "Missing ${label}: ${local_path}" >&2
    exit 1
  fi
  mkdir -p "$(dirname "${local_path}")"
  echo "Staging ${label}: ${remote_path} -> ${local_path}"
  "${PYTHON_BIN}" "${PROJECT}/scripts/moxing_transfer.py" \
    "${remote_path}" "${local_path}"
}

require_scale_cluster() {
  local expected_nodes=${EXPECTED_NNODES:-6}
  if [[ "${NNODES}" -ne "${expected_nodes}" ]]; then
    echo "[WARN] Scale job expected ${expected_nodes} nodes, got ${NNODES}." >&2
  fi
  if [[ "${NUM_NPUS}" -ne 8 ]]; then
    echo "[WARN] Scale job expected 8 NPUs per node, got ${NUM_NPUS}." >&2
  fi
}

start_output_sync() {
  local local_dir=$1
  local remote_dir=$2
  MOX_SYNC_PID=""
  if [[ "${NODE_RANK}" -ne 0 || -z "${remote_dir}" ]]; then
    return
  fi
  mkdir -p "${local_dir}"
  "${PYTHON_BIN}" "${PROJECT}/scripts/moxing_transfer.py" \
    "${local_dir}" "${remote_dir}" --directory --watch \
    --interval "${OUTPUT_SYNC_SECONDS:-300}" &
  MOX_SYNC_PID=$!
}

stop_output_sync() {
  local local_dir=$1
  local remote_dir=$2
  if [[ "${NODE_RANK}" -ne 0 || -z "${remote_dir}" ]]; then
    return
  fi
  if [[ -n "${MOX_SYNC_PID:-}" ]]; then
    kill "${MOX_SYNC_PID}" 2>/dev/null || true
    wait "${MOX_SYNC_PID}" 2>/dev/null || true
  fi
  "${PYTHON_BIN}" "${PROJECT}/scripts/moxing_transfer.py" \
    "${local_dir}" "${remote_dir}" --directory || true
}

run_torchrun() {
  PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}" \
    OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
    "${TORCHRUN_BIN}" \
      --nnodes="${NNODES}" \
      --node_rank="${NODE_RANK}" \
      --nproc_per_node="${NUM_NPUS}" \
      --master_addr="${MASTER_ADDR}" \
      --master_port="${MASTER_PORT}" \
      "$@"
}
