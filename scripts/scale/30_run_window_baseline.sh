#!/bin/bash
# One ModelArts command on every node: new cache -> merge -> optional text -> train.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"
export AE_VARIANT="${AE_VARIANT:-t2v2}"
export PREDICTION="${PREDICTION:-x0}"
export WINDOW_NAMESPACE="${WINDOW_NAMESPACE:-r7_window_${AE_VARIANT}_${PREDICTION}_diag_v1}"
[[ "${WINDOW_NAMESPACE}" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Unsafe namespace' >&2; exit 2; }
export MASTER_PORT="${MASTER_PORT:-29880}"
configure_modelarts_distributed
export PYTHONPATH="${PROJECT}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ "${RESUME:-0}" == 0 && "${NODE_RANK}" == 0 ]]; then
  "${PYTHON_BIN}" "${PROJECT}/scripts/window_run_io.py" guard \
    --root "${SCALE_ROOT}/${WINDOW_NAMESPACE}" \
    --root "${SCALE_REMOTE_ROOT}/${WINDOW_NAMESPACE}" \
    --root "${SCALE_MIRROR_ROOT}/${WINDOW_NAMESPACE}"
fi
run_distributed_barrier
if [[ "${PREPARE_CACHE:-1}" == 1 && "${RESUME:-0}" == 0 ]]; then
  # One partition across the entire cluster; each rank writes disjoint tar shards.
  export CACHE_NUM_PARTITIONS=1 CACHE_PARTITION_ID=0
  MODE=train MASTER_PORT=29881 bash "${SCRIPT_DIR}/28_prepare_window_cache.sh"
  run_distributed_barrier
  MODE=eval MASTER_PORT=29882 bash "${SCRIPT_DIR}/28_prepare_window_cache.sh"
  run_distributed_barrier
  MODE=merge bash "${SCRIPT_DIR}/28_prepare_window_cache.sh"
  run_distributed_barrier
fi
if [[ "${NO_TEXT:-0}" != 1 && -z "${TEXT_DIR:-}" && "${RESUME:-0}" == 0 ]]; then
  bash "${SCRIPT_DIR}/15_precompute_wan_text_embeddings.sh"
  run_distributed_barrier
fi
MASTER_PORT=29890 bash "${SCRIPT_DIR}/29_train_window_diffusion.sh"
