#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"

CACHE_NUM_PARTITIONS=1
MAX_FAILURE_RATE=0.01

configure_modelarts_distributed
require_output_url
if [[ "${NODE_RANK}" -ne 0 ]]; then
  exit 0
fi

LOG_DIR="${SCALE_ROOT}/cache_generation/merge"
REMOTE_LOG_DIR="${SCALE_REMOTE_ROOT}/cache_generation/merge"
start_output_sync "${LOG_DIR}" "${REMOTE_LOG_DIR}"
trap 'stop_output_sync "${LOG_DIR}" "${REMOTE_LOG_DIR}"' EXIT
exec > >(tee -a "${STAGE_LOG_FILE}") 2>&1

"${PYTHON_BIN}" "${PROJECT}/merge_latent_cache.py" \
  --cache_dir "${SCALE_TRAIN_CACHE_DIR}" \
  --expected_partitions "${CACHE_NUM_PARTITIONS}" \
  --max_failure_rate "${MAX_FAILURE_RATE}"

"${PYTHON_BIN}" "${PROJECT}/merge_latent_cache.py" \
  --cache_dir "${SCALE_EVAL_CACHE_DIR}" \
  --expected_partitions 1 \
  --max_failure_rate "${MAX_FAILURE_RATE}"
