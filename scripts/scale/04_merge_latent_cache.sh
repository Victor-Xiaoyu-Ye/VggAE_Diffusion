#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"

CACHE_NUM_PARTITIONS=1
MAX_FAILURE_RATE=0.01

configure_modelarts_distributed
if [[ "${NODE_RANK}" -ne 0 ]]; then
  exit 0
fi

"${PYTHON_BIN}" "${PROJECT}/merge_latent_cache.py" \
  --cache_dir "${SCALE_TRAIN_CACHE_DIR}" \
  --expected_partitions "${CACHE_NUM_PARTITIONS}" \
  --max_failure_rate "${MAX_FAILURE_RATE}"

"${PYTHON_BIN}" "${PROJECT}/merge_latent_cache.py" \
  --cache_dir "${SCALE_EVAL_CACHE_DIR}" \
  --expected_partitions 1 \
  --max_failure_rate "${MAX_FAILURE_RATE}"
