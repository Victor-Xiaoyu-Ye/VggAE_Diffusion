#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"

CACHE_PARTITION_ID=0
CACHE_NUM_PARTITIONS=1

configure_modelarts_distributed

"${PYTHON_BIN}" "${PROJECT}/check_latent_cache_status.py" \
  --cache_dir "${SCALE_TRAIN_CACHE_DIR}" \
  --partition_id "${CACHE_PARTITION_ID}" \
  --num_partitions "${CACHE_NUM_PARTITIONS}" \
  --world_size "${WORLD_SIZE}"
