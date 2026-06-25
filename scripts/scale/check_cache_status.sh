#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"

CACHE_SPLIT="${1:-train}"
CACHE_PARTITION_ID=0
CACHE_NUM_PARTITIONS=1

configure_modelarts_distributed

case "${CACHE_SPLIT}" in
  train)
    CACHE_DIR="${SCALE_TRAIN_CACHE_DIR}"
    ;;
  eval)
    CACHE_DIR="${SCALE_EVAL_CACHE_DIR}"
    ;;
  *)
    echo "Usage: bash scripts/scale/check_cache_status.sh [train|eval]" >&2
    exit 2
    ;;
esac

"${PYTHON_BIN}" "${PROJECT}/check_latent_cache_status.py" \
  --cache_dir "${CACHE_DIR}" \
  --partition_id "${CACHE_PARTITION_ID}" \
  --num_partitions "${CACHE_NUM_PARTITIONS}" \
  --world_size "${WORLD_SIZE}"
