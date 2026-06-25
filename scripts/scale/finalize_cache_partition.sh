#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"

CACHE_SPLIT="${1:-train}"
CACHE_PARTITION_ID=0
CACHE_NUM_PARTITIONS=1
FORCE="${FORCE:-0}"

configure_modelarts_distributed
if [[ "${NODE_RANK}" -ne 0 ]]; then
  exit 0
fi

case "${CACHE_SPLIT}" in
  train)
    CACHE_DIR="${SCALE_TRAIN_CACHE_DIR}"
    ;;
  eval)
    CACHE_DIR="${SCALE_EVAL_CACHE_DIR}"
    ;;
  *)
    echo "Usage: bash scripts/scale/finalize_cache_partition.sh [train|eval]" >&2
    exit 2
    ;;
esac

ARGS=(
  --cache_dir "${CACHE_DIR}"
  --partition_id "${CACHE_PARTITION_ID}"
  --num_partitions "${CACHE_NUM_PARTITIONS}"
  --world_size "${WORLD_SIZE}"
  --latent_grid "${SCALE_LATENT_GRID}"
)
if [[ "${FORCE}" == "1" ]]; then
  ARGS+=(--force)
fi

"${PYTHON_BIN}" "${PROJECT}/finalize_latent_cache_partition.py" "${ARGS[@]}"
