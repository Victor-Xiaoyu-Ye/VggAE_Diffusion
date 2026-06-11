#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"

# ----------------------------- editable settings -----------------------------
TRAIN_INDEX_NUM_SHARDS=256
MAX_FAILURE_RATE=0.01
# -----------------------------------------------------------------------------

python3 "${PROJECT}/merge_latent_cache.py" \
  --cache_dir "${SCALE_TRAIN_CACHE_DIR}" \
  --expected_partitions "${TRAIN_INDEX_NUM_SHARDS}" \
  --max_failure_rate "${MAX_FAILURE_RATE}"

python3 "${PROJECT}/merge_latent_cache.py" \
  --cache_dir "${SCALE_EVAL_CACHE_DIR}" \
  --expected_partitions 1 \
  --max_failure_rate "${MAX_FAILURE_RATE}"
