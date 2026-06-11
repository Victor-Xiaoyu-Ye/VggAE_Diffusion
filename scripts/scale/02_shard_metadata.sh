#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/spatialvid.sh"

# ----------------------------- editable settings -----------------------------
TRAIN_INDEX_NUM_SHARDS=256
EVAL_INDEX_NUM_SHARDS=1
# -----------------------------------------------------------------------------

ensure_spatialvid_scale_splits

python3 "${PROJECT}/shard_metadata_csv.py" \
  --csv "${SPATIALVID_FULL_TRAIN_CSV}" \
  --output_dir "${SCALE_CSV_SHARD_ROOT}/train" \
  --num_shards "${TRAIN_INDEX_NUM_SHARDS}"

python3 "${PROJECT}/shard_metadata_csv.py" \
  --csv "${SPATIALVID_EVAL_CSV}" \
  --output_dir "${SCALE_CSV_SHARD_ROOT}/eval" \
  --num_shards "${EVAL_INDEX_NUM_SHARDS}"
