#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/spatialvid.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"

# Optional utility. The default 24-node cache job reads train_full.csv directly;
# use this only when scheduling cache creation as multiple independent jobs.
TRAIN_INDEX_NUM_SHARDS=256
EVAL_INDEX_NUM_SHARDS=1

configure_modelarts_distributed
ensure_spatialvid_scale_splits
if [[ "${NODE_RANK}" -ne 0 ]]; then
  exit 0
fi

"${PYTHON_BIN}" "${PROJECT}/shard_metadata_csv.py" \
  --csv "${SPATIALVID_FULL_TRAIN_CSV}" \
  --output_dir "${SCALE_CSV_SHARD_ROOT}/train" \
  --num_shards "${TRAIN_INDEX_NUM_SHARDS}"

"${PYTHON_BIN}" "${PROJECT}/shard_metadata_csv.py" \
  --csv "${SPATIALVID_EVAL_CSV}" \
  --output_dir "${SCALE_CSV_SHARD_ROOT}/eval" \
  --num_shards "${EVAL_INDEX_NUM_SHARDS}"
