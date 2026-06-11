#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT=${PROJECT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
CSV=${CSV:?Set CSV to the full training metadata CSV}
CSV_SHARD_DIR=${CSV_SHARD_DIR:?Set CSV_SHARD_DIR for array-job metadata}
INDEX_NUM_SHARDS=${INDEX_NUM_SHARDS:-256}

python3 "${PROJECT}/shard_metadata_csv.py" \
  --csv "${CSV}" \
  --output_dir "${CSV_SHARD_DIR}" \
  --num_shards "${INDEX_NUM_SHARDS}"
