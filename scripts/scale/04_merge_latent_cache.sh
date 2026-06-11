#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT=${PROJECT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
CACHE_DIR=${CACHE_DIR:?Set CACHE_DIR to the completed latent cache}
INDEX_NUM_SHARDS=${INDEX_NUM_SHARDS:-256}
MAX_FAILURE_RATE=${MAX_FAILURE_RATE:-0.01}

python3 "${PROJECT}/merge_latent_cache.py" \
  --cache_dir "${CACHE_DIR}" \
  --expected_partitions "${INDEX_NUM_SHARDS}" \
  --max_failure_rate "${MAX_FAILURE_RATE}"
