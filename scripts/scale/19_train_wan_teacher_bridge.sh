#!/bin/bash
# Train the diagnostic R7-to-Wan native-patch bridge from a local teacher cache.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"

configure_modelarts_distributed
require_scale_cluster
require_output_url
[[ "${NODE_RANK}" -eq 0 ]] || exit 0

TEACHER_DIR="${WAN_TEACHER_LOCAL_DIR:-${LOCAL_CACHE_ROOT}/teacher_cache/r7_wan_native_1k_v1}"
TEACHER_URL="${WAN_TEACHER_OUTPUT:-${PERSISTENT_OBS_ROOT}/teacher_cache/r7_wan_native_1k_v1}"
# Keep t1/t2 bridge runs in separate namespaces so their outputs never collide.
BRIDGE_NAMESPACE="${BRIDGE_NAMESPACE:-r7_wan_teacher_bridge_v1}"
OUTPUT_DIR="${BRIDGE_OUTPUT_DIR:-${SCALE_ROOT}/${BRIDGE_NAMESPACE}}"
REMOTE_OUTPUT_DIR="${BRIDGE_REMOTE_OUTPUT_DIR:-${SCALE_REMOTE_ROOT}/${BRIDGE_NAMESPACE}}"
mkdir -p "${TEACHER_DIR}" "${OUTPUT_DIR}"

# The cache is a small diagnostic directory. Stage its manifest and numbered
# shards explicitly because directory copy of many OBS objects can be partial.
PYTHONPATH="${PROJECT}" "${PYTHON_BIN}" - "${TEACHER_URL}" "${TEACHER_DIR}" <<'PY'
import os, sys, torch
from utils.moxing_io import copy_file, join_remote
remote, local = sys.argv[1], sys.argv[2]
os.makedirs(local, exist_ok=True)
marker = os.path.join(local, "_SUCCESS.pt")
copy_file(join_remote(remote, "_SUCCESS.pt"), marker)
meta = torch.load(marker, map_location="cpu", weights_only=False)
for index in range(int(meta["num_shards"])):
    name = f"teacher-{index:06d}.pt"
    copy_file(join_remote(remote, name), os.path.join(local, name))
PY

start_output_sync "${OUTPUT_DIR}" "${REMOTE_OUTPUT_DIR}"
trap 'stop_output_sync "${OUTPUT_DIR}" "${REMOTE_OUTPUT_DIR}"' EXIT
PYTHONPATH="${PROJECT}" "${PYTHON_BIN}" \
  "${PROJECT}/train_r7_wan_teacher_bridge.py" \
  --teacher_dir "${TEACHER_DIR}" \
  --output "${OUTPUT_DIR}/checkpoint_best.pt" \
  --steps "${STEPS:-2000}" --batch_size "${BATCH_SIZE:-4}" \
  --lr "${LEARNING_RATE:-2e-4}" --seed "${SEED:-42}"
