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
# Refuse stale/mixed local shards before staging a representation-specific pack.
rm -f "${TEACHER_DIR}"/teacher-*.pt "${TEACHER_DIR}/_SUCCESS.pt"

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
expected_factor = int(os.environ.get("TEMPORAL_FACTOR", meta["r7_config"]["temporal_factor"]))
actual_factor = int(meta["r7_config"]["temporal_factor"])
if actual_factor != expected_factor:
    raise RuntimeError(
        f"teacher cache temporal_factor={actual_factor} != expected {expected_factor}")
shapes = meta.get("tensor_shapes") or {}
expected_r7_frames = 1 + (int(meta["seq_len"]) - 1) // actual_factor
r7_shape = shapes.get("r7") or []
if not r7_shape or int(r7_shape[0]) != expected_r7_frames:
    raise RuntimeError(
        f"teacher cache R7 shape {r7_shape} is incompatible with factor {actual_factor}")
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
  --metrics "${OUTPUT_DIR}/metrics.jsonl" \
  --steps "${STEPS:-2000}" --batch_size "${BATCH_SIZE:-4}" \
  --eval_every "${EVAL_EVERY:-100}" --eval_fraction "${EVAL_FRACTION:-0.1}" \
  --lr "${LEARNING_RATE:-2e-4}" --seed "${SEED:-42}"
