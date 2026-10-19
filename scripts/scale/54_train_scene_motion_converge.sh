#!/bin/bash
# Matched fixed-window control/candidate, three nodes x8. Convergence based.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
export SCENE_SOURCE_NAMESPACE="${SCENE_MOTION_SOURCE_NAMESPACE:-scene_street_prefix4_48h_v1}"
export SCENE_PREFIX_CONFIG="${SCENE_MOTION_CONFIG:-${PROJECT_DIR}/configs/scene_motion_converge_v1.json}"
export SCENE_PREFIX_ENTRYPOINT=train_scene_motion.py
export SCENE_PREFIX_SOURCE_SUFFIX=''
export SCENE_PREFIX_NEEDS_ENCODER=0
export MASTER_PORT="${MASTER_PORT:-29954}"
BASE_NAMESPACE="${SCENE_MOTION_NAMESPACE:-scene_street_motion_converge_v1}"
ARMS="${SCENE_MOTION_ARMS:-control motion}"
for arm in ${ARMS}; do
  [[ "${arm}" == control || "${arm}" == motion ]] || { echo "Invalid arm: ${arm}"; exit 2; }
  export SCENE_MOTION_ARM="${arm}"
  export SCENE_PREFIX_NAMESPACE="${BASE_NAMESPACE}_${arm}"
  echo "Starting ${arm}; optimization plateau stops training, visual quality remains report-only."
  bash "${SCRIPT_DIR}/53_train_scene_prefix_48h.sh"
done
