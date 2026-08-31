#!/bin/bash
# R7 quality redesign diagnostic chain. Does not overwrite v3/v1 artifacts.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
STAGES="${STAGES:-factor1,wan_smoke}"

want() { [[ ",${STAGES}," == *",$1,"* ]]; }

if want factor1; then
  STAGES="${FACTOR1_STAGES:-codec,joint}" \
    bash "${SCRIPT_DIR}/17_probe_r7_factor1.sh"
fi
if want wan_teacher_cache; then
  WAN_TEACHER_OUTPUT="${WAN_TEACHER_OUTPUT:-${PERSISTENT_OBS_ROOT}/teacher_cache/r7_wan_native_1k_t1_v1}" \
    R7_NAMESPACE="${R7_NAMESPACE:-r7_t1_c192_probe_v1}" \
    bash "${SCRIPT_DIR}/18_cache_wan_teacher_bridge.sh"
fi
if want wan_teacher_train; then
  WAN_TEACHER_OUTPUT="${WAN_TEACHER_OUTPUT:-${PERSISTENT_OBS_ROOT}/teacher_cache/r7_wan_native_1k_t1_v1}" \
    WAN_TEACHER_LOCAL_DIR="${WAN_TEACHER_LOCAL_DIR:-${LOCAL_CACHE_ROOT}/teacher_cache/r7_wan_native_1k_t1_v1}" \
    BRIDGE_NAMESPACE="${BRIDGE_NAMESPACE:-r7_wan_teacher_bridge_t1_probe_v1}" \
    TEMPORAL_FACTOR="${TEMPORAL_FACTOR:-1}" \
    bash "${SCRIPT_DIR}/19_train_wan_teacher_bridge.sh"
fi
if want wan_smoke; then
  bash "${SCRIPT_DIR}/smoke_wan_t2v.sh"
fi

printf 'R7 redesign diagnostic stages complete: %s\n' "${STAGES}"
