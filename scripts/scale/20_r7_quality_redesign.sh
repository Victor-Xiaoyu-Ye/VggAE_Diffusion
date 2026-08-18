#!/bin/bash
# R7 quality redesign diagnostic chain. Does not overwrite v3/v1 artifacts.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STAGES="${STAGES:-factor1,wan_smoke}"

want() { [[ ",${STAGES}," == *",$1,"* ]]; }

if want factor1; then
  STAGES="${FACTOR1_STAGES:-codec,joint}" \
    bash "${SCRIPT_DIR}/17_probe_r7_factor1.sh"
fi
if want wan_teacher_cache; then
  bash "${SCRIPT_DIR}/18_cache_wan_teacher_bridge.sh"
fi
if want wan_teacher_train; then
  bash "${SCRIPT_DIR}/19_train_wan_teacher_bridge.sh"
fi
if want wan_smoke; then
  bash "${SCRIPT_DIR}/smoke_wan_t2v.sh"
fi

printf 'R7 redesign diagnostic stages complete: %s\n' "${STAGES}"
