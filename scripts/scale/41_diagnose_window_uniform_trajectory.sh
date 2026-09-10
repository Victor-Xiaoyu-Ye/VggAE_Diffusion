#!/bin/bash
# Run only after stage40 has produced checkpoint_final.pt at step6000.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export DIAGNOSTIC_SOURCE="${DIAGNOSTIC_SOURCE:-r7_window_t2v2_legacy_memory64_uniform_v1}"
export WINDOW_NAMESPACE="${WINDOW_NAMESPACE:-r7_window_memory64_uniform_trajectory_s6000_v1}"
exec bash "${SCRIPT_DIR}/39_diagnose_window_trajectory.sh"
