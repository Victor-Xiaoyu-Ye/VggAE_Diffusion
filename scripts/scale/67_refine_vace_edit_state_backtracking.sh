#!/bin/bash
# Reuse Stage66 lifecycle/IO with a separate adaptive-backtracking experiment.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
export DEFAULT_FEEDBACK_CONFIG="${PROJECT_ROOT}/configs/edit_state_feedback_bt_v2.json"
export FEEDBACK_NAMESPACE="${FEEDBACK_NAMESPACE:-edit_state_feedback_bt_v2}"
export MASTER_PORT="${MASTER_PORT:-30800}"
echo "Stage67 launcher: namespace=${FEEDBACK_NAMESPACE}; config=${FEEDBACK_CONFIG:-${DEFAULT_FEEDBACK_CONFIG}}"
# Do not set FEEDBACK_CONFIG here: only a user's override is explicit. This
# lets Stage66 recover a saved smoke config on a fresh node without replacing it.
exec bash "${SCRIPT_DIR}/66_refine_vace_edit_state.sh" "$@"
