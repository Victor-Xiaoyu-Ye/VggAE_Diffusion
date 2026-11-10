#!/bin/bash
# Priority three-node run: live native RGB context, with frozen matched controls.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
export DEFAULT_FEEDBACK_CONFIG="${PROJECT_ROOT}/configs/edit_state_context_sync_v1.json"
export DEFAULT_FEEDBACK_ARMS='native rgb_base rae_base rae_joint rae_sync rgb_joint rgb_sync'
export FEEDBACK_NAMESPACE="${FEEDBACK_NAMESPACE:-edit_state_context_sync_v1}"
export MASTER_PORT="${MASTER_PORT:-30900}"
echo "Stage68: calibrated reference -> native reactive context + state + velocity; frozen Stage65 weights."
echo "namespace=${FEEDBACK_NAMESPACE}; config=${FEEDBACK_CONFIG:-${DEFAULT_FEEDBACK_CONFIG}}"
exec bash "${SCRIPT_DIR}/66_refine_vace_edit_state.sh" "$@"
