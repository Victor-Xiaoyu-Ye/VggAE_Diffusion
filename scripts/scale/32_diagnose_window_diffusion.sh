#!/bin/bash
# Reuse stage29 storage/distributed lifecycle; diagnostic branch never trains.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export WINDOW_DIAGNOSTIC_ONLY=1 RESUME=0 NO_TEXT=1
export AE_VARIANT="${AE_VARIANT:-t2v2}"
export DIAGNOSTIC_SOURCE="${DIAGNOSTIC_SOURCE:-r7_window_t2v2_legacy_x0_diag_v2}"
[[ "${DIAGNOSTIC_SOURCE}" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Unsafe diagnostic source'; exit 2; }
export WINDOW_NAMESPACE="${WINDOW_NAMESPACE:-r7_window_t2v2_legacy_x0_diagnostics_v1}"
[[ "${WINDOW_NAMESPACE}" != "${DIAGNOSTIC_SOURCE}" ]] || { echo 'Output must differ from training source'; exit 2; }
exec bash "${SCRIPT_DIR}/29_train_window_diffusion.sh"
