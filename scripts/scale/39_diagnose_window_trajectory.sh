#!/bin/bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export AE_VARIANT=t2v2 WINDOW_AE_NORM=legacy
export R7_CACHE_VERSION=r7_window_t2v2_legacy_diag_v2
export WINDOW_NAMESPACE="${WINDOW_NAMESPACE:-r7_window_memory64_trajectory_s6000_v1}"
export WINDOW_DIAGNOSTIC_ONLY=1 DIAGNOSTIC_MODE=trajectory RESUME=0
export DIAGNOSTIC_SOURCE="${DIAGNOSTIC_SOURCE:-r7_window_t2v2_legacy_memory64_v1}"
[[ "${DIAGNOSTIC_SOURCE}" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Unsafe diagnostic source'; exit 2; }
export DIAGNOSTIC_CKPT_NAME=checkpoint_final.pt EXPECTED_STEP=6000
export DIAGNOSTIC_CLIPS=16 PREVIEW_CLIPS=2 DIAGNOSTIC_SEEDS='101 211'
exec bash "${SCRIPT_DIR}/29_train_window_diffusion.sh"
