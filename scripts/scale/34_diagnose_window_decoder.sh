#!/bin/bash
# Frozen decoder stress test, no optimizer or training checkpoint writes.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export WINDOW_DIAGNOSTIC_ONLY=1 DIAGNOSTIC_MODE=perturbation RESUME=0 NO_TEXT=1
export AE_VARIANT=t2v2 WINDOW_AE_NORM=legacy
export R7_CACHE_VERSION=r7_window_t2v2_legacy_diag_v2
export DIAGNOSTIC_SOURCE="${DIAGNOSTIC_SOURCE:-r7_window_t2v2_legacy_x0_shift3_v1}"
export DIAGNOSTIC_CKPT_NAME="${DIAGNOSTIC_CKPT_NAME:-checkpoint_best_reconstruction.pt}"
export EXPECTED_STEP="${EXPECTED_STEP:-2500}"
export WINDOW_NAMESPACE="${WINDOW_NAMESPACE:-r7_window_t2v2_legacy_decoder_shift3_s2500_v1}"
[[ "${DIAGNOSTIC_SOURCE}" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Unsafe source'; exit 2; }
[[ "${WINDOW_NAMESPACE}" != "${DIAGNOSTIC_SOURCE}" ]] || { echo 'Output must differ from source'; exit 2; }
exec bash "${SCRIPT_DIR}/29_train_window_diffusion.sh"
