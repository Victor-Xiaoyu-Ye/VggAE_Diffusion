#!/bin/bash
# Read-only GT oracle on the established no-aux EMA2500 reference.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export AE_VARIANT=t2v2 WINDOW_AE_NORM=legacy
export R7_CACHE_VERSION=r7_window_t2v2_legacy_diag_v2
export WINDOW_NAMESPACE="${WINDOW_NAMESPACE:-r7_window_t2v2_legacy_subspace_s2500_v1}"
export WINDOW_DIAGNOSTIC_ONLY=1 DIAGNOSTIC_MODE=subspace RESUME=0
export DIAGNOSTIC_SOURCE=r7_window_t2v2_legacy_x0_shift3_v1
export DIAGNOSTIC_CKPT_NAME=checkpoint_best_reconstruction.pt EXPECTED_STEP=2500
export DIAGNOSTIC_CLIPS=16 PREVIEW_CLIPS=16
exec bash "${SCRIPT_DIR}/29_train_window_diffusion.sh"
