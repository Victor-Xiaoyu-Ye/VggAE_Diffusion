#!/bin/bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export AE_VARIANT=t2v2 WINDOW_AE_NORM=legacy
export R7_CACHE_VERSION=r7_domain_single_t2v2_legacy_diag_v2
export WINDOW_NAMESPACE="${WINDOW_NAMESPACE:-r7_domain_single_trajectory_s6000_v1}"
export WINDOW_DIAGNOSTIC_ONLY=1 DIAGNOSTIC_MODE=domain_trajectory RESUME=0
export DIAGNOSTIC_SOURCE=r7_domain_single_uniform_reviewed_v1
export DIAGNOSTIC_CKPT_NAME=checkpoint_final.pt EXPECTED_STEP=6000
export DIAGNOSTIC_CLIPS=32 PREVIEW_CLIPS=8 DIAGNOSTIC_SEEDS='101 211'
export AE_REFERENCE_FILE="${SCRIPT_DIR}/../../configs/ae_reference_single_v1.json"
unset R7_CACHE_OBS_ROOT TRAIN_MANIFEST TRAIN_STATS EVAL_MANIFEST EVAL_STATS ALTERNATE_EVAL_CSV_SHA256
exec bash "${SCRIPT_DIR}/29_train_window_diffusion.sh"
