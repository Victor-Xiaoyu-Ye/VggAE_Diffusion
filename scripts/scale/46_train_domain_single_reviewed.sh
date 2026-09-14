#!/bin/bash
# Fresh single-domain diffusion using the completed exact-cohort AE audit.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export ARM=single PREPARE_CACHE=0
export WINDOW_NAMESPACE="${WINDOW_NAMESPACE:-r7_domain_single_uniform_reviewed_v1}"
export RESUME="${RESUME:-0}"
export MIN_AE_PSNR=22.3
export AE_REFERENCE_FILE="${SCRIPT_DIR}/../../configs/ae_reference_single_v1.json"
exec bash "${SCRIPT_DIR}/43_train_domain_uniform.sh"
