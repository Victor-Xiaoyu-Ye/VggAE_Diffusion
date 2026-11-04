#!/bin/bash
# Matched text-only and region-supervision experiments; same cluster bootstrap.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "${SCRIPT_DIR}/../.." && pwd)
export EDIT_NAMESPACE="${EDIT_NAMESPACE:-vace_edit_aligned_v2}"
export EDIT_CONFIG="${EDIT_CONFIG:-${REPO}/configs/vace_edit_aligned_v2.json}"
export EDIT_BANK_NAMESPACE="${EDIT_BANK_NAMESPACE:-vace_edit_pilot_v1}"
export EDIT_REUSE_BANK=1
export EDIT_ARMS="${EDIT_ARMS:-sft rae rae_detached region_sft rae_balanced}"
export EDIT_CONTROLS="${EDIT_CONTROLS-${REPO}/configs/editing_source_regions_dev_v1.json}"
exec bash "${SCRIPT_DIR}/60_train_vace_edit_pilot.sh"
