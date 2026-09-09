#!/bin/bash
# Depth control vs stage35: auxiliary block6 -> block8 only.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export WINDOW_NAMESPACE="${WINDOW_NAMESPACE:-r7_window_t2v2_legacy_x0_shift3_aux8_w05_v1}"
export AUX_LAYER=8 AUX_WEIGHT=0.5
exec bash "${SCRIPT_DIR}/33_train_window_shift3.sh"
