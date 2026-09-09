#!/bin/bash
# Single intervention: clean-target deep supervision at block6, weight0.5.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export WINDOW_NAMESPACE="${WINDOW_NAMESPACE:-r7_window_t2v2_legacy_x0_shift3_aux6_w05_v1}"
export AUX_LAYER=6 AUX_WEIGHT=0.5
# Inherit exactly the stage33 recipe and its storage/full-resume lifecycle.
exec bash "${SCRIPT_DIR}/33_train_window_shift3.sh"
