#!/bin/bash
# Two-step factor-1 codec smoke. Run before the 6K factor-1 probe.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

PROBE_CONTRACT=1 \
R7_NAMESPACE="${R7_NAMESPACE:-r7_smoke_t1_c192_probe_v1}" \
TEMPORAL_FACTOR=1 PHASE=codec MAX_STEPS=2 MIN_STEPS=2 AUTO_RESUME=0 \
NUM_WORKERS=0 EVAL_CLIPS=1 EARLY_STOP_PATIENCE=0 LOG_EVERY=1 \
EVAL_EVERY=1 SAVE_EVERY=1 ACCUM_STEPS=1 \
bash "${SCRIPT_DIR}/11_train_causal_tokenizer.sh"
