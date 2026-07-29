#!/bin/bash
# Two-step R7 end-to-end cluster smoke.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TEMPORAL_FACTOR="${TEMPORAL_FACTOR:-2}" \
PHASE=codec MAX_STEPS=2 AUTO_RESUME=0 NUM_WORKERS=0 EVAL_CLIPS=1 \
LOG_EVERY=1 EVAL_EVERY=1 SAVE_EVERY=1 ACCUM_STEPS=1 \
bash "${SCRIPT_DIR}/11_train_causal_tokenizer.sh"
