#!/bin/bash
# Two-step R7 codec-trainer smoke in an isolated namespace.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# LEGACY_INIT_CKPT is optional. Without it, wrapper 11 resolves the original
# causal_dual_tokenizer_t2_c192_codec checkpoint from the standard output/mirror.
EXTRA_ENV=()
if [[ -n "${LEGACY_INIT_CKPT:-}" ]]; then
  EXTRA_ENV+=(LEGACY_INIT_CKPT="${LEGACY_INIT_CKPT}")
fi

env "${EXTRA_ENV[@]}" \
R7_NAMESPACE="${R7_NAMESPACE:-r7_smoke_t2_c192_v2}" \
TEMPORAL_FACTOR="${TEMPORAL_FACTOR:-2}" \
PHASE=codec MAX_STEPS=2 MIN_STEPS=2 AUTO_RESUME=0 NUM_WORKERS=0 EVAL_CLIPS=1 \
EARLY_STOP_PATIENCE=0 LOG_EVERY=1 EVAL_EVERY=1 SAVE_EVERY=1 ACCUM_STEPS=1 \
bash "${SCRIPT_DIR}/11_train_causal_tokenizer.sh"

# This smoke intentionally validates codec forward/backward/eval/save/sync.
