#!/bin/bash
# Same full-window generation chain; train subset only, never train on eval.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export AE_VARIANT=t2v2 WINDOW_AE_NORM=legacy NO_TEXT=1 PREDICTION=x0
export R7_CACHE_VERSION=r7_window_t2v2_legacy_diag_v2
export WINDOW_NAMESPACE="${WINDOW_NAMESPACE:-r7_window_t2v2_legacy_memory64_v1}"
export MEMORIZE_CLIPS=64 AUX_LAYER=0 AUX_WEIGHT=0 TIME_SHIFT=3
export MAX_STEPS=6000 WARMUP_STEPS=300 LR=1e-4 WD=0.01
export WIDTH=768 DEPTH=12 HEADS=12 BATCH_SIZE=1 ACCUM_STEPS=2
export DTYPE=bf16 EMA_DECAY=0.999 TEXT_DROPOUT=0.1 LOSS_FLOOR=0.05
export SAMPLE_STEPS=64 SAMPLE_METHOD=euler SAMPLE_SEEDS=101,211,307,401 SEED=42
export EVAL_EVERY=500 SAVE_EVERY=500 LOG_EVERY=10 EVAL_CLIPS=16 PREVIEW_CLIPS=64
export RESUME="${RESUME:-0}"
unset WINDOW_DIAGNOSTIC_ONLY
exec bash "${SCRIPT_DIR}/29_train_window_diffusion.sh"
