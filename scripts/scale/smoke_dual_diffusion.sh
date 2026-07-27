#!/bin/bash
# Two-step cluster smoke for E9-v3's online encode, motion losses, EMA eval,
# checkpoint, and RGB preview paths.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

STAGES=diffusion \
PROBE_SUFFIX_OVERRIDE="c128_cf0_stmotion_smoke" \
MAX_STEPS_DIFF=2 \
WARMUP_STEPS=1 \
NUM_WORKERS=0 \
STAT_BATCHES=1 \
EVAL_CLIPS=1 \
SAMPLE_CLIPS=1 \
SAMPLE_STEPS=2 \
EVAL_EVERY=1 \
SAVE_EVERY=1 \
LOG_EVERY=1 \
AUX_WARMUP_STEPS=0 \
AUX_RAMP_STEPS=1 \
AUTO_RESUME=0 \
bash "${SCRIPT_DIR}/10_bottleneck_then_diffusion.sh"
