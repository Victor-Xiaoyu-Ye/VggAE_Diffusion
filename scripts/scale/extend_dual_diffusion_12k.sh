#!/bin/bash
# Continue the completed E9-v3 run from 6k to 12k steps. Reuses the exact
# stmotion namespace/checkpoint and restarts only the exhausted LR schedule.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

STAGES=diffusion \
PROBE_SUFFIX_OVERRIDE="c128_cf0_stmotion" \
MAX_STEPS_DIFF=12000 \
EXTENSION_LR="${EXTENSION_LR:-1e-5}" \
EXTENSION_WARMUP_STEPS="${EXTENSION_WARMUP_STEPS:-200}" \
AUTO_RESUME=1 \
bash "${SCRIPT_DIR}/10_bottleneck_then_diffusion.sh"
