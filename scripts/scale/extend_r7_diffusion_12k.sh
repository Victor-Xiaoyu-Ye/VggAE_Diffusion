#!/bin/bash
# Extend the production R7 diffusion in-place from 6k to 12k steps.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

MODE=train \
EXTEND=1 \
MAX_STEPS=12000 \
EXTENSION_LR="${EXTENSION_LR:-2e-5}" \
EXTENSION_WARMUP_STEPS="${EXTENSION_WARMUP_STEPS:-200}" \
bash "${SCRIPT_DIR}/13_train_causal_video_diffusion.sh"
