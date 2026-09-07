#!/bin/bash
# Two-step one-pair smokes for deterministic and flow single-target paths.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT=$(cd "${SCRIPT_DIR}/../.." && pwd)

PYTHONPATH="${PROJECT}${PYTHONPATH:+:${PYTHONPATH}}" \
  python "${SCRIPT_DIR}/../test_vggt_quick_probes.py"

if [[ "${FLOW_ONLY:-0}" != 1 ]]; then
  MODE=train \
  OUTPUT_NAME="${OUTPUT_NAME:-smoke_r7_vggt_single_target_v1}" \
  GEN_MODE=deterministic TARGET_INDEX=1 \
  MAX_SAMPLES=1 EVAL_SAMPLES=1 MAX_STEPS=2 BATCH_SIZE=1 \
  HIDDEN_DIM="${HIDDEN_DIM:-192}" DEPTH="${DEPTH:-1}" \
  LAMBDA_RGB=0 LAMBDA_LPIPS=0 EVAL_LPIPS=0 \
  EVAL_EVERY=1 LOG_EVERY=1 NUM_WORKERS=0 \
  bash "${SCRIPT_DIR}/22_probe_vggt_generation.sh"
fi

if [[ "${SKIP_FLOW_SMOKE:-0}" != 1 ]]; then
  MODE=train \
  OUTPUT_NAME="${FLOW_OUTPUT_NAME:-smoke_r7_vggt_single_target_flow_v1}" \
  GEN_MODE=flow TARGET_INDEX=1 \
  MAX_SAMPLES=1 EVAL_SAMPLES=1 MAX_STEPS=2 BATCH_SIZE=1 \
  HIDDEN_DIM="${HIDDEN_DIM:-192}" DEPTH="${DEPTH:-1}" \
  LAMBDA_RGB=0 LAMBDA_LPIPS=0 EVAL_LPIPS=0 SAMPLE_STEPS=2 \
  EVAL_EVERY=1 LOG_EVERY=1 NUM_WORKERS=0 \
  bash "${SCRIPT_DIR}/22_probe_vggt_generation.sh"
fi

printf 'VGGT deterministic and flow smoke probes complete\n'
