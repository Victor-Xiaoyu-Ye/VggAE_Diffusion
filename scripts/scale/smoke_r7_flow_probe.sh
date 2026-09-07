#!/bin/bash
# Test contracts, then run two optimizer steps for each head on one NPU/GPU.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"
configure_modelarts_distributed
if [[ "${NODE_RANK}" -ne 0 ]]; then exit 0; fi
PYTHONPATH="${PROJECT}" "${PYTHON_BIN}" "${PROJECT}/scripts/test_r7_flow_probe.py"
export STAGE=smoke FLOW_NAMESPACE="${FLOW_NAMESPACE:-r7_geo112_flow_smoke_probe_v1}"
export HIDDEN_DIM="${HIDDEN_DIM:-96}" DEPTH="${DEPTH:-1}" NUM_HEADS="${NUM_HEADS:-3}"
export EVAL_EVERY=1 SAVE_EVERY=1 LOG_EVERY=1 WARMUP_STEPS=0
export EVAL_SAMPLES=1 TRAIN_EVAL_SAMPLES=1 SAMPLE_STEPS=2 SAMPLE_SEEDS=42
export EVAL_LPIPS=0 EVAL_EMA=1 NUM_WORKERS=0
for kind in plain_x0 preconditioned; do
  PREDICTION="${kind}" STOP_AFTER_STEPS=1 bash "${SCRIPT_DIR}/25_run_r7_flow_probe.sh"
  latest="${SCALE_ROOT}/${FLOW_NAMESPACE}/smoke_${kind}_n1_f1/checkpoint_latest.pt"
  PREDICTION="${kind}" RESUME="${latest}" STOP_AFTER_STEPS=0 \
    bash "${SCRIPT_DIR}/25_run_r7_flow_probe.sh"
done
