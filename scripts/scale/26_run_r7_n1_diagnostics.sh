#!/bin/bash
# One ModelArts submission, bounded to n1 only. Never promotes to n16.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"
configure_modelarts_distributed
if [[ "${NODE_RANK}" -ne 0 ]]; then exit 0; fi
export FLOW_NAMESPACE="${FLOW_NAMESPACE:-r7_geo112_flow_n1_diagnostics_probe_v2}"
export MAX_STEPS="${MAX_STEPS:-2000}" STOP_AFTER_STEPS=0
export EVAL_EVERY="${EVAL_EVERY:-100}" DIAGNOSTIC_EVERY="${DIAGNOSTIC_EVERY:-500}"
export SAMPLE_STEPS="${SAMPLE_STEPS:-30}" DIAGNOSTIC_SAMPLE_STEPS="${DIAGNOSTIC_SAMPLE_STEPS:-60}"
export ORACLE_STARTS="${ORACLE_STARTS:-0.5,0.7,0.9}" SAMPLE_SEEDS="${SAMPLE_SEEDS:-42,43,44,45}"
export LOG_EVERY="${LOG_EVERY:-10}" SAVE_EVERY="${SAVE_EVERY:-500}"
export BATCH_SIZE="${BATCH_SIZE:-1}" HIDDEN_DIM="${HIDDEN_DIM:-384}" DEPTH="${DEPTH:-4}" NUM_HEADS="${NUM_HEADS:-6}"
export EVAL_LPIPS="${EVAL_LPIPS:-1}" EVAL_EMA="${EVAL_EMA:-1}"
unset RESUME
if [[ "${RUN_SMOKE:-1}" == 1 ]]; then
  FLOW_NAMESPACE="${FLOW_NAMESPACE}_smoke" bash "${SCRIPT_DIR}/smoke_r7_flow_probe.sh"
fi
# Always run both requested parameterizations; a failed quality gate is data,
# whereas a trainer or synchronization error stops this script with nonzero exit.
for kind in preconditioned direct_velocity; do
  printf '[n1-diagnostics] starting %s random-noise arm\n' "${kind}"
  STAGE=n1 PREDICTION="${kind}" bash "${SCRIPT_DIR}/25_run_r7_flow_probe.sh"
done
# Fixed-path is a diagnostic fallback, not a new-noise success claim.
need_fixed=$("${PYTHON_BIN}" - "${SCALE_ROOT}/${FLOW_NAMESPACE}" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1])
passed=[]
for kind in ('preconditioned','direct_velocity'):
    status=json.loads((root/f'n1_{kind}_n1_f1/memory_status.json').read_text())
    passed.append(status.get('passed') is True)
print('0' if any(passed) else '1')
PY
)
if [[ "${RUN_FIXED_PATH:-auto}" == 1 || ( "${RUN_FIXED_PATH:-auto}" == auto && "${need_fixed}" == 1 ) ]]; then
  printf '[n1-diagnostics] running plain-x0 fixed-noise/full-time-path diagnostic\n'
  STAGE=fixed_path PREDICTION=plain_x0 FIXED_PATH_SEED=42 \
    SAMPLE_SEEDS=43,44,45,46 FIXED_PATH_STEPS="${FIXED_PATH_STEPS:-32}" \
    bash "${SCRIPT_DIR}/25_run_r7_flow_probe.sh"
fi
printf '[n1-diagnostics] completed. Inspect run_status.json and memory_status.json separately. n16 was not launched.\n'
