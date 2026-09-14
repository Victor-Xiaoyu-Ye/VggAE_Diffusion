#!/bin/bash
# Read-only RAW replay: both failed domain cohorts and historical control.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"
configure_modelarts_distributed
[[ "${NODE_RANK}" == 0 ]] || exit 0
export WINDOW_AE_NORM=legacy AUDIT_REPORT_ONLY=1
export AUDIT_PREVIEWS="${AUDIT_PREVIEWS:-4}"
# Do not inherit per-cache overrides from an earlier diagnostic invocation.
unset AUDIT_MANIFEST AUDIT_STATS
failed=0
for cohort in single mixed historical; do
  export WINDOW_NAMESPACE="r7_domain_ae_replay_${cohort}_v1"
  if [[ "${cohort}" == historical ]]; then
    export AUDIT_CACHE="${PERSISTENT_OBS_ROOT}/cache_latents/r7_window_t2v2_legacy_diag_v2/eval"
    export AUDIT_BASELINE_RUN=r7_window_t2v2_legacy_x0_diag_v2 AUDIT_CLIPS=16
  else
    export AUDIT_CACHE="${PERSISTENT_OBS_ROOT}/cache_latents/r7_domain_${cohort}_t2v2_legacy_diag_v2/eval"
    export AUDIT_BASELINE_RUN="r7_domain_${cohort}_uniform_v2" AUDIT_CLIPS=32
  fi
  # Low PSNR is reported, not a runtime failure. Actual errors remain nonzero;
  # still attempt other cohorts so a missing object does not hide the control.
  if ! bash "${SCRIPT_DIR}/31_audit_window_ae.sh"; then failed=1; fi
done
exit "${failed}"
