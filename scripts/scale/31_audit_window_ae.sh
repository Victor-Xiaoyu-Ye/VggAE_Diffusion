#!/bin/bash
# Same raw clips, same weights, historical-vs-current GroupNorm. No training.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"
configure_modelarts_distributed
[[ "${NODE_RANK}" == 0 ]] || exit 0
export PYTHONPATH="${PROJECT}${PYTHONPATH:+:${PYTHONPATH}}"
R7_NAMESPACE=r7_t2_c192_v2
WINDOW_NAMESPACE="${WINDOW_NAMESPACE:-r7_window_t2v2_legacy_x0_diag_v2}"
[[ "${WINDOW_NAMESPACE}" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Unsafe namespace'; exit 2; }
ATTEMPT="attempt$(date -u +%Y%m%dT%H%M%S)"
OUT="${SCALE_ROOT}/window_ae_audits/${WINDOW_NAMESPACE}/${ATTEMPT}"
CURRENT="${SCALE_REMOTE_ROOT}/window_ae_audits/${WINDOW_NAMESPACE}/${ATTEMPT}"
MIRROR="${SCALE_MIRROR_ROOT}/window_ae_audits/${WINDOW_NAMESPACE}/${ATTEMPT}"
IO="${PROJECT}/scripts/window_run_io.py"
"${PYTHON_BIN}" "${IO}" guard --root "${OUT}" --root "${CURRENT}" --root "${MIRROR}"
mkdir -p "${OUT}/logs"
"${PYTHON_BIN}" "${IO}" publish --source "${OUT}" --root "${CURRENT}" --root "${MIRROR}" --watch &
SYNC_PID=$!
finish() {
  local code=$?
  trap - EXIT
  kill "${SYNC_PID}" 2>/dev/null || true
  wait "${SYNC_PID}" 2>/dev/null || true
  "${PYTHON_BIN}" - "${OUT}" "${code}" <<'PY'
import sys,time
from pathlib import Path
from scripts.window_run_io import atomic
atomic(Path(sys.argv[1])/'launcher_exit.json',dict(exit_code=int(sys.argv[2]),unix_time=time.time()))
PY
  if ! "${PYTHON_BIN}" "${IO}" publish --source "${OUT}" --root "${CURRENT}" --root "${MIRROR}"; then
    [[ "${code}" != 0 ]] || code=74
  fi
  exit "${code}"
}
trap finish EXIT
exec > >(tee -a "${OUT}/logs/audit.log") 2>&1
R7_CKPT="${R7_CKPT:-${SCALE_ROOT}/${R7_NAMESPACE}/joint/checkpoint_best.pt}"
ensure_local_checkpoint "${R7_CKPT}" \
  "${R7_CKPT_URL:-${SCALE_REMOTE_ROOT}/${R7_NAMESPACE}/joint/checkpoint_best.pt}" \
  'historical t2v2 AE' "${R7_CKPT_MIRROR_URL:-${SCALE_MIRROR_ROOT}/${R7_NAMESPACE}/joint/checkpoint_best.pt}"
AUDIT_CACHE="${AUDIT_CACHE:-${PERSISTENT_OBS_ROOT}/cache_latents/r7_window_t2v2_diag_v1/eval}"
"${PYTHON_BIN}" -u "${PROJECT}/audit_window_ae.py" \
  --manifest "${AUDIT_MANIFEST:-${AUDIT_CACHE}/manifest.txt}" \
  --stats "${AUDIT_STATS:-${AUDIT_CACHE}/stats.pt}" \
  --r7_ckpt "${R7_CKPT}" --encoder_ckpt "${STREAMVGGT_CKPT}" \
  --output_dir "${OUT}" --selected_norm "${WINDOW_AE_NORM:-legacy}" \
  --clips "${AUDIT_CLIPS:-16}" --previews "${AUDIT_PREVIEWS:-4}" \
  --min_psnr "${MIN_AE_PSNR:-23.5}" --min_gain "${MIN_AE_NORM_GAIN:-2.0}"
