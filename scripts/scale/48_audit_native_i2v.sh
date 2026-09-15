#!/bin/bash
# Native generation / R7 replay / EMA6000 comparison, no optimization.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"
export PYTHONPATH="${PROJECT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MASTER_PORT="${MASTER_PORT:-29948}"
export AUDIT_TIMEOUT_SECONDS="${AUDIT_TIMEOUT_SECONDS:-172800}"
configure_modelarts_distributed
if [[ "${1:-}" != --leader ]]; then
  exec "${PYTHON_BIN}" -u "${PROJECT}/scripts/coordinated_diagnostic.py" \
    bash "${BASH_SOURCE[0]}" --leader
fi
[[ "${NODE_RANK}" == 0 ]] || exit 2
WINDOW_NAMESPACE="${WINDOW_NAMESPACE:-r7_native_i2v_roundtrip_v1}"
[[ "${WINDOW_NAMESPACE}" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Unsafe namespace'; exit 2; }
RESUME="${RESUME:-0}"
[[ "${RESUME}" == 0 || "${RESUME}" == 1 ]] || exit 2
OUT="${SCALE_ROOT}/${WINDOW_NAMESPACE}"
CURRENT="${SCALE_REMOTE_ROOT}/${WINDOW_NAMESPACE}"
MIRROR="${SCALE_MIRROR_ROOT}/${WINDOW_NAMESPACE}"
IO="${PROJECT}/scripts/window_run_io.py"
if [[ "${RESUME}" == 0 ]]; then
  "${PYTHON_BIN}" "${IO}" guard --root "${OUT}" --root "${CURRENT}" --root "${MIRROR}"
fi
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
exec > >(tee -a "${OUT}/logs/native_audit.log") 2>&1
echo "Native audit: one NPU on leader; other ModelArts nodes remain alive via coordinator."
echo "No training. Native 81frames/16fps, comparison first1second/9frames."

# Intentionally do not inherit spatialvid_config.sh's T2V fallback.
NATIVE_WAN_CKPT_DIR="${NATIVE_WAN_CKPT_DIR:-}"
if [[ -z "${NATIVE_WAN_CKPT_DIR}" ]]; then
  for candidate in "${VGGAE_REF_ROOT}/Wan2.1-I2V-14B-480P" \
      "${VGGAE_REF_ROOT}/Wan2.1/checkpoints/Wan2.1-I2V-14B-480P" \
      /public2/LiZhen/yexiaoyu/ckpt/Wan2.1-I2V-14B-480P; do
    if [[ -d "${candidate}" ]]; then NATIVE_WAN_CKPT_DIR="${candidate}"; break; fi
  done
fi
if [[ -n "${NATIVE_WAN_URL:-}" ]]; then
  NATIVE_WAN_CKPT_DIR="${NATIVE_WAN_CKPT_DIR:-${VGGAE_REF_ROOT}/Wan2.1-I2V-14B-480P}"
  "${PYTHON_BIN}" - "${NATIVE_WAN_CKPT_DIR}" "${NATIVE_WAN_URL}" "${NATIVE_WAN_MIRROR_URL:-}" <<'PY'
import sys
from utils.native_audit_io import validate_wan_checkpoint
from utils.moxing_io import copy_directory
target,*sources=sys.argv[1:]
try:
    validate_wan_checkpoint(target)
except Exception:
    errors=[]
    for source in dict.fromkeys(sources):
        if not source: continue
        try:
            copy_directory(source,target)
            validate_wan_checkpoint(target)
            break
        except Exception as exc: errors.append(repr(exc))
    else: raise RuntimeError('Native I2V staging failed: '+str(errors))
PY
fi
[[ -n "${NATIVE_WAN_CKPT_DIR}" && -d "${NATIVE_WAN_CKPT_DIR}" ]] || {
  echo 'Native I2V weights missing. Set NATIVE_WAN_CKPT_DIR to the complete Wan2.1-I2V-14B-480P directory,'
  echo 'or NATIVE_WAN_URL (optional NATIVE_WAN_MIRROR_URL) to its OBS directory. T2V fallback is forbidden.'
  exit 2
}
R7_CKPT="${R7_CKPT:-${SCALE_ROOT}/r7_t2_c192_v2/joint/checkpoint_best.pt}"
ensure_local_checkpoint "${R7_CKPT}" \
  "${R7_CKPT_URL:-${SCALE_REMOTE_ROOT}/r7_t2_c192_v2/joint/checkpoint_best.pt}" 'frozen reviewed R7' \
  "${R7_CKPT_MIRROR_URL:-${SCALE_MIRROR_ROOT}/r7_t2_c192_v2/joint/checkpoint_best.pt}"
ensure_local_checkpoint "${STREAMVGGT_CKPT}" "${STREAMVGGT_URL:-}" 'StreamVGGT' "${STREAMVGGT_MIRROR_URL:-}"
SOURCE_RUN=r7_domain_single_uniform_reviewed_v1
SOURCE_CKPT="${LOCAL_CACHE_ROOT}/native_audit_inputs/${SOURCE_RUN}/checkpoint_final.pt"
"${PYTHON_BIN}" "${IO}" stage --destination "${SOURCE_CKPT}" \
  --root "${DIAGNOSTIC_CKPT_URL:-${SCALE_REMOTE_ROOT}/${SOURCE_RUN}/checkpoint_final.pt}" \
  --root "${DIAGNOSTIC_CKPT_MIRROR_URL:-${SCALE_MIRROR_ROOT}/${SOURCE_RUN}/checkpoint_final.pt}"
CACHE="${PERSISTENT_OBS_ROOT}/cache_latents/r7_domain_single_t2v2_legacy_diag_v2/eval"
ARGS=(--output_dir "${OUT}" --work_dir "${LOCAL_CACHE_ROOT}/native_audit_inputs/${WINDOW_NAMESPACE}"
  --wan_ckpt "${NATIVE_WAN_CKPT_DIR}" --checkpoint "${SOURCE_CKPT}"
  --r7_ckpt "${R7_CKPT}" --encoder_ckpt "${STREAMVGGT_CKPT}"
  --eval_manifest "${EVAL_MANIFEST:-${CACHE}/manifest.txt}" --eval_stats "${EVAL_STATS:-${CACHE}/stats.pt}"
  --ae_reference "${PROJECT}/configs/ae_reference_single_v1.json"
  --clips "${NATIVE_CLIPS:-32}" --seeds 101 211
  --prompt "${NATIVE_PROMPT:-A realistic continuous video of the scene.}"
  --read_root "${CURRENT}" --read_root "${MIRROR}")
if [[ "${RESUME}" == 1 ]]; then ARGS+=(--resume); fi
for phase in prepare native replay; do
  "${PYTHON_BIN}" -u "${PROJECT}/audit_native_i2v.py" "${phase}" "${ARGS[@]}"
done
"${PYTHON_BIN}" - "${OUT}" "${NATIVE_CLIPS:-32}" <<'PY'
import json,sys
from pathlib import Path
out=Path(sys.argv[1]);expected=int(sys.argv[2])*2
summary=json.loads((out/'summary.json').read_text())
if summary['status']!='completed' or summary['cases']!=expected:
    raise RuntimeError('incomplete native comparison')
print(f'PASS: {expected} comparisons committed. Visual quality still requires review.')
PY
