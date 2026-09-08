#!/bin/bash
# Legacy filename retained: read-only sampler checks ONLY, never a trainer.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/spatialvid.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"
configure_modelarts_distributed
require_output_url
if [[ "${NODE_RANK}" -ne 0 ]]; then exit 0; fi
export PYTHONUNBUFFERED=1
PYTHONPATH="${PROJECT}" "${PYTHON_BIN}" "${PROJECT}/scripts/test_r7_sampler_contract.py"
export FLOW_NAMESPACE="${FLOW_NAMESPACE:-r7_sampler_validation_n1_probe_v2}"
[[ "${FLOW_NAMESPACE}" =~ ^[a-zA-Z0-9_-]+$ && ( "${FLOW_NAMESPACE}" == *probe* || "${FLOW_NAMESPACE}" == *diag* ) ]] || {
  echo 'Use a plain, unique FLOW_NAMESPACE containing probe or diag.' >&2; exit 2;
}
[[ "${RUN_N1:-0}" == 0 ]] || {
  echo 'Stage 27 is validation only. Automatic stage 25/26 training has been withdrawn.' >&2; exit 2;
}
[[ -z "${RESUME:-}" ]] || { echo 'Read-only validation does not accept training RESUME.' >&2; exit 2; }
export R7_NAMESPACE="${R7_NAMESPACE:-r7_t1_c192_geo112_tex80_probe_v1}"
export R7_CKPT="${R7_CKPT:-${SCALE_ROOT}/${R7_NAMESPACE}/joint/checkpoint_best.pt}"
ensure_local_checkpoint "${R7_CKPT}" \
  "${R7_CKPT_URL:-${SCALE_REMOTE_ROOT}/${R7_NAMESPACE}/joint/checkpoint_best.pt}" \
  'frozen R7' "${R7_CKPT_MIRROR_URL:-${SCALE_MIRROR_ROOT}/${R7_NAMESPACE}/joint/checkpoint_best.pt}"
DET_NAMESPACE="${DET_NAMESPACE:-r7_vggt_quick_geo112_tex80_v2/det_k1_n1}"
DET_CKPT="${DET_CKPT:-${SCALE_ROOT}/${DET_NAMESPACE}/checkpoint_latest.pt}"
ensure_local_checkpoint "${DET_CKPT}" \
  "${DET_CKPT_URL:-${SCALE_REMOTE_ROOT}/${DET_NAMESPACE}/checkpoint_latest.pt}" \
  'historical deterministic n1' "${DET_CKPT_MIRROR_URL:-${SCALE_MIRROR_ROOT}/${DET_NAMESPACE}/checkpoint_latest.pt}"
require_file "${STREAMVGGT_CKPT}" 'StreamVGGT encoder'
SPATIALVID_OFT_ROOT="${SPATIALVID_OFT_ROOT:-${PERSISTENT_OBS_ROOT}/spatial-vid-hq-oft}"
SPATIALVID_METADATA_URL="${SPATIALVID_OFT_ROOT}/data/train/SpatialVID_HQ_metadata.csv"
SPATIALVID_VIDEO_ROOT="${SPATIALVID_OFT_ROOT}/videos/SpatialVid/HQ/videos"
SPATIALVID_METADATA="${LOCAL_CACHE_ROOT}/metadata/SpatialVID_HQ_metadata_oft.csv"
SPATIALVID_SPLIT_DIR="${RUN_ROOT}/metadata/spatialvid_oft_seed${SPLIT_SEED}"
SPATIALVID_TRAIN_10K_CSV="${SPATIALVID_SPLIT_DIR}/train_10k.csv"
SPATIALVID_EVAL_CSV="${SPATIALVID_SPLIT_DIR}/eval.csv"
output="${SCALE_ROOT}/${FLOW_NAMESPACE}/contract"
remote="${SCALE_REMOTE_ROOT}/${FLOW_NAMESPACE}/contract"
mirror="${SCALE_MIRROR_ROOT}/${FLOW_NAMESPACE}/contract"
# Check both persistent destinations before sync can create anything.
PYTHONPATH="${PROJECT}" "${PYTHON_BIN}" - "${output}" "${remote}" "${mirror}" <<'PY'
import pathlib, sys
from utils.moxing_io import is_remote_path
for root in sys.argv[1:]:
    path = root.rstrip('/') + '/contract_status.json'
    if is_remote_path(path):
        import moxing as mox
        exists = mox.file.exists(path)
    else:
        exists = pathlib.Path(path).exists()
    if exists:
        raise SystemExit('Existing contract output; select a fresh FLOW_NAMESPACE: ' + root)
PY
ensure_spatialvid_subset_splits
mkdir -p "${output}/logs"
start_output_sync "${output}" "${remote}"
trap 'stop_output_sync "${output}" "${remote}"' EXIT
set +e
PYTHONPATH="${PROJECT}" "${PYTHON_BIN}" -u "${PROJECT}/validate_r7_sampler_contract.py" \
  --eval_csv "${SPATIALVID_EVAL_CSV}" --video_root "${SPATIALVID_VIDEO_ROOT}" \
  --encoder_ckpt "${STREAMVGGT_CKPT}" --r7_ckpt "${R7_CKPT}" \
  --deterministic_ckpt "${DET_CKPT}" --output_dir "${output}" \
  --dtype "${DTYPE:-bf16}" --num_workers "${CONTRACT_NUM_WORKERS:-0}" \
  --sample_steps "${CONTRACT_SAMPLE_STEPS:-1,30,60}" \
  --sample_seeds "${SAMPLE_SEEDS:-42,43,44,45}" \
  2>&1 | tee -a "${output}/logs/validation.log"
codes=("${PIPESTATUS[@]}")
set -e
PYTHONPATH="${PROJECT}" "${PYTHON_BIN}" - "${output}/contract_status.json" "${codes[0]}" "${codes[1]}" <<'PY'
import json,pathlib,sys
from utils.flow_run_status import atomic_json
p=pathlib.Path(sys.argv[1])
s=json.loads(p.read_text()) if p.exists() else {'schema':'r7-sampler-contract-v1'}
s.update(validator_exit_code=int(sys.argv[2]),tee_exit_code=int(sys.argv[3]))
if any(int(v) for v in sys.argv[2:]):
    s.update(status='failed',passed=False,error=s.get('error','validator/tee exited nonzero; see exit codes and last phase'))
atomic_json(p,s)
PY
if [[ "${codes[0]}" -ne 0 ]]; then exit "${codes[0]}"; fi
if [[ "${codes[1]}" -ne 0 ]]; then exit "${codes[1]}"; fi
# Publication must succeed before the next phase starts.
"${PYTHON_BIN}" "${PROJECT}/scripts/moxing_transfer.py" "${output}" "${remote}" --directory
"${PYTHON_BIN}" "${PROJECT}/scripts/moxing_transfer.py" "${output}" "${mirror}" --directory
stop_output_sync "${output}" "${remote}"
trap - EXIT
echo 'Read-only validation finished. contract_status.json checks sampler plumbing only. No diffusion training launched.'
