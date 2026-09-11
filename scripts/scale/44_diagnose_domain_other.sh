#!/bin/bash
# Evaluate the alternate outdoor validation set after either arm reaches6000.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"
ARM="${ARM:-single}"
[[ "${ARM}" == single || "${ARM}" == mixed ]] || exit 2
DOMAIN_DIR="${LOCAL_CACHE_ROOT}/metadata/domain_csv_v2"
ensure_local_checkpoint "${SPATIALVID_METADATA}" "${SPATIALVID_METADATA_URL}" 'HQ metadata'
"${PYTHON_BIN}" "${PROJECT}/scripts/prepare_domain_csv.py" --metadata "${SPATIALVID_METADATA}" \
  --selection "${PROJECT}/configs/spatialvid_domain_v2.json" --output "${DOMAIN_DIR}"
ALTERNATE_EVAL_CSV_SHA256=$("${PYTHON_BIN}" -c 'import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "${DOMAIN_DIR}/eval_other_128.csv")
export ALTERNATE_EVAL_CSV_SHA256
export AE_VARIANT=t2v2 WINDOW_AE_NORM=legacy WINDOW_DIAGNOSTIC_ONLY=1
export R7_CACHE_VERSION="r7_domain_${ARM}_t2v2_legacy_diag_v2"
export DIAGNOSTIC_SOURCE="${DIAGNOSTIC_SOURCE:-r7_domain_${ARM}_uniform_v2}"
export WINDOW_NAMESPACE="${WINDOW_NAMESPACE:-r7_domain_${ARM}_other_s6000_v2}"
export DIAGNOSTIC_CKPT_NAME=checkpoint_final.pt EXPECTED_STEP=6000 RESUME=0
export DIAGNOSTIC_MODE=standard DIAGNOSTIC_CLIPS=32 PREVIEW_CLIPS=4 DIAGNOSTIC_SEEDS='101 211'
export EVAL_MANIFEST="${PERSISTENT_OBS_ROOT}/cache_latents/${R7_CACHE_VERSION}/other/eval/manifest.txt"
export EVAL_STATS="${PERSISTENT_OBS_ROOT}/cache_latents/${R7_CACHE_VERSION}/other/eval/stats.pt"
exec bash "${SCRIPT_DIR}/29_train_window_diffusion.sh"
