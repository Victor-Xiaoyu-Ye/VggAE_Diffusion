#!/bin/bash
# CSV only on the desktop; bounded video staging and AE encoding on ModelArts.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"
ARM="${ARM:-single}"
[[ "${ARM}" == single || "${ARM}" == mixed ]] || { echo 'ARM must be single or mixed'; exit 2; }
export MASTER_PORT="${MASTER_PORT:-29920}"
configure_modelarts_distributed
require_scale_cluster
require_output_url
export PYTHONPATH="${PROJECT}${PYTHONPATH:+:${PYTHONPATH}}"
LOCAL_STAGE="${SCALE_ROOT}/domain_cache_${ARM}_v2"
CURRENT_STAGE="${SCALE_REMOTE_ROOT}/domain_cache_${ARM}_v2"
MIRROR_STAGE="${SCALE_MIRROR_ROOT}/domain_cache_${ARM}_v2"
if [[ "${NODE_RANK}" != 0 ]]; then
  CURRENT_STAGE="${CURRENT_STAGE}/workers/node${NODE_RANK}"
  MIRROR_STAGE="${MIRROR_STAGE}/workers/node${NODE_RANK}"
fi
mkdir -p "${LOCAL_STAGE}"
IO="${PROJECT}/scripts/window_run_io.py"
"${PYTHON_BIN}" "${IO}" publish --source "${LOCAL_STAGE}" --root "${CURRENT_STAGE}" \
  --root "${MIRROR_STAGE}" --watch --interval 60 &
DOMAIN_SYNC_PID=$!
finish_domain_cache() {
  local code=$?
  trap - EXIT
  kill "${DOMAIN_SYNC_PID}" 2>/dev/null || true
  wait "${DOMAIN_SYNC_PID}" 2>/dev/null || true
  "${PYTHON_BIN}" - "${LOCAL_STAGE}" "${code}" <<'PY'
import sys,time
from pathlib import Path
from scripts.window_run_io import atomic
atomic(Path(sys.argv[1])/'launcher_exit.json',dict(status='exited',exit_code=int(sys.argv[2]),unix_time=time.time()))
PY
  if ! "${PYTHON_BIN}" "${IO}" publish --source "${LOCAL_STAGE}" \
      --root "${CURRENT_STAGE}" --root "${MIRROR_STAGE}"; then
    [[ "${code}" != 0 ]] || code=74
  fi
  exit "${code}"
}
trap finish_domain_cache EXIT
exec > >(tee -a "${LOCAL_STAGE}/prepare.log") 2>&1
DOMAIN_DIR="${LOCAL_CACHE_ROOT}/metadata/domain_csv_v2"
mkdir -p "${DOMAIN_DIR}"
ensure_local_checkpoint "${SPATIALVID_METADATA}" "${SPATIALVID_METADATA_URL}" 'HQ metadata'
"${PYTHON_BIN}" "${PROJECT}/scripts/prepare_domain_csv.py" \
  --metadata "${SPATIALVID_METADATA}" --selection "${PROJECT}/configs/spatialvid_domain_v2.json" \
  --output "${DOMAIN_DIR}"
cp "${DOMAIN_DIR}/domain_manifest.json" "${LOCAL_STAGE}/domain_manifest.json"
cp "${PROJECT}/configs/spatialvid_domain_v2.json" "${LOCAL_STAGE}/selection.json"
"${PYTHON_BIN}" "${PROJECT}/scripts/preflight_domain_objects.py" \
  --csv "${DOMAIN_DIR}/train_${ARM}_2048.csv" "${DOMAIN_DIR}/eval_street_128.csv" "${DOMAIN_DIR}/eval_other_128.csv" \
  --video_root "${SPATIALVID_VIDEO_ROOT}" --output "${LOCAL_STAGE}/object_preflight.json"
run_distributed_barrier
export DOMAIN_VIDEO_ROOT="${SPATIALVID_VIDEO_ROOT}"
export AE_VARIANT=t2v2 WINDOW_AE_NORM=legacy
export CACHE_NUM_PARTITIONS=1 CACHE_PARTITION_ID=0 MAX_FAILURE_RATE=0
export SAMPLES_PER_TAR=64 BATCH_SIZE=1 NUM_WORKERS="${NUM_WORKERS:-2}"
export MOX_CACHE_WRITER_DIR="${LOCAL_CACHE_ROOT}/cache/domain_writer_v2/${ARM}"
BASE="${PERSISTENT_OBS_ROOT}/cache_latents/r7_domain_${ARM}_t2v2_legacy_diag_v2"
cache_split() {
  local name=$1 mode=$2 root=$3 version=$4 windows=$5
  export DOMAIN_CSV_FILE="${DOMAIN_DIR}/${name}.csv"
  DOMAIN_CSV_SHA256=$("${PYTHON_BIN}" -c 'import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "${DOMAIN_CSV_FILE}")
  export DOMAIN_CSV_SHA256 R7_CACHE_OBS_ROOT="${root}" R7_CACHE_VERSION="${version}" CLIPS_PER_VIDEO="${windows}"
  MODE="${mode}" bash "${SCRIPT_DIR}/28_prepare_window_cache.sh"
  run_distributed_barrier
}
cache_split "train_${ARM}_2048" train "${BASE}" "r7_domain_${ARM}_t2v2_legacy_diag_v2" 4
cache_split eval_street_128 eval "${BASE}" "r7_domain_${ARM}_t2v2_legacy_diag_v2" 1
cache_split eval_other_128 eval "${BASE}/other" "r7_domain_${ARM}_other_t2v2_legacy_diag_v2" 1
if [[ "${NODE_RANK}" == 0 ]]; then
  for suffix in train eval other/eval; do
    "${PYTHON_BIN}" "${PROJECT}/merge_latent_cache.py" --cache_dir "${BASE}/${suffix}" \
      --expected_partitions 1 --max_failure_rate 0
  done
  "${PYTHON_BIN}" "${PROJECT}/scripts/verify_domain_cache.py" \
    --root "${BASE}" --domain_manifest "${DOMAIN_DIR}/domain_manifest.json" --arm "${ARM}"
fi
run_distributed_barrier
echo "PASS: ${ARM} caches complete; test CSVs remain unused."
