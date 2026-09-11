#!/bin/bash
# One command on all six nodes; fresh single/mixed cohort training, optional cache.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"
export ARM="${ARM:-single}"
[[ "${ARM}" == single || "${ARM}" == mixed ]] || { echo 'ARM must be single or mixed'; exit 2; }
export MASTER_PORT="${MASTER_PORT:-29920}"
configure_modelarts_distributed
export AE_VARIANT=t2v2 WINDOW_AE_NORM=legacy NO_TEXT=1 PREDICTION=x0
export R7_CACHE_VERSION="r7_domain_${ARM}_t2v2_legacy_diag_v2"
export R7_CACHE_OBS_ROOT="${PERSISTENT_OBS_ROOT}/cache_latents/${R7_CACHE_VERSION}"
export WINDOW_NAMESPACE="${WINDOW_NAMESPACE:-r7_domain_${ARM}_uniform_v2}"
[[ "${WINDOW_NAMESPACE}" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
export MEMORIZE_CLIPS=0 AUX_LAYER=0 AUX_WEIGHT=0 TIME_DISTRIBUTION=uniform TIME_SHIFT=1
export MAX_STEPS=6000 WARMUP_STEPS=300 LR=1e-4 WD=0.01
export WIDTH=768 DEPTH=12 HEADS=12 BATCH_SIZE=1 ACCUM_STEPS=2
export DTYPE=bf16 EMA_DECAY=0.999 TEXT_DROPOUT=0.1 LOSS_FLOOR=0.05
export SAMPLE_STEPS=64 SAMPLE_METHOD=euler SAMPLE_SEEDS=101,211 SEED=42
export EVAL_EVERY=500 SAVE_EVERY=500 LOG_EVERY=10 EVAL_CLIPS=32 PREVIEW_CLIPS=8
export RESUME="${RESUME:-0}"
export DOMAIN_PROTOCOL_FILE="${PROJECT}/configs/spatialvid_domain_v2.json"
if [[ "${NODE_RANK}" == 0 && "${RESUME}" == 0 ]]; then
  "${PYTHON_BIN}" "${PROJECT}/scripts/window_run_io.py" guard \
    --root "${SCALE_ROOT}/${WINDOW_NAMESPACE}" --root "${SCALE_REMOTE_ROOT}/${WINDOW_NAMESPACE}" \
    --root "${SCALE_MIRROR_ROOT}/${WINDOW_NAMESPACE}"
fi
run_distributed_barrier
if [[ "${PREPARE_CACHE:-1}" == 1 && "${RESUME}" == 0 ]]; then
  bash "${SCRIPT_DIR}/42_prepare_domain_cache.sh"
fi
# Re-materialize immutable CSV identity on fresh nodes, including full resume.
DOMAIN_DIR="${LOCAL_CACHE_ROOT}/metadata/domain_csv_v2"
ensure_local_checkpoint "${SPATIALVID_METADATA}" "${SPATIALVID_METADATA_URL}" 'HQ metadata'
"${PYTHON_BIN}" "${PROJECT}/scripts/prepare_domain_csv.py" --metadata "${SPATIALVID_METADATA}" \
  --selection "${PROJECT}/configs/spatialvid_domain_v2.json" --output "${DOMAIN_DIR}"
"${PYTHON_BIN}" "${PROJECT}/scripts/verify_domain_cache.py" --root "${R7_CACHE_OBS_ROOT}" \
  --domain_manifest "${DOMAIN_DIR}/domain_manifest.json" --arm "${ARM}"
unset WINDOW_DIAGNOSTIC_ONLY TRAIN_MANIFEST TRAIN_STATS EVAL_MANIFEST EVAL_STATS
exec bash "${SCRIPT_DIR}/29_train_window_diffusion.sh"
