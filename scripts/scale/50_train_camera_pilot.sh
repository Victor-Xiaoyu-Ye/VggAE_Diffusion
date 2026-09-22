#!/bin/bash
# Matched pose/null fine-tuning arms: shared frozen EMA, cohort, AE and schedule.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"
export PYTHONPATH="${PROJECT}${PYTHONPATH:+:${PYTHONPATH}}"
export MASTER_PORT="${MASTER_PORT:-29950}" HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-7200}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export AUDIT_TIMEOUT_SECONDS="${AUDIT_TIMEOUT_SECONDS:-172800}"
configure_modelarts_distributed
[[ "${NNODES}" == 3 && "${NUM_NPUS}" == 8 ]] || { echo 'Stage50 requires exactly three nodes, eight NPUs per node'; exit 2; }
CAMERA_ARM="${CAMERA_ARM:-pose}"
CAMERA_STAGE="${CAMERA_STAGE:-all}"
[[ "${CAMERA_ARM}" =~ ^(pose|null)$ && "${CAMERA_STAGE}" =~ ^(all|prepare|train)$ ]] || exit 2
export WINDOW_NAMESPACE="${WINDOW_NAMESPACE:-r7_camera_pilot_${CAMERA_ARM}_v1}"
[[ "${WINDOW_NAMESPACE}" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
INPUTS="${LOCAL_CACHE_ROOT}/camera_pilot_v1"
INPUT_CURRENT="${SCALE_REMOTE_ROOT}/r7_camera_pilot_inputs_v1"
INPUT_MIRROR="${SCALE_MIRROR_ROOT}/r7_camera_pilot_inputs_v1"
FULL_CACHE="${PERSISTENT_OBS_ROOT}/cache_latents/r7_fullhq_t2v2_legacy_diag_v1/train"
OLD_EVAL="${PERSISTENT_OBS_ROOT}/cache_latents/r7_domain_single_t2v2_legacy_diag_v2/eval"
HELPER="${PROJECT}/scripts/stage_camera_pilot.py"
SHARED_ARGS=(--output "${INPUTS}" --root "${INPUT_CURRENT}" --root "${INPUT_MIRROR}")

prepare_leader() {
  [[ "${NODE_RANK}" == 0 ]] || return 2
  # Once committed, reuse exact bytes. Do not reselect videos or a moving best
  # checkpoint independently for the second arm or an interrupted training run.
  if "${PYTHON_BIN}" -u "${HELPER}" fetch "${SHARED_ARGS[@]}" --allow_missing; then
    echo '[camera inputs] Existing committed cohort reused without reselection.'
    return 0
  else
    code=$?
    [[ "${code}" == 3 ]] || return "${code}"
  fi
  [[ "${CAMERA_STAGE}" != train ]] || { echo 'CAMERA_STAGE=train requires prepared shared inputs'; return 2; }
  ensure_local_checkpoint "${SPATIALVID_METADATA}" "${SPATIALVID_METADATA_URL}" 'full HQ metadata'
  "${PYTHON_BIN}" -u "${HELPER}" freeze "${SHARED_ARGS[@]}" \
    --source "${SCALE_REMOTE_ROOT}/r7_fullhq_ti2v_1650m_v1/checkpoint_best_reconstruction.pt" \
    --source "${SCALE_MIRROR_ROOT}/r7_fullhq_ti2v_1650m_v1/checkpoint_best_reconstruction.pt"
  "${PYTHON_BIN}" -u "${PROJECT}/scripts/prepare_camera_pilot.py" \
    --metadata "${SPATIALVID_METADATA}" --train_manifest "${FULL_CACHE}/manifest.txt" \
    --eval_manifest "${OLD_EVAL}/manifest.txt" \
    --annotation_root "${SPATIALVID_OBS_ROOT}/annotations/SpatialVID/annotations" \
    --output_dir "${INPUTS}" --train_shards 96 --seed 42 \
    --progress_dir "${SCALE_ROOT}/r7_camera_pilot_inputs_v1/logs/${CAMERA_ARM}/node${NODE_RANK}"
  "${PYTHON_BIN}" -u "${HELPER}" publish "${SHARED_ARGS[@]}" \
    --file train_manifest.txt --file eval_manifest.txt --file camera_bank.pt --file camera_audit.json
}

if [[ "${1:-}" == --prepare-leader ]]; then
  prepare_leader
  exit 0
fi

# Preparation has its own node logs and receipts; the continuing full-HQ run
# remains untouched. No raw-video staging or full latent recaching is performed.
LOG_DIR="${SCALE_ROOT}/r7_camera_pilot_inputs_v1/logs/${CAMERA_ARM}/node${NODE_RANK}"
mkdir -p "${LOG_DIR}" "${LOCAL_CACHE_ROOT}/tmp"
export TMPDIR="${LOCAL_CACHE_ROOT}/tmp"
IO="${PROJECT}/scripts/window_run_io.py"
LOG_CURRENT="${INPUT_CURRENT}/logs/${CAMERA_ARM}/node${NODE_RANK}"
LOG_MIRROR="${INPUT_MIRROR}/logs/${CAMERA_ARM}/node${NODE_RANK}"
"${PYTHON_BIN}" "${IO}" publish --source "${LOG_DIR}" --root "${LOG_CURRENT}" --root "${LOG_MIRROR}" --watch &
SYNC_PID=$!
finish() {
  local code=$?
  trap - EXIT
  kill "${SYNC_PID}" 2>/dev/null || true
  wait "${SYNC_PID}" 2>/dev/null || true
  "${PYTHON_BIN}" - "${LOG_DIR}" "${code}" <<'PY'
import sys,time
from pathlib import Path
from scripts.window_run_io import atomic
atomic(Path(sys.argv[1])/'launcher_exit.json',dict(exit_code=int(sys.argv[2]),unix_time=time.time()))
PY
  if ! "${PYTHON_BIN}" "${IO}" publish --source "${LOG_DIR}" --root "${LOG_CURRENT}" --root "${LOG_MIRROR}"; then
    [[ "${code}" != 0 ]] || code=74
  fi
  exit "${code}"
}
trap finish EXIT
exec > >(tee -a "${LOG_DIR}/pipeline.log") 2>&1

# TCP coordination, not an HCCL barrier: annotation auditing and snapshot
# staging can take longer than the process-group timeout on waiting nodes.
TRAIN_PORT="${MASTER_PORT}"
MASTER_PORT=$((TRAIN_PORT+1)) "${PYTHON_BIN}" -u "${PROJECT}/scripts/coordinated_diagnostic.py" \
  bash "${BASH_SOURCE[0]}" --prepare-leader
if [[ "${CAMERA_STAGE}" == prepare ]]; then
  echo 'Shared camera cohort and source EMA committed. Both arms can now use CAMERA_STAGE=train.'
  exit 0
fi
MASTER_PORT=$((TRAIN_PORT+2)) "${PYTHON_BIN}" -u "${HELPER}" fetch_all "${SHARED_ARGS[@]}" \
  --timeout "${AUDIT_TIMEOUT_SECONDS}"
export MASTER_PORT="${TRAIN_PORT}"

read -r INIT_STEP INIT_NAME < <("${PYTHON_BIN}" - "${INPUTS}/initialization.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]));print(r['step'],r['file'])
PY
)
export INIT_STEP INIT_FROM="${INPUTS}/${INIT_NAME}" INIT_WEIGHTS=ema
export CAMERA_BANK="${INPUTS}/camera_bank.pt" CAMERA_MODE="${CAMERA_ARM}" CAMERA_DROPOUT=0.1
export TRAIN_MANIFEST="${INPUTS}/train_manifest.txt" TRAIN_STATS="${FULL_CACHE}/stats.pt"
export EVAL_MANIFEST="${INPUTS}/eval_manifest.txt" EVAL_STATS="${OLD_EVAL}/stats.pt"
export DOMAIN_PROTOCOL_FILE="${INPUTS}/camera_audit.json"
export AE_REFERENCE_FILE="${PROJECT}/configs/ae_reference_single_v1.json" MIN_AE_PSNR=22.3
export AE_VARIANT=t2v2 WINDOW_AE_NORM=legacy LARGE_RUN=1 NO_TEXT=0
export TEXT_DIR="${PERSISTENT_OBS_ROOT}/text_embeddings/fullhq_umt5_256_v1"
export WIDTH=1536 DEPTH=24 HEADS=24 BATCH_SIZE=1 ACCUM_STEPS=8
export MAX_STEPS=2000 WARMUP_STEPS=100 LR=2e-5 WD=0.01 DTYPE=bf16 EMA_DECAY=0.999
export PREDICTION=x0 TIME_DISTRIBUTION=uniform TIME_SHIFT=1 LOSS_FLOOR=0.05
export AUX_LAYER=0 AUX_WEIGHT=0 MEMORIZE_CLIPS=0 TEXT_DROPOUT=0.1 TEXT_CFG=1
export SAVE_EVERY=500 EVAL_EVERY=500 EVAL_CLIPS=32 PREVIEW_CLIPS=8
export SAMPLE_STEPS=64 SAMPLE_METHOD=euler SAMPLE_SEEDS=101,211 SEED=42 LOG_EVERY=10
export MEMORY_LIMIT_GIB="${MEMORY_LIMIT_GIB:-52}"
export RESUME="${RESUME:-0}" STOP_AFTER_STEPS="${STOP_AFTER_STEPS:-0}"
echo "Camera arm=${CAMERA_ARM}; shared EMA source step=${INIT_STEP}; new optimizer/schedule; 2000 updates."
echo 'Per-rank memory acceptance runs during training; DI_throughput counts future latent tokens per active NPU.'
bash "${SCRIPT_DIR}/29_train_window_diffusion.sh"
