#!/bin/bash
# Full HQ, frozen historical R7, scratch 1.65B TI2V on exactly 3x8 Ascend.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"
export PYTHONPATH="${PROJECT}${PYTHONPATH:+:${PYTHONPATH}}"
export MASTER_PORT="${MASTER_PORT:-29949}" HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-7200}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
configure_modelarts_distributed
[[ "${NNODES}" == 3 && "${NUM_NPUS}" == 8 ]] || { echo 'Stage49 requires exactly three nodes, eight NPUs per node'; exit 2; }
FULLHQ_STAGE="${FULLHQ_STAGE:-all}"
[[ "${FULLHQ_STAGE}" =~ ^(all|probe|prepare|text|cache|train)$ ]] || exit 2
export WINDOW_NAMESPACE="${WINDOW_NAMESPACE:-r7_fullhq_ti2v_1650m_v1}"
[[ "${WINDOW_NAMESPACE}" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
META="${LOCAL_CACHE_ROOT}/metadata/fullhq_v1"
CACHE="${PERSISTENT_OBS_ROOT}/cache_latents/r7_fullhq_t2v2_legacy_diag_v1"
TEXT="${PERSISTENT_OBS_ROOT}/text_embeddings/fullhq_umt5_256_v1"
INPUT_LOG="${SCALE_ROOT}/fullhq_inputs_v1/node${NODE_RANK}"
mkdir -p "${META}" "${INPUT_LOG}" "${LOCAL_CACHE_ROOT}/tmp"
export TMPDIR="${LOCAL_CACHE_ROOT}/tmp"
export MOX_VIDEO_CACHE_GB="${FULLHQ_VIDEO_CACHE_GB:-200}"
IO="${PROJECT}/scripts/window_run_io.py"
CURRENT="${SCALE_REMOTE_ROOT}/fullhq_inputs_v1/node${NODE_RANK}"
MIRROR="${SCALE_MIRROR_ROOT}/fullhq_inputs_v1/node${NODE_RANK}"
"${PYTHON_BIN}" "${IO}" publish --source "${INPUT_LOG}" --root "${CURRENT}" --root "${MIRROR}" --watch &
SYNC_PID=$!
finish() {
  local code=$?
  trap - EXIT
  kill "${SYNC_PID}" 2>/dev/null || true
  wait "${SYNC_PID}" 2>/dev/null || true
  "${PYTHON_BIN}" - "${INPUT_LOG}" "${code}" <<'PY'
import sys,time
from pathlib import Path
from scripts.window_run_io import atomic
atomic(Path(sys.argv[1])/'launcher_exit.json',dict(exit_code=int(sys.argv[2]),unix_time=time.time()))
PY
  if ! "${PYTHON_BIN}" "${IO}" publish --source "${INPUT_LOG}" --root "${CURRENT}" --root "${MIRROR}"; then
    [[ "${code}" != 0 ]] || code=74
  fi
  exit "${code}"
}
trap finish EXIT
exec > >(tee -a "${INPUT_LOG}/pipeline.log") 2>&1
R7_CKPT="${R7_CKPT:-${SCALE_ROOT}/r7_t2_c192_v2/joint/checkpoint_best.pt}"
export R7_CKPT
ensure_local_checkpoint "${R7_CKPT}" "${SCALE_REMOTE_ROOT}/r7_t2_c192_v2/joint/checkpoint_best.pt" 'frozen R7' "${SCALE_MIRROR_ROOT}/r7_t2_c192_v2/joint/checkpoint_best.pt"
ensure_local_checkpoint "${SPATIALVID_METADATA}" "${SPATIALVID_METADATA_URL}" 'full HQ metadata'
"${PYTHON_BIN}" "${PROJECT}/scripts/prepare_fullhq.py" --metadata "${SPATIALVID_METADATA}" \
  --protocol "${PROJECT}/configs/spatialvid_domain_v2.json" --output "${META}"
cp "${META}/selection.json" "${INPUT_LOG}/selection.json"
if [[ "${FULLHQ_STAGE}" == prepare ]]; then exit 0; fi
if [[ "${FULLHQ_STAGE}" == all || "${FULLHQ_STAGE}" == probe || "${FULLHQ_STAGE}" == train ]]; then
  STAGE_LOG_FILE="" run_torchrun "${PROJECT}/scripts/probe_fullhq_memory.py" \
    --r7_ckpt "${R7_CKPT}" --output "${INPUT_LOG}" --limit_gib "${MEMORY_LIMIT_GIB:-52}"
  [[ "${FULLHQ_STAGE}" != probe ]] || exit 0
fi
if [[ "${FULLHQ_STAGE}" == all || "${FULLHQ_STAGE}" == text ]]; then
  # Text encoding: one T5 per node, never 8 copies on a node.
  T5_DIR="${T5_CKPT_DIR:-${WAN_CKPT_DIR:-}}"
  if [[ -z "${T5_DIR}" ]]; then
    for candidate in "${VGGAE_REF_ROOT}/Wan2.1-T2V-1.3B" "${VGGAE_REF_ROOT}/Wan2.1-I2V-14B-480P"; do
      if [[ -f "${candidate}/models_t5_umt5-xxl-enc-bf16.pth" ]]; then T5_DIR="${candidate}"; break; fi
    done
  fi
  [[ -f "${T5_DIR}/models_t5_umt5-xxl-enc-bf16.pth" ]] || { echo 'Set T5_CKPT_DIR to existing Wan directory containing UMT5 + tokenizer'; exit 2; }
  TEXT_ARGS=(--csv "${META}/text.csv" --annotation_root "${SPATIALVID_OBS_ROOT}/annotations/SpatialVID/annotations"
    --output "${TEXT}" --staging "${LOCAL_CACHE_ROOT}/fullhq_text_staging/node${NODE_RANK}" --wan_ckpt "${T5_DIR}")
  "${PYTHON_BIN}" -u "${PROJECT}/scripts/precompute_fullhq_text.py" "${TEXT_ARGS[@]}"
  run_distributed_barrier
  if [[ "${NODE_RANK}" == 0 ]]; then
    "${PYTHON_BIN}" -u "${PROJECT}/scripts/precompute_fullhq_text.py" "${TEXT_ARGS[@]}" --merge
  fi
  run_distributed_barrier
  [[ "${FULLHQ_STAGE}" != text ]] || exit 0
fi
if [[ "${FULLHQ_STAGE}" == all || "${FULLHQ_STAGE}" == cache ]]; then
  ensure_local_checkpoint "${STREAMVGGT_CKPT}" "${STREAMVGGT_URL:-}" 'StreamVGGT' "${STREAMVGGT_MIRROR_URL:-}"
  SHA=$("${PYTHON_BIN}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["csv_sha256"]["train.csv"])' "${META}/selection.json")
  export MOX_CACHE_WRITER_DIR="${LOCAL_CACHE_ROOT}/cache/fullhq_latent_writer"
  STAGE_LOG_FILE="" run_torchrun "${PROJECT}/cache_causal_video_latents.py" \
    --csv "${META}/train.csv" --csv_sha256 "${SHA}" --video_root "${SPATIALVID_VIDEO_ROOT}" \
    --encoder_ckpt "${STREAMVGGT_CKPT}" --r7_ckpt "${R7_CKPT}" --output_dir "${CACHE}/train" \
    --split train --partition_id 0 --num_partitions 1 --clips_per_video 4 --samples_per_tar 64 \
    --resume_cache --independent_anchor --window_ae_norm legacy --dtype fp16 \
    --num_workers "${CACHE_WORKERS:-2}" --skip_failed_data --max_data_failure_rate "${MAX_DATA_FAILURE_RATE:-0.05}"
  if [[ "${NODE_RANK}" == 0 ]]; then
    "${PYTHON_BIN}" "${PROJECT}/merge_latent_cache.py" --cache_dir "${CACHE}/train" \
      --expected_partitions 1 --max_failure_rate "${MAX_DATA_FAILURE_RATE:-0.05}"
  fi
  run_distributed_barrier
  [[ "${FULLHQ_STAGE}" != cache ]] || exit 0
fi
# Same reviewed street evaluation anchor set; reserve all historical eval/test IDs.
# It is a regression set, not a claim of full-HQ domain coverage.
export TRAIN_MANIFEST="${CACHE}/train/manifest.txt" TRAIN_STATS="${CACHE}/train/stats.pt"
OLD_EVAL="${PERSISTENT_OBS_ROOT}/cache_latents/r7_domain_single_t2v2_legacy_diag_v2/eval"
export EVAL_MANIFEST="${OLD_EVAL}/manifest.txt" EVAL_STATS="${OLD_EVAL}/stats.pt"
export AE_REFERENCE_FILE="${PROJECT}/configs/ae_reference_single_v1.json" MIN_AE_PSNR=22.3
export AE_VARIANT=t2v2 WINDOW_AE_NORM=legacy LARGE_RUN=1 NO_TEXT=0 TEXT_DIR="${TEXT}"
export WIDTH=1536 DEPTH=24 HEADS=24 BATCH_SIZE=1 ACCUM_STEPS=8
export MAX_STEPS=100000 WARMUP_STEPS=2000 LR=1e-4 WD=0.01 DTYPE=bf16 EMA_DECAY=0.999
export PREDICTION=x0 TIME_DISTRIBUTION=uniform TIME_SHIFT=1 LOSS_FLOOR=0.05
export AUX_LAYER=0 AUX_WEIGHT=0 MEMORIZE_CLIPS=0 TEXT_DROPOUT=0.1 TEXT_CFG="${TEXT_CFG:-3}"
export SAVE_EVERY=1000 EVAL_EVERY=2000 EVAL_CLIPS=32 PREVIEW_CLIPS=8
export SAMPLE_STEPS=64 SAMPLE_METHOD=euler SAMPLE_SEEDS=101,211 SEED=42 LOG_EVERY=10
export RESUME="${RESUME:-0}" STOP_AFTER_STEPS="${STOP_AFTER_STEPS:-0}"
export DOMAIN_PROTOCOL_FILE="${META}/selection.json"
bash "${SCRIPT_DIR}/29_train_window_diffusion.sh"
