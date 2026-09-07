#!/bin/bash
# Diagnostic only: one accelerator on node 0. Other nodes exit; no collective waits.
# STAGE=n1 compares both heads; n16/prefix requires a selected successful n1 arm.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/spatialvid.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"
configure_modelarts_distributed
require_output_url
if [[ "${NODE_RANK}" -ne 0 ]]; then
  printf 'R7 flow probe uses only node 0/device 0; node %s exits without a barrier.\n' "${NODE_RANK}"
  exit 0
fi
STAGE="${STAGE:-n1}" # smoke | n1 | n16 | prefix
case "${STAGE}" in smoke|n1|n16|prefix) ;; *) printf 'Unknown STAGE=%s\n' "${STAGE}" >&2; exit 2;; esac
R7_NAMESPACE="${R7_NAMESPACE:-r7_t1_c192_geo112_tex80_probe_v1}"
FLOW_NAMESPACE="${FLOW_NAMESPACE:-r7_geo112_flow_probe_v1}"
[[ "${FLOW_NAMESPACE}" == *probe* || "${FLOW_NAMESPACE}" == *diag* ]] || {
  printf 'FLOW_NAMESPACE must contain probe or diag.\n' >&2; exit 2;
}
SPATIALVID_OFT_ROOT="${SPATIALVID_OFT_ROOT:-${PERSISTENT_OBS_ROOT}/spatial-vid-hq-oft}"
SPATIALVID_METADATA_URL="${SPATIALVID_OFT_ROOT}/data/train/SpatialVID_HQ_metadata.csv"
SPATIALVID_VIDEO_ROOT="${SPATIALVID_OFT_ROOT}/videos/SpatialVid/HQ/videos"
SPATIALVID_METADATA="${LOCAL_CACHE_ROOT}/metadata/SpatialVID_HQ_metadata_oft.csv"
SPATIALVID_SPLIT_DIR="${RUN_ROOT}/metadata/spatialvid_oft_seed${SPLIT_SEED}"
SPATIALVID_TRAIN_10K_CSV="${SPATIALVID_SPLIT_DIR}/train_10k.csv"
SPATIALVID_EVAL_CSV="${SPATIALVID_SPLIT_DIR}/eval.csv"
ensure_spatialvid_subset_splits
R7_CKPT="${R7_CKPT:-${SCALE_ROOT}/${R7_NAMESPACE}/joint/checkpoint_best.pt}"
ensure_local_checkpoint "${R7_CKPT}" \
  "${R7_CKPT_URL:-${SCALE_REMOTE_ROOT}/${R7_NAMESPACE}/joint/checkpoint_best.pt}" \
  'frozen geo112 R7 checkpoint' \
  "${R7_CKPT_MIRROR_URL:-${SCALE_MIRROR_ROOT}/${R7_NAMESPACE}/joint/checkpoint_best.pt}"
require_file "${STREAMVGGT_CKPT}" 'StreamVGGT encoder'

run_arm() (
  kind=$1; count=$2; frames=$3; steps=$4
  name="${STAGE}_${kind}_n${count}_f${frames}"
  output="${SCALE_ROOT}/${FLOW_NAMESPACE}/${name}"
  remote="${SCALE_REMOTE_ROOT}/${FLOW_NAMESPACE}/${name}"
  extra=()
  if [[ -n "${RESUME:-}" ]]; then
    [[ "${PREDICTION:-both}" != both ]] || { printf 'Resume requires one PREDICTION.\n' >&2; exit 2; }
    checkpoint=$(stage_resume_checkpoint "${RESUME}" "${LOCAL_CACHE_ROOT}/resume/${FLOW_NAMESPACE}_${name}.pt")
    extra+=(--resume "${checkpoint}")
    # Preserve durable history before starting the directory watcher.
    for history in metrics.jsonl eval_samples.jsonl denoising.jsonl; do
      if [[ ! -s "${output}/${history}" ]]; then
        "${PYTHON_BIN}" "${PROJECT}/scripts/moxing_transfer.py" \
          "${remote}/${history}" "${output}/${history}" || {
          printf 'Resume history unavailable: %s. Use the original output namespace.\n' "${history}" >&2; exit 1;
        }
      fi
    done
  else
    PYTHONPATH="${PROJECT}" "${PYTHON_BIN}" - "${output}" "${remote}" <<'PY'
import pathlib,sys
from utils.moxing_io import is_remote_path
local,remote=sys.argv[1:]
if (pathlib.Path(local)/'metrics.jsonl').exists() or list(pathlib.Path(local).glob('checkpoint*.pt')):
    raise SystemExit('Existing probe output: use a new namespace or explicit resume')
if is_remote_path(remote):
    import moxing as mox
    if mox.file.exists(remote+'/metrics.jsonl'):
        raise SystemExit('Existing durable output: use new namespace or explicit resume')
PY
  fi
  if [[ -n "${TEXT_EMBEDDING_DIR:-}" ]]; then
    extra+=(--text_embedding_dir "${TEXT_EMBEDDING_DIR}")
  elif [[ "${count}" -gt 1 && "${ALLOW_NO_TEXT:-0}" != 1 ]]; then
    printf 'n16/prefix requires TEXT_EMBEDDING_DIR (local or OBS sidecar).\n' >&2; exit 2
  elif [[ "${ALLOW_NO_TEXT:-0}" == 1 ]]; then extra+=(--allow_no_text); fi
  [[ "${EVAL_LPIPS:-1}" == 1 ]] && extra+=(--eval_lpips)
  [[ "${EVAL_EMA:-1}" == 0 ]] && extra+=(--no-eval_ema)
  start_output_sync "${output}" "${remote}"
  trap 'stop_output_sync "${output}" "${remote}"' EXIT
  PYTHONPATH="${PROJECT}" "${PYTHON_BIN}" "${PROJECT}/train_r7_flow_probe.py" \
    --csv "${SPATIALVID_TRAIN_10K_CSV}" --eval_csv "${SPATIALVID_EVAL_CSV}" \
    --video_root "${SPATIALVID_VIDEO_ROOT}" --encoder_ckpt "${STREAMVGGT_CKPT}" \
    --r7_ckpt "${R7_CKPT}" --output_dir "${output}" --prediction "${kind}" \
    --future_frames "${frames}" --max_samples "${count}" \
    --eval_samples "${EVAL_SAMPLES:-16}" --train_eval_samples "${TRAIN_EVAL_SAMPLES:-16}" \
    --max_steps "${steps}" --stop_after_steps "${STOP_AFTER_STEPS:-0}" --batch_size "${BATCH_SIZE:-1}" \
    --hidden_dim "${HIDDEN_DIM:-384}" --depth "${DEPTH:-4}" --num_heads "${NUM_HEADS:-6}" \
    --lr "${LEARNING_RATE:-2e-4}" --wd "${WEIGHT_DECAY:-0.01}" \
    --warmup_steps "${WARMUP_STEPS:-100}" --std_floor "${STD_FLOOR:-1e-4}" \
    --ema_decay "${EMA_DECAY:-0.999}" --eval_every "${EVAL_EVERY:-100}" \
    --log_every "${LOG_EVERY:-10}" --save_every "${SAVE_EVERY:-500}" \
    --sample_steps "${SAMPLE_STEPS:-30}" --sample_seeds "${SAMPLE_SEEDS:-42,43,44,45}" \
    --preview_clips "${PREVIEW_CLIPS:-1}" --num_workers "${NUM_WORKERS:-2}" \
    --dtype "${DTYPE:-bf16}" --seed "${SEED:-42}" "${extra[@]}" \
    2>&1 | tee -a "${STAGE_LOG_FILE}"
  # Synchronous publication must succeed before considering the stage complete.
  "${PYTHON_BIN}" "${PROJECT}/scripts/moxing_transfer.py" "${output}" "${remote}" --directory
)

if [[ "${STAGE}" == n16 || "${STAGE}" == prefix ]]; then
  [[ "${PREDICTION:-}" == plain_x0 || "${PREDICTION:-}" == preconditioned ]] || {
    printf 'Select PREDICTION=plain_x0 or preconditioned after reviewing n1.\n' >&2; exit 2;
  }
  [[ -n "${N1_STATUS:-}" ]] || { printf 'N1_STATUS must explicitly identify the selected n1 memory_status.json.\n' >&2; exit 2; }
  status=$(stage_resume_checkpoint "${N1_STATUS}" "${LOCAL_CACHE_ROOT}/resume/${FLOW_NAMESPACE}_n1_status.json")
  PYTHONPATH="${PROJECT}" "${PYTHON_BIN}" - "${status}" "${PREDICTION}" "${R7_CKPT}" <<'PY'
import json,sys
from utils.file_signature import sampled_file_signature
s=json.load(open(sys.argv[1]))
if not (s.get('schema')=='r7-prefix-flow-v1' and s.get('passed') is True
        and s.get('all_train_evaluated') is True and s.get('total_train_clips')==1
        and s.get('future_frames')==1
        and s.get('r7_signature')==sampled_file_signature(sys.argv[3])
        and s.get('prediction')==sys.argv[2]):
    raise SystemExit('Selected n1 flow has not passed its generated-RGB memory gate')
PY
fi
if [[ "${STAGE}" == prefix ]]; then
  [[ -n "${PRIOR_STATUS:-}" ]] || { printf 'prefix requires PRIOR_STATUS from the preceding n16/shorter-prefix run.\n' >&2; exit 2; }
  prior=$(stage_resume_checkpoint "${PRIOR_STATUS}" "${LOCAL_CACHE_ROOT}/resume/${FLOW_NAMESPACE}_prior_status.json")
  PYTHONPATH="${PROJECT}" "${PYTHON_BIN}" - "${prior}" "${PREDICTION}" "${R7_CKPT}" "${FUTURE_FRAMES:-0}" "${MAX_SAMPLES:-16}" <<'PY'
import json,sys
from utils.file_signature import sampled_file_signature
s=json.load(open(sys.argv[1])); frames=int(sys.argv[4]); count=int(sys.argv[5])
if not (frames in (2,4,8) and s.get('passed') is True
        and s.get('schema')=='r7-prefix-flow-v1' and s.get('all_train_evaluated') is True
        and s.get('future_frames')==frames//2 and s.get('total_train_clips')==count
        and s.get('prediction')==sys.argv[2]
        and s.get('r7_signature')==sampled_file_signature(sys.argv[3])):
    raise SystemExit('Prefix expansion requires the preceding same-codec/data-count memory gate; review held-out previews separately')
PY
fi
predictions="${PREDICTION:-both}"
[[ "${predictions}" != both ]] || predictions='plain_x0 preconditioned'
for kind in ${predictions}; do
  case "${kind}" in plain_x0|preconditioned) ;; *) exit 2;; esac
  case "${STAGE}" in
    smoke) run_arm "${kind}" 1 1 2 ;;
    n1) run_arm "${kind}" 1 1 "${MAX_STEPS:-2000}" ;;
    n16) run_arm "${kind}" 16 1 "${MAX_STEPS:-4000}" ;;
    prefix)
      [[ "${FUTURE_FRAMES:-}" == 2 || "${FUTURE_FRAMES:-}" == 4 || "${FUTURE_FRAMES:-}" == 8 ]] || exit 2
      run_arm "${kind}" "${MAX_SAMPLES:-16}" "${FUTURE_FRAMES}" "${MAX_STEPS:-4000}" ;;
  esac
done
printf 'Probe stage complete. Inspect online/EMA, RAW/AE, per-seed and denoising reports; no production gate was written.\n'
