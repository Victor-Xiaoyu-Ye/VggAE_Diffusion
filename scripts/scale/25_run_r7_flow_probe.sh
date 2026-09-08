#!/bin/bash
# Isolated frozen-R7 flow diagnostics; node 0/device 0 only, no collective waits.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/spatialvid.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"
configure_modelarts_distributed
require_output_url
if [[ "${NODE_RANK}" -ne 0 ]]; then
  printf 'Flow probe: node %s idle; node 0 owns all training/status.\n' "${NODE_RANK}"
  exit 0
fi
export PYTHONUNBUFFERED=1
STAGE="${STAGE:-n1}" # smoke | n1 | fixed_path | n16 | prefix
case "${STAGE}" in smoke|n1|fixed_path|n16|prefix) ;; *) printf 'Unknown STAGE=%s\n' "${STAGE}" >&2; exit 2;; esac
PREDICTION="${PREDICTION:-preconditioned}"
case "${PREDICTION}" in plain_x0|preconditioned|direct_velocity) ;; *) printf 'Select one valid PREDICTION.\n' >&2; exit 2;; esac
R7_NAMESPACE="${R7_NAMESPACE:-r7_t1_c192_geo112_tex80_probe_v1}"
FLOW_NAMESPACE="${FLOW_NAMESPACE:-r7_geo112_flow_probe_v2}"
[[ "${FLOW_NAMESPACE}" == *probe* || "${FLOW_NAMESPACE}" == *diag* ]] || {
  printf 'FLOW_NAMESPACE must contain probe or diag.\n' >&2; exit 2;
}
[[ "${SAMPLE_STEPS:-30}" =~ ^[1-9][0-9]*$ ]] || {
  printf 'SAMPLE_STEPS is one integer; use DIAGNOSTIC_SAMPLE_STEPS=60,120 for comparisons.\n' >&2; exit 2;
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

# Fixed-path success cannot promote; preserve the strict small-to-large ladder.
if [[ "${STAGE}" == n16 || "${STAGE}" == prefix ]]; then
  : "${N1_STATUS:?Set N1_STATUS to the selected random-noise n1 memory_status.json}"
  status=$(stage_resume_checkpoint "${N1_STATUS}" "${LOCAL_CACHE_ROOT}/resume/${FLOW_NAMESPACE}_n1_status.json")
  PYTHONPATH="${PROJECT}" "${PYTHON_BIN}" - "${status}" "${PREDICTION}" "${R7_CKPT}" <<'PY'
import json,sys
from utils.file_signature import sampled_file_signature
s=json.load(open(sys.argv[1]))
if not (s.get('schema')=='r7-prefix-flow-v2' and s.get('passed') is True
        and s.get('noise_mode')=='random' and s.get('all_train_evaluated') is True
        and s.get('total_train_clips')==1 and s.get('future_frames')==1
        and s.get('r7_signature')==sampled_file_signature(sys.argv[3])
        and s.get('prediction')==sys.argv[2]):
    raise SystemExit('Selected n1 random-noise generated-RGB memory gate has not passed')
PY
fi
if [[ "${STAGE}" == prefix ]]; then
  : "${PRIOR_STATUS:?Set preceding same-data-count short-prefix memory_status.json}"
  prior=$(stage_resume_checkpoint "${PRIOR_STATUS}" "${LOCAL_CACHE_ROOT}/resume/${FLOW_NAMESPACE}_prior_status.json")
  PYTHONPATH="${PROJECT}" "${PYTHON_BIN}" - "${prior}" "${PREDICTION}" "${R7_CKPT}" "${FUTURE_FRAMES:-0}" "${MAX_SAMPLES:-16}" <<'PY'
import json,sys
from utils.file_signature import sampled_file_signature
s=json.load(open(sys.argv[1])); frames=int(sys.argv[4]); count=int(sys.argv[5])
if not (frames in (2,4,8) and s.get('passed') is True and s.get('noise_mode')=='random'
        and s.get('schema')=='r7-prefix-flow-v2' and s.get('all_train_evaluated') is True
        and s.get('future_frames')==frames//2 and s.get('total_train_clips')==count
        and s.get('prediction')==sys.argv[2]
        and s.get('r7_signature')==sampled_file_signature(sys.argv[3])):
    raise SystemExit('Prefix expansion requires the preceding random-noise memory gate')
PY
fi

run_arm() (
  kind=$1; count=$2; frames=$3; steps=$4; mode=$5
  name="${STAGE}_${kind}_n${count}_f${frames}"
  output="${SCALE_ROOT}/${FLOW_NAMESPACE}/${name}"
  remote="${SCALE_REMOTE_ROOT}/${FLOW_NAMESPACE}/${name}"
  extra=()
  if [[ -n "${RESUME:-}" ]]; then
    checkpoint=$(stage_resume_checkpoint "${RESUME}" "${LOCAL_CACHE_ROOT}/resume/${FLOW_NAMESPACE}_${name}.pt")
    extra+=(--resume "${checkpoint}")
    mkdir -p "${output}"
    for history in metrics.jsonl eval_samples.jsonl denoising.jsonl; do
      if [[ ! -s "${output}/${history}" ]]; then
        "${PYTHON_BIN}" "${PROJECT}/scripts/moxing_transfer.py" \
          "${remote}/${history}" "${output}/${history}" || {
          printf 'Resume history unavailable: %s; use original namespace/source history.\n' "${history}" >&2; exit 1;
        }
      fi
    done
    # Extra diagnostics need not have run before the last committed checkpoint.
    if [[ ! -e "${output}/sampling_diagnostics.jsonl" ]]; then
      "${PYTHON_BIN}" "${PROJECT}/scripts/moxing_transfer.py" \
        "${remote}/sampling_diagnostics.jsonl" "${output}/sampling_diagnostics.jsonl" || true
    fi
  else
    PYTHONPATH="${PROJECT}" "${PYTHON_BIN}" - "${output}" "${remote}" <<'PY'
import pathlib,sys
from utils.moxing_io import is_remote_path
local,remote=sys.argv[1:]
names=('metrics.jsonl','run_status.json')
if any((pathlib.Path(local)/n).exists() for n in names) or list(pathlib.Path(local).glob('checkpoint*.pt')):
    raise SystemExit('Existing probe output: new namespace or explicit resume required')
if is_remote_path(remote):
    import moxing as mox
    if any(mox.file.exists(remote+'/'+n) for n in names):
        raise SystemExit('Existing durable output: new namespace or explicit resume required')
else:
    if any((pathlib.Path(remote)/n).exists() for n in names):
        raise SystemExit('Existing mounted output: new namespace or explicit resume required')
PY
  fi
  if [[ -n "${TEXT_EMBEDDING_DIR:-}" ]]; then
    extra+=(--text_embedding_dir "${TEXT_EMBEDDING_DIR}")
  elif [[ "${count}" -gt 1 && "${ALLOW_NO_TEXT:-0}" != 1 ]]; then
    printf 'n16/prefix requires TEXT_EMBEDDING_DIR or explicit ALLOW_NO_TEXT=1.\n' >&2; exit 2
  elif [[ "${ALLOW_NO_TEXT:-0}" == 1 ]]; then extra+=(--allow_no_text); fi
  [[ "${EVAL_LPIPS:-1}" == 1 ]] && extra+=(--eval_lpips)
  [[ "${EVAL_EMA:-1}" == 0 ]] && extra+=(--no-eval_ema)
  start_output_sync "${output}" "${remote}"
  trap 'stop_output_sync "${output}" "${remote}"' EXIT
  printf '[flow-launch] prediction=%s noise=%s max_steps=%s stop_after=%s output=%s\n' \
    "${kind}" "${mode}" "${steps}" "${STOP_AFTER_STEPS:-0}" "${output}" | tee -a "${STAGE_LOG_FILE}"
  set +e
  PYTHONPATH="${PROJECT}" "${PYTHON_BIN}" -u "${PROJECT}/train_r7_flow_probe.py" \
    --csv "${SPATIALVID_TRAIN_10K_CSV}" --eval_csv "${SPATIALVID_EVAL_CSV}" \
    --video_root "${SPATIALVID_VIDEO_ROOT}" --encoder_ckpt "${STREAMVGGT_CKPT}" \
    --r7_ckpt "${R7_CKPT}" --output_dir "${output}" --prediction "${kind}" --noise_mode "${mode}" \
    --future_frames "${frames}" --max_samples "${count}" \
    --eval_samples "${EVAL_SAMPLES:-16}" --train_eval_samples "${TRAIN_EVAL_SAMPLES:-16}" \
    --max_steps "${steps}" --stop_after_steps "${STOP_AFTER_STEPS:-0}" --batch_size "${BATCH_SIZE:-1}" \
    --hidden_dim "${HIDDEN_DIM:-384}" --depth "${DEPTH:-4}" --num_heads "${NUM_HEADS:-6}" \
    --lr "${LEARNING_RATE:-2e-4}" --wd "${WEIGHT_DECAY:-0.01}" \
    --warmup_steps "${WARMUP_STEPS:-100}" --std_floor "${STD_FLOOR:-1e-4}" \
    --ema_decay "${EMA_DECAY:-0.999}" --eval_every "${EVAL_EVERY:-100}" \
    --log_every "${LOG_EVERY:-10}" --save_every "${SAVE_EVERY:-500}" \
    --sample_steps "${SAMPLE_STEPS:-30}" --sample_seeds "${SAMPLE_SEEDS:-42,43,44,45}" \
    --diagnostic_sample_steps "${DIAGNOSTIC_SAMPLE_STEPS-60}" --oracle_starts "${ORACLE_STARTS-0.5,0.7,0.9}" \
    --denoising_times "${DENOISING_TIMES:-0,0.01,0.05,0.1,0.3,0.5,0.7,0.9,0.99}" \
    --diagnostic_every "${DIAGNOSTIC_EVERY:-500}" \
    --fixed_path_seed "${FIXED_PATH_SEED:-42}" --fixed_path_steps "${FIXED_PATH_STEPS:-8}" \
    --preview_clips "${PREVIEW_CLIPS:-1}" --num_workers "${NUM_WORKERS:-2}" \
    --dtype "${DTYPE:-bf16}" --seed "${SEED:-42}" "${extra[@]}" \
    2>&1 | tee -a "${STAGE_LOG_FILE}"
  codes=("${PIPESTATUS[@]}")
  set -e
  printf '[flow-exit] trainer=%s tee=%s\n' "${codes[0]}" "${codes[1]}" | tee -a "${STAGE_LOG_FILE}"
  # Python may be killed before writing its failure record. Preserve its last phase.
  PYTHONPATH="${PROJECT}" "${PYTHON_BIN}" - "${output}" "${codes[0]}" "${codes[1]}" <<'PY'
import json,pathlib,sys
from utils.flow_run_status import atomic_json
path=pathlib.Path(sys.argv[1])/'run_status.json'
s=json.loads(path.read_text()) if path.exists() else {'schema':'r7-flow-run-status-v1'}
s.update(trainer_exit_code=int(sys.argv[2]),tee_exit_code=int(sys.argv[3]))
if any(int(x) for x in sys.argv[2:]):
    s.update(status='failed',error=s.get('error','trainer/tee exited nonzero; inspect job log'))
atomic_json(path,s)
PY
  if [[ "${codes[0]}" -ne 0 ]]; then exit "${codes[0]}"; fi
  if [[ "${codes[1]}" -ne 0 ]]; then exit "${codes[1]}"; fi
  PYTHONPATH="${PROJECT}" "${PYTHON_BIN}" - "${output}" "${steps}" "${STOP_AFTER_STEPS:-0}" <<'PY'
import sys
from utils.flow_run_status import verify_run
print('[flow-verified]', verify_run(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))['status'])
PY
  # Fail if final publication fails; exit trap performs a best-effort retry.
  "${PYTHON_BIN}" "${PROJECT}/scripts/moxing_transfer.py" "${output}" "${remote}" --directory
)

case "${STAGE}" in
  smoke) run_arm "${PREDICTION}" 1 1 2 random ;;
  n1) run_arm "${PREDICTION}" 1 1 "${MAX_STEPS:-2000}" random ;;
  fixed_path) run_arm "${PREDICTION}" 1 1 "${MAX_STEPS:-2000}" fixed_path ;;
  n16) run_arm "${PREDICTION}" 16 1 "${MAX_STEPS:-4000}" random ;;
  prefix) run_arm "${PREDICTION}" "${MAX_SAMPLES:-16}" "${FUTURE_FRAMES}" "${MAX_STEPS:-4000}" random ;;
esac
printf 'Requested flow budget and final checkpoint verified. Quality is separate; inspect memory_status.json.\n'
