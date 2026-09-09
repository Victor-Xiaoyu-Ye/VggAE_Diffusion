#!/bin/bash
# New full-window I2V training. Existing AE/cache/DDP/OBS infrastructure only.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"
export PYTHONPATH="${PROJECT}${PYTHONPATH:+:${PYTHONPATH}}"
AE_VARIANT="${AE_VARIANT:-t2v2}"
case "${AE_VARIANT}" in
  t2v2) R7_NAMESPACE=r7_t2_c192_v2 ;;
  t1geo112) R7_NAMESPACE=r7_t1_c192_geo112_tex80_probe_v1 ;;
  t1equal) R7_NAMESPACE=r7_t1_c192_probe_v1 ;;
  t2v3) R7_NAMESPACE=r7_t2_c192_v3 ;;
  *) echo "Unknown AE_VARIANT=${AE_VARIANT}" >&2; exit 2 ;;
esac
PREDICTION="${PREDICTION:-x0}"
if [[ "${AE_VARIANT}" == t2v2 ]]; then DEFAULT_NORM=legacy; else DEFAULT_NORM=framewise; fi
WINDOW_AE_NORM="${WINDOW_AE_NORM:-${DEFAULT_NORM}}"
WINDOW_NAMESPACE="${WINDOW_NAMESPACE:-r7_window_${AE_VARIANT}_${WINDOW_AE_NORM}_${PREDICTION}_diag_v2}"
[[ "${WINDOW_NAMESPACE}" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Unsafe namespace' >&2; exit 2; }
R7_CACHE_VERSION="${R7_CACHE_VERSION:-r7_window_${AE_VARIANT}_${WINDOW_AE_NORM}_diag_v2}"
CACHE_ROOT="${R7_CACHE_OBS_ROOT:-${PERSISTENT_OBS_ROOT}/cache_latents/${R7_CACHE_VERSION}}"
LOCAL_OUT="${SCALE_ROOT}/${WINDOW_NAMESPACE}"
CURRENT_OUT="${SCALE_REMOTE_ROOT}/${WINDOW_NAMESPACE}"
MIRROR_OUT="${SCALE_MIRROR_ROOT}/${WINDOW_NAMESPACE}"
R7_CKPT="${R7_CKPT:-${SCALE_ROOT}/${R7_NAMESPACE}/joint/checkpoint_best.pt}"
R7_CKPT_URL="${R7_CKPT_URL:-${SCALE_REMOTE_ROOT}/${R7_NAMESPACE}/joint/checkpoint_best.pt}"
R7_CKPT_MIRROR_URL="${R7_CKPT_MIRROR_URL:-${SCALE_MIRROR_ROOT}/${R7_NAMESPACE}/joint/checkpoint_best.pt}"
MASTER_PORT="${MASTER_PORT:-29790}"
configure_modelarts_distributed
require_scale_cluster
require_output_url
RESUME="${RESUME:-0}"
[[ "${RESUME}" == 0 || "${RESUME}" == 1 ]] || { echo 'RESUME must be 0 or 1'; exit 2; }
if [[ "${NODE_RANK}" == 0 && "${RESUME}" == 0 ]]; then
  "${PYTHON_BIN}" "${PROJECT}/scripts/window_run_io.py" guard \
    --root "${LOCAL_OUT}" --root "${CURRENT_OUT}" --root "${MIRROR_OUT}"
fi
# No writes until the fresh-output check has succeeded on node zero.
run_distributed_barrier
mkdir -p "${LOCAL_OUT}/logs/npu"
"${PYTHON_BIN}" - "${LOCAL_OUT}" <<'PY'
import sys,time,shutil
from pathlib import Path
from scripts.window_run_io import atomic
out=Path(sys.argv[1]);old=out/'launcher_exit.json'
if old.exists():
    archived=out/'launcher_history'/f'exit_{time.time_ns()}.json'
    archived.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old,archived)
atomic(old,dict(status='running',exit_code=None,unix_time=time.time()))
PY
export STAGE_LOG_FILE="${LOCAL_OUT}/logs/train_node${NODE_RANK}.log"
export ASCEND_PROCESS_LOG_PATH="${LOCAL_OUT}/logs/npu"
IO="${PROJECT}/scripts/window_run_io.py"
if [[ "${NODE_RANK}" == 0 ]]; then
  SYNC_CURRENT="${CURRENT_OUT}"; SYNC_MIRROR="${MIRROR_OUT}"
else
  SYNC_CURRENT="${CURRENT_OUT}/workers/node${NODE_RANK}"
  SYNC_MIRROR="${MIRROR_OUT}/workers/node${NODE_RANK}"
fi
"${PYTHON_BIN}" "${IO}" publish --source "${LOCAL_OUT}" \
  --root "${SYNC_CURRENT}" --root "${SYNC_MIRROR}" --watch --interval 60 &
SYNC_PID=$!
finish() {
  local code=$?
  trap - EXIT
  kill "${SYNC_PID}" 2>/dev/null || true
  wait "${SYNC_PID}" 2>/dev/null || true
  # Failure during staging/torchrun must be visible even if Python never started.
  "${PYTHON_BIN}" - "${LOCAL_OUT}" "${code}" <<'PY'
import sys,time
from pathlib import Path
from scripts.window_run_io import atomic
atomic(Path(sys.argv[1])/'launcher_exit.json',dict(status='exited',exit_code=int(sys.argv[2]),unix_time=time.time()))
PY
  if ! "${PYTHON_BIN}" "${IO}" publish --source "${LOCAL_OUT}" \
      --root "${SYNC_CURRENT}" --root "${SYNC_MIRROR}"; then
    echo 'Final double-write failed; see publication_status.json' >&2
    [[ "${code}" != 0 ]] || code=74
  fi
  exit "${code}"
}
trap finish EXIT
exec > >(tee -a "${STAGE_LOG_FILE}") 2>&1
ensure_local_checkpoint "${R7_CKPT}" "${R7_CKPT_URL}" "frozen historical AE" "${R7_CKPT_MIRROR_URL}"
if [[ "${WINDOW_DIAGNOSTIC_ONLY:-0}" == 1 ]]; then
  [[ "${RESUME}" == 0 ]] || { echo 'Diagnostics use fresh outputs, not training resume'; exit 2; }
  DIAG_FILE="${DIAGNOSTIC_CKPT_NAME:-checkpoint_latest.pt}"
  [[ "${DIAG_FILE}" =~ ^checkpoint_[a-zA-Z0-9_]+\.pt$ ]] || { echo 'Unsafe checkpoint name'; exit 2; }
  DIAG_CKPT="${LOCAL_CACHE_ROOT}/window_diagnostics/${DIAGNOSTIC_SOURCE}/${DIAG_FILE}"
  "${PYTHON_BIN}" "${IO}" stage --destination "${DIAG_CKPT}" \
    --root "${DIAGNOSTIC_CKPT_URL:-${SCALE_REMOTE_ROOT}/${DIAGNOSTIC_SOURCE}/${DIAG_FILE}}" \
    --root "${DIAGNOSTIC_CKPT_MIRROR_URL:-${SCALE_MIRROR_ROOT}/${DIAGNOSTIC_SOURCE}/${DIAG_FILE}}"
  STAGE_LOG_FILE="" run_torchrun "${PROJECT}/diagnose_window_diffusion.py" \
    --checkpoint "${DIAG_CKPT}" --r7_ckpt "${R7_CKPT}" --output_dir "${LOCAL_OUT}" \
    --manifest "${TRAIN_MANIFEST:-${CACHE_ROOT}/train/manifest.txt}" \
    --eval_manifest "${EVAL_MANIFEST:-${CACHE_ROOT}/eval/manifest.txt}" \
    --eval_stats "${EVAL_STATS:-${CACHE_ROOT}/eval/stats.pt}" \
    --expected_step "${EXPECTED_STEP:-6000}" --clips "${DIAGNOSTIC_CLIPS:-16}" \
    --previews "${PREVIEW_CLIPS:-4}" --mode "${DIAGNOSTIC_MODE:-standard}"
  exit 0
fi
EXTRA=()
if [[ "${NO_TEXT:-0}" == 1 ]]; then
  EXTRA+=(--no_text)
else
  # Sidecars are immutable inputs; use the existing stage 15 if not already built.
  TEXT_VERSION="${TEXT_EMBEDDING_VERSION:-umt5xxl_spatialvid_10k_v1}"
  TEXT_LOCAL="${LOCAL_CACHE_ROOT}/text_embeddings/${TEXT_VERSION}"
  "${PYTHON_BIN}" - "${TEXT_DIR:-${PERSISTENT_OBS_ROOT}/text_embeddings/${TEXT_VERSION}}" "${TEXT_LOCAL}" <<'PY'
import json,sys
from pathlib import Path
from utils.moxing_io import copy_file,read_text
from scripts.window_run_io import child
src,dst=sys.argv[1:]; dst=Path(dst); dst.mkdir(parents=True,exist_ok=True)
index=json.loads(read_text(child(src,'index.json')))
for name in ['index.json','empty_prompt.pt',*sorted(set(index.values()))]:
    if not (dst/name).is_file(): copy_file(child(src,name),str(dst/name))
copy_file(child(src,'_SUCCESS'),str(dst/'_SUCCESS'))
PY
  EXTRA+=(--text_dir "${TEXT_LOCAL}")
fi
if [[ "${RESUME}" == 1 ]]; then
  RESUME_LOCAL="${LOCAL_CACHE_ROOT}/window_resume/${WINDOW_NAMESPACE}/checkpoint.pt"
  "${PYTHON_BIN}" "${IO}" stage --destination "${RESUME_LOCAL}" \
    --root "${RESUME_URL:-${CURRENT_OUT}/checkpoint_latest.pt}" \
    --root "${RESUME_MIRROR_URL:-${MIRROR_OUT}/checkpoint_latest.pt}"
  EXTRA+=(--resume "${RESUME_LOCAL}")
  # Preserve prior history on a new node; required metrics cannot silently reset.
  if [[ "${NODE_RANK}" == 0 ]]; then
    for name in metrics.jsonl; do
      "${PYTHON_BIN}" "${IO}" stage --destination "${LOCAL_OUT}/${name}" \
        --root "${CURRENT_OUT}/${name}" --root "${MIRROR_OUT}/${name}"
    done
    # A checkpoint is saved before evaluation; a crash there legitimately has
    # no eval_samples.jsonl yet. Retain existing history whenever present.
    "${PYTHON_BIN}" "${IO}" stage --destination "${LOCAL_OUT}/eval_samples.jsonl" \
      --root "${CURRENT_OUT}/eval_samples.jsonl" --root "${MIRROR_OUT}/eval_samples.jsonl" \
      || echo 'No previous evaluation history recovered; recorded checkpoints remain authoritative.'
  fi
fi
EXTRA+=(--stop_after_steps "${STOP_AFTER_STEPS:-0}")
# Use one tee (the shell above), so run_torchrun's own tee is disabled here.
STAGE_LOG_FILE="" run_torchrun "${PROJECT}/train_r7_window_diffusion.py" \
  --manifest "${TRAIN_MANIFEST:-${CACHE_ROOT}/train/manifest.txt}" \
  --stats "${TRAIN_STATS:-${CACHE_ROOT}/train/stats.pt}" \
  --eval_manifest "${EVAL_MANIFEST:-${CACHE_ROOT}/eval/manifest.txt}" \
  --eval_stats "${EVAL_STATS:-${CACHE_ROOT}/eval/stats.pt}" \
  --r7_ckpt "${R7_CKPT}" --output_dir "${LOCAL_OUT}" \
  --ae_norm "${WINDOW_AE_NORM}" --min_ae_psnr "${MIN_AE_PSNR:-23.5}" \
  --prediction "${PREDICTION}" --time_shift "${TIME_SHIFT:-1}" --loss_floor "${LOSS_FLOOR:-0.05}" \
  --width "${WIDTH:-768}" --depth "${DEPTH:-12}" --heads "${HEADS:-12}" \
  --batch_size "${BATCH_SIZE:-1}" --accum_steps "${ACCUM_STEPS:-2}" \
  --max_steps "${MAX_STEPS:-6000}" --warmup_steps "${WARMUP_STEPS:-300}" \
  --lr "${LR:-1e-4}" --wd "${WD:-0.01}" --dtype "${DTYPE:-bf16}" \
  --ema_decay "${EMA_DECAY:-0.999}" --text_dropout "${TEXT_DROPOUT:-0.1}" \
  --eval_every "${EVAL_EVERY:-500}" --save_every "${SAVE_EVERY:-500}" \
  --log_every "${LOG_EVERY:-10}" --eval_clips "${EVAL_CLIPS:-16}" \
  --preview_clips "${PREVIEW_CLIPS:-4}" --sample_steps "${SAMPLE_STEPS:-64}" \
  --sample_method "${SAMPLE_METHOD:-euler}" --sample_seeds "${SAMPLE_SEEDS:-42,43}" \
  --shuffle_buffer "${SHUFFLE_BUFFER:-128}" --seed "${SEED:-42}" "${EXTRA[@]}"
"${PYTHON_BIN}" - "${LOCAL_OUT}" "${MAX_STEPS:-6000}" "${STOP_AFTER_STEPS:-0}" "${NODE_RANK}" "${NUM_NPUS}" <<'PY'
import json,sys
from pathlib import Path
out=Path(sys.argv[1]); maximum,pause,node,npu=map(int,sys.argv[2:])
stop=min(maximum,pause) if pause else maximum
for rank in range(node*npu,(node+1)*npu):
    status=json.loads((out/f'run_status_rank{rank:03d}.json').read_text())
    if status['step'] != stop or status['status'] != ('completed' if stop==maximum else 'paused'):
        raise RuntimeError(f'Trainer did not finish requested budget: {status}')
print('PASS: all local ranks reached the requested training budget')
PY
