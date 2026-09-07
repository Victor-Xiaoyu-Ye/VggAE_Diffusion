#!/bin/bash
# Replay explicitly selected old checkpoint, never substitute a current run.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"
configure_modelarts_distributed
require_output_url
if [[ "${NODE_RANK}" -ne 0 ]]; then
  printf 'Read-only replay runs on node 0; other nodes exit without collectives.\n'
  exit 0
fi
: "${REPLAY_CHECKPOINT:?Set explicit old checkpoint path or OBS URL}"
: "${REPLAY_MANIFEST:?Set matching evaluation manifest}"
: "${TEXT_EMBEDDING_DIR:?Set local or OBS UMT5 sidecar}"
REPLAY_NAMESPACE="${REPLAY_NAMESPACE:-r7_wan_replay_diag_v1}"
[[ "${REPLAY_NAMESPACE}" == *diag* || "${REPLAY_NAMESPACE}" == *probe* ]] || exit 2
output="${SCALE_ROOT}/${REPLAY_NAMESPACE}"
remote="${SCALE_REMOTE_ROOT}/${REPLAY_NAMESPACE}"
PYTHONPATH="${PROJECT}" "${PYTHON_BIN}" - "${output}" "${remote}" <<'PY'
import pathlib,sys
from utils.moxing_io import is_remote_path
local,remote=sys.argv[1:]
if any((pathlib.Path(local)/p).exists() for p in ('metrics.jsonl','replay.json','summary.json')):
    raise SystemExit('Replay output exists; use a new namespace')
if is_remote_path(remote):
    import moxing as mox
    if mox.file.exists(remote+'/replay.json'):
        raise SystemExit('Durable replay already exists; use a new namespace')
PY
checkpoint=$(stage_resume_checkpoint "${REPLAY_CHECKPOINT}" "${LOCAL_CACHE_ROOT}/resume/${REPLAY_NAMESPACE}.pt")
extra=()
[[ -n "${REPLAY_EVAL_STATS:-}" ]] && extra+=(--eval_stats "${REPLAY_EVAL_STATS}")
if [[ -n "${REPLAY_R7_CKPT:-}" ]]; then
  codec=$(stage_resume_checkpoint "${REPLAY_R7_CKPT}" "${LOCAL_CACHE_ROOT}/resume/${REPLAY_NAMESPACE}_codec.pt")
  extra+=(--r7_ckpt "${codec}")
fi
if [[ -n "${REPLAY_DECODER_CKPT:-}" ]]; then
  decoder=$(stage_resume_checkpoint "${REPLAY_DECODER_CKPT}" "${LOCAL_CACHE_ROOT}/resume/${REPLAY_NAMESPACE}_decoder.pt")
  extra+=(--decoder_ckpt "${decoder}")
fi
[[ "${CONDITION_ABLATIONS:-0}" == 1 ]] && extra+=(--condition_ablations)
[[ "${ALLOW_MISSING_TEXT:-0}" == 1 ]] && extra+=(--allow_missing_text)
start_output_sync "${output}" "${remote}"
trap 'stop_output_sync "${output}" "${remote}"' EXIT
PYTHONPATH="${PROJECT}" "${PYTHON_BIN}" "${PROJECT}/evaluate_r7_wan_denoising.py" \
  --checkpoint "${checkpoint}" --wan_ckpt_dir "${WAN_T2V_13B_DIR:-${VGGAE_REF_ROOT}/Wan2.1-T2V-1.3B}" \
  --manifest "${REPLAY_MANIFEST}" --text_embedding_dir "${TEXT_EMBEDDING_DIR}" \
  --output_dir "${output}" --weights "${REPLAY_WEIGHTS:-model,ema}" \
  --times "${REPLAY_TIMES:-0.1,0.3,0.5,0.7,0.9,0.99}" \
  --cfg_scales "${CFG_SCALES:-1,3}" --sample_steps "${SAMPLE_STEPS:-30,60}" \
  --sampling_grids "${SAMPLING_GRIDS:-trained}" --seeds "${SAMPLE_SEEDS:-42}" \
  --eval_clips "${EVAL_CLIPS:-4}" --preview_clips "${PREVIEW_CLIPS:-1}" \
  --oracle_starts "${ORACLE_STARTS:-}" --dtype "${DTYPE:-bf16}" \
  "${extra[@]}" 2>&1 | tee -a "${STAGE_LOG_FILE}"
"${PYTHON_BIN}" "${PROJECT}/scripts/moxing_transfer.py" "${output}" "${remote}" --directory
printf 'Replay published. Compare model/EMA and analytical baselines before any retraining.\n'
