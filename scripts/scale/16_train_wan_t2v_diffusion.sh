#!/bin/bash
# R7 Wan2.1-T2V-1.3B full-parameter diffusion: fixed 1 context + 4 future
# chunks over the t2/c192 cache, x0-prediction flow matching, UMT5-xxl text
# conditioning from the precomputed sidecar (stage 15), and the decoder_robust
# checkpoint for eval-time RGB decode.
# Examples:
#   bash scripts/scale/16_train_wan_t2v_diffusion.sh
#   MODE=sample bash scripts/scale/16_train_wan_t2v_diffusion.sh
#   EXTEND=1 MAX_STEPS=24000 EXTENSION_LR=5e-6 bash scripts/scale/16_train_wan_t2v_diffusion.sh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/spatialvid.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"

MODE="${MODE:-train}"                         # train | sample
TEMPORAL_FACTOR="${TEMPORAL_FACTOR:-2}"
LATENT_DIM="${LATENT_DIM:-192}"
CONTEXT_CHUNKS="${CONTEXT_CHUNKS:-1}"
FUTURE_CHUNKS="${FUTURE_CHUNKS:-4}"
R7_NAMESPACE="${R7_NAMESPACE:-r7_t${TEMPORAL_FACTOR}_c${LATENT_DIM}_v3}"
R7_CACHE_VERSION="${R7_CACHE_VERSION:-${R7_NAMESPACE}_seq9_frame_channel_v1}"
R7_CACHE_OBS_ROOT="${R7_CACHE_OBS_ROOT:-${PERSISTENT_OBS_ROOT}/cache_latents/${R7_CACHE_VERSION}}"
TRAIN_MANIFEST="${R7_MANIFEST:-${R7_CACHE_OBS_ROOT}/train/manifest.txt}"
TRAIN_STATS="${R7_STATS:-${R7_CACHE_OBS_ROOT}/train/stats.pt}"
EVAL_MANIFEST="${R7_EVAL_MANIFEST:-${R7_CACHE_OBS_ROOT}/eval/manifest.txt}"
EVAL_STATS="${R7_EVAL_STATS:-${R7_CACHE_OBS_ROOT}/eval/stats.pt}"
R7_CKPT="${R7_CKPT:-${SCALE_ROOT}/${R7_NAMESPACE}/joint/checkpoint_best.pt}"
R7_CKPT_URL="${R7_CKPT_URL:-${SCALE_REMOTE_ROOT}/${R7_NAMESPACE}/joint/checkpoint_best.pt}"
R7_CKPT_MIRROR_URL="${R7_CKPT_MIRROR_URL:-${SCALE_MIRROR_ROOT}/${R7_NAMESPACE}/joint/checkpoint_best.pt}"
# The noise-robust decoder is mandatory for this stage: without it, generated
# latents decode to noise and the LPIPS-based best selection is meaningless.
DECODER_CKPT="${DECODER_CKPT:-${SCALE_ROOT}/${R7_NAMESPACE}/decoder_robust/checkpoint_best.pt}"
DECODER_CKPT_URL="${DECODER_CKPT_URL:-${SCALE_REMOTE_ROOT}/${R7_NAMESPACE}/decoder_robust/checkpoint_best.pt}"
DECODER_CKPT_MIRROR_URL="${DECODER_CKPT_MIRROR_URL:-${SCALE_MIRROR_ROOT}/${R7_NAMESPACE}/decoder_robust/checkpoint_best.pt}"

WAN_T2V_13B_DIR="${WAN_T2V_13B_DIR:-${VGGAE_REF_ROOT}/Wan2.1-T2V-1.3B}"
TEXT_EMBEDDING_VERSION="${TEXT_EMBEDDING_VERSION:-umt5xxl_spatialvid_10k_v1}"
TEXT_EMBEDDING_OBS_DIR="${TEXT_EMBEDDING_OBS_DIR:-${PERSISTENT_OBS_ROOT}/text_embeddings/${TEXT_EMBEDDING_VERSION}}"

DIFFUSION_NAMESPACE="${DIFFUSION_NAMESPACE:-}"
if [[ -z "${DIFFUSION_NAMESPACE}" ]]; then
  if [[ "${TEMPORAL_FACTOR}" -eq 1 ]]; then
    DIFFUSION_NAMESPACE="r7_wan13b_i2v_anchor_memory_ctx1_fut8_v1"
  else
    DIFFUSION_NAMESPACE="r7_wan13b_i2v_anchor_memory_ctx1_fut4_v2"
  fi
fi
OUTPUT_DIR="${SCALE_ROOT}/${DIFFUSION_NAMESPACE}"
REMOTE_OUTPUT_DIR="${SCALE_REMOTE_ROOT}/${DIFFUSION_NAMESPACE}"
MIRROR_OUTPUT_DIR="${SCALE_MIRROR_ROOT}/${DIFFUSION_NAMESPACE}"
RESUME="${RESUME:-}"
AUTO_RESUME="${AUTO_RESUME:-1}"
EXTEND="${EXTEND:-0}"
MAX_STEPS="${MAX_STEPS:-16000}"
ADAPTER_LR="${ADAPTER_LR:-1e-4}"
WAN_LR="${WAN_LR:-1e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
WAN_FREEZE_STEPS="${WAN_FREEZE_STEPS:-500}"
EXTENSION_LR="${EXTENSION_LR:-0}"
EXTENSION_WARMUP_STEPS="${EXTENSION_WARMUP_STEPS:-200}"
TIME_SHIFT_ALPHA="${TIME_SHIFT_ALPHA:-3.0}"
TEXT_DROP_PROB="${TEXT_DROP_PROB:-0.1}"
CFG_SCALE="${CFG_SCALE:-3.0}"
PREVIEW_CLIPS="${PREVIEW_CLIPS:-1}"
PREVIEW_FPS="${PREVIEW_FPS:-8}"
ENFORCE_QUALITY_GUARDS="${ENFORCE_QUALITY_GUARDS:-0}"
GUARD_MOTION_RATIO_MIN="${GUARD_MOTION_RATIO_MIN:-0.55}"
GUARD_MOTION_RATIO_MAX="${GUARD_MOTION_RATIO_MAX:-1.50}"
GUARD_MOTION_COSINE_CHUNK3="${GUARD_MOTION_COSINE_CHUNK3:-0.45}"
GUARD_MOTION_COSINE_CHUNK4="${GUARD_MOTION_COSINE_CHUNK4:-0.35}"
GUARD_EXPANDED_GEO_COSINE="${GUARD_EXPANDED_GEO_COSINE:-0.10}"
HORIZON_WEIGHTS="${HORIZON_WEIGHTS:-}"
if [[ -z "${HORIZON_WEIGHTS}" ]]; then
  if [[ "${TEMPORAL_FACTOR}" -eq 1 ]]; then
    HORIZON_WEIGHTS="1,1.5,2,3,4,5,6,7"   # 8 future chunks
  else
    HORIZON_WEIGHTS="1,1.5,2,3"           # 4 future chunks
  fi
fi
echo "DI_throughput reports raw tokens/s/npu"
MASTER_PORT="${MASTER_PORT:-29685}"

[[ "${MODE}" == "train" || "${MODE}" == "sample" ]] || { echo "MODE must be train or sample." >&2; exit 2; }
[[ "${LATENT_DIM}" -eq 192 && "${CONTEXT_CHUNKS}" -eq 1 && \
   ( ( "${TEMPORAL_FACTOR}" -eq 2 && "${FUTURE_CHUNKS}" -eq 4 ) || \
     ( "${TEMPORAL_FACTOR}" -eq 1 && "${FUTURE_CHUNKS}" -eq 8 ) ) ]] || {
  echo "Diffusion contract is t2/c192/context1/future4 or t1/c192/context1/future8." >&2; exit 2;
}
case "$(basename "${WAN_T2V_13B_DIR}")" in
  *14B*) echo "This stage is restricted to Wan2.1-T2V-1.3B (14B OOMs under replicated DDP)." >&2; exit 2;;
esac
if [[ "${EXTEND}" == 1 ]]; then
  "${PYTHON_BIN}" -c "import sys; sys.exit(0 if float('${EXTENSION_LR}') > 0 else 1)" || {
    echo "EXTEND=1 requires EXTENSION_LR > 0." >&2; exit 2;
  }
fi

configure_modelarts_distributed
require_scale_cluster
require_output_url
require_dir "${WAN_T2V_13B_DIR}" "Wan2.1-T2V-1.3B checkpoint directory"
require_file "${WAN_T2V_13B_DIR}/config.json" "Wan DiT config.json"
ls "${WAN_T2V_13B_DIR}"/diffusion_pytorch_model*.safetensors >/dev/null 2>&1 || {
  echo "Missing diffusion_pytorch_model*.safetensors in ${WAN_T2V_13B_DIR}" >&2; exit 1;
}
mkdir -p "${OUTPUT_DIR}" "${LOCAL_CACHE_ROOT}/resume" \
  "${LOCAL_CACHE_ROOT}/sample_inputs" "${LOCAL_CACHE_ROOT}/r7_inputs/${R7_CACHE_VERSION}" \
  "${LOCAL_CACHE_ROOT}/text_embeddings"
if [[ "${NODE_RANK}" -ne 0 ]]; then rm -f "${R7_CKPT}" "${DECODER_CKPT}"; fi
ensure_local_checkpoint "${R7_CKPT}" "${R7_CKPT_URL}" \
  "accepted R7 checkpoint" "${R7_CKPT_MIRROR_URL}"
ensure_local_checkpoint "${DECODER_CKPT}" "${DECODER_CKPT_URL}" \
  "decoder_robust checkpoint" "${DECODER_CKPT_MIRROR_URL}"

# Text-embedding sidecar: stage the whole directory once per node.
LOCAL_TEXT_EMBEDDING_DIR="${LOCAL_CACHE_ROOT}/text_embeddings/${TEXT_EMBEDDING_VERSION}"
if [[ ! -s "${LOCAL_TEXT_EMBEDDING_DIR}/_SUCCESS" ]]; then
  echo "Staging text embeddings from ${TEXT_EMBEDDING_OBS_DIR}"
  # Keyed per-file reads: copy_parallel directory staging is silently
  # unreliable for many-object prefixes (same failure as the annotations
  # tree). Pull index/empty/_SUCCESS explicitly, then every indexed shard,
  # and only then write the local _SUCCESS marker.
  PYTHONPATH="${PROJECT}" "${PYTHON_BIN}" - \
    "${TEXT_EMBEDDING_OBS_DIR}" "${LOCAL_TEXT_EMBEDDING_DIR}" <<'PY'
import json, os, sys
from utils.moxing_io import copy_file, join_remote

remote, local = sys.argv[1], sys.argv[2]
os.makedirs(local, exist_ok=True)
for name in ("index.json", "empty_prompt.pt"):
    copy_file(join_remote(remote, name), os.path.join(local, name))
with open(os.path.join(local, "index.json"), encoding="utf-8") as stream:
    index = json.load(stream)
shards = sorted(set(index.values()))
for count, shard in enumerate(shards, 1):
    copy_file(join_remote(remote, shard), os.path.join(local, shard))
    if count % 5 == 0 or count == len(shards):
        print(f"text-embedding shards staged: {count}/{len(shards)}")
copy_file(join_remote(remote, "_SUCCESS"), os.path.join(local, "_SUCCESS"))
print(f"text-embedding sidecar staged: {len(shards)} shards + "
      f"index/empty/_SUCCESS -> {local}")
PY
fi
require_file "${LOCAL_TEXT_EMBEDDING_DIR}/_SUCCESS" "text-embedding sidecar marker"

stage_cache_input() {
  local source=$1 local_name=$2
  case "${source}" in
    obs://*|s3://*)
      local destination="${LOCAL_CACHE_ROOT}/r7_inputs/${R7_CACHE_VERSION}/${local_name}"
      rm -f "${destination}"
      stage_resume_checkpoint "${source}" "${destination}"
      ;;
    *) require_file "${source}" "R7 ${local_name}"; printf '%s' "${source}";;
  esac
}
# Best selection is decoded-RGB-LPIPS based; fail rather than silently falling
# back to a latent-only best when LPIPS is unavailable.
"${PYTHON_BIN}" -c "import lpips" || {
  echo "The Wan T2V diffusion stage requires the lpips package." >&2; exit 1;
}

LOCAL_EVAL_STATS=$(stage_cache_input "${EVAL_STATS}" eval_stats.pt)

if [[ "${MODE}" == "sample" ]]; then
  LOCAL_TRAIN_MANIFEST=$(stage_cache_input "${TRAIN_MANIFEST}" train_manifest.txt)
  LOCAL_TRAIN_STATS=$(stage_cache_input "${TRAIN_STATS}" train_stats.pt)
  LOCAL_EVAL_MANIFEST=$(stage_cache_input "${EVAL_MANIFEST}" eval_manifest.txt)
  SAMPLE_CKPT="${SAMPLE_CKPT:-${OUTPUT_DIR}/checkpoint_best.pt}"
  SAMPLE_CKPT_URL="${SAMPLE_CKPT_URL:-${REMOTE_OUTPUT_DIR}/checkpoint_best.pt}"
  SAMPLE_CKPT_MIRROR_URL="${SAMPLE_CKPT_MIRROR_URL:-${MIRROR_OUTPUT_DIR}/checkpoint_best.pt}"
  LOCAL_SAMPLE_CKPT="${LOCAL_CACHE_ROOT}/resume/${DIFFUSION_NAMESPACE}_best.pt"
  if [[ "${SAMPLE_CKPT}" == obs://* || "${SAMPLE_CKPT}" == s3://* ]]; then
    LOCAL_SAMPLE_CKPT=$(stage_resume_checkpoint "${SAMPLE_CKPT}" "${LOCAL_SAMPLE_CKPT}")
  elif [[ -s "${SAMPLE_CKPT}" ]]; then
    LOCAL_SAMPLE_CKPT="${SAMPLE_CKPT}"
  else
    rm -f "${LOCAL_SAMPLE_CKPT}"
    ensure_local_checkpoint "${LOCAL_SAMPLE_CKPT}" "${SAMPLE_CKPT_URL}" \
      "best diffusion checkpoint" "${SAMPLE_CKPT_MIRROR_URL}"
  fi
  SAMPLE_DIR="${OUTPUT_DIR}/samples_manual"
  mkdir -p "${SAMPLE_DIR}"
  start_output_sync "${OUTPUT_DIR}" "${REMOTE_OUTPUT_DIR}"
  trap 'stop_output_sync "${OUTPUT_DIR}" "${REMOTE_OUTPUT_DIR}"' EXIT
  if [[ "${NODE_RANK}" -eq 0 ]]; then
    PYTHONPATH="${PROJECT}" "${PYTHON_BIN}" \
      "${PROJECT}/sample_causal_wan_video_diffusion.py" \
      --checkpoint "${LOCAL_SAMPLE_CKPT}" --wan_ckpt_dir "${WAN_T2V_13B_DIR}" \
      --manifest "${LOCAL_EVAL_MANIFEST}" \
      --sample_index "${SAMPLE_INDEX:-0}" \
      --stats "${LOCAL_TRAIN_STATS}" --r7_ckpt "${R7_CKPT}" \
      --decoder_ckpt "${DECODER_CKPT}" \
      --text_embedding_dir "${LOCAL_TEXT_EMBEDDING_DIR}" \
      --cfg_scale "${CFG_SCALE}" \
      --output "${SAMPLE_OUTPUT:-${SAMPLE_DIR}/best_ema.pt}" \
      --sample_steps "${SAMPLE_STEPS:-30}" --weights ema \
      --preview_fps "${PREVIEW_FPS}" \
      --decode_chunk_size "${DECODE_CHUNK_SIZE:-0}" --dtype bf16
  fi
  exit 0
fi

LOCAL_TRAIN_MANIFEST=$(stage_cache_input "${TRAIN_MANIFEST}" train_manifest.txt)
LOCAL_TRAIN_STATS=$(stage_cache_input "${TRAIN_STATS}" train_stats.pt)
LOCAL_EVAL_MANIFEST=$(stage_cache_input "${EVAL_MANIFEST}" eval_manifest.txt)
# LatentShardDataset assigns whole tar shards across rank*worker consumers.
NUM_WORKERS_VALUE="${NUM_WORKERS:-1}"
SHARD_COUNT=$("${PYTHON_BIN}" - "${LOCAL_TRAIN_MANIFEST}" <<'PY'
import sys
with open(sys.argv[1], encoding='utf-8') as stream:
    print(sum(1 for line in stream if line.strip() and not line.startswith('#')))
PY
)
if (( SHARD_COUNT < WORLD_SIZE * NUM_WORKERS_VALUE )); then
  echo "R7 cache has ${SHARD_COUNT} shards but diffusion needs at least $((WORLD_SIZE * NUM_WORKERS_VALUE)) for WORLD_SIZE=${WORLD_SIZE}, NUM_WORKERS=${NUM_WORKERS_VALUE}." >&2
  exit 1
fi
if [[ "${AUTO_RESUME}" == 1 || -n "${RESUME}" ]]; then
  RESUME=$(resolve_resume_checkpoint "${RESUME}" \
    "${OUTPUT_DIR}/checkpoint_latest.pt" "${REMOTE_OUTPUT_DIR}/checkpoint_latest.pt" \
    "${LOCAL_CACHE_ROOT}/resume/${DIFFUSION_NAMESPACE}.pt" \
    "${MIRROR_OUTPUT_DIR}/checkpoint_latest.pt")
fi
if [[ "${EXTEND}" == 1 && -z "${RESUME}" ]]; then
  echo "EXTEND=1 requires same-namespace completed full-state resume." >&2; exit 1
fi
# Restore metrics before start_output_sync to preserve durable history.
if [[ -n "${RESUME}" && ! -s "${OUTPUT_DIR}/metrics.jsonl" ]]; then
  for source in "${RESUME_METRICS_URL:-${REMOTE_OUTPUT_DIR}/metrics.jsonl}" \
                "${MIRROR_OUTPUT_DIR}/metrics.jsonl"; do
    "${PYTHON_BIN}" "${PROJECT}/scripts/moxing_transfer.py" \
      "${source}" "${OUTPUT_DIR}/metrics.jsonl" 2>/dev/null && break || true
  done
fi
EXTRA_ARGS=(); [[ -n "${RESUME}" ]] && EXTRA_ARGS+=(--resume "${RESUME}")
[[ "${ENFORCE_QUALITY_GUARDS}" == 1 ]] && EXTRA_ARGS+=(--enforce_quality_guards)
start_output_sync "${OUTPUT_DIR}" "${REMOTE_OUTPUT_DIR}"
trap 'stop_output_sync "${OUTPUT_DIR}" "${REMOTE_OUTPUT_DIR}"' EXIT
run_torchrun "${PROJECT}/train_causal_wan_video_diffusion.py" \
  --manifest "${LOCAL_TRAIN_MANIFEST}" --stats "${LOCAL_TRAIN_STATS}" \
  --eval_manifest "${LOCAL_EVAL_MANIFEST}" --eval_stats "${LOCAL_EVAL_STATS}" \
  --wan_ckpt_dir "${WAN_T2V_13B_DIR}" \
  --r7_ckpt "${R7_CKPT}" --decoder_ckpt "${DECODER_CKPT}" \
  --text_embedding_dir "${LOCAL_TEXT_EMBEDDING_DIR}" \
  --require_rgb_lpips --output_dir "${OUTPUT_DIR}" \
  --latent_dim "${LATENT_DIM}" --latent_grid 18 \
  --context_chunks "${CONTEXT_CHUNKS}" --future_chunks "${FUTURE_CHUNKS}" \
  --temporal_factor "${TEMPORAL_FACTOR}" \
  --normalization_mode zscore \
  --time_shift_alpha "${TIME_SHIFT_ALPHA}" \
  --batch_size "${BATCH_SIZE:-4}" --accum_steps "${ACCUM_STEPS:-1}" \
  --max_steps "${MAX_STEPS}" \
  --adapter_lr "${ADAPTER_LR}" --wan_lr "${WAN_LR}" --wd "${WEIGHT_DECAY:-1e-2}" \
  --warmup_steps "${WARMUP_STEPS}" --wan_freeze_steps "${WAN_FREEZE_STEPS}" \
  --extension_lr "${EXTENSION_LR}" \
  --extension_warmup_steps "${EXTENSION_WARMUP_STEPS}" \
  --ema_decay "${EMA_DECAY:-0.9999}" \
  --text_drop_prob "${TEXT_DROP_PROB}" --cfg_scale "${CFG_SCALE}" \
  --lambda_motion "${LAMBDA_MOTION:-0.10}" --lambda_accel "${LAMBDA_ACCEL:-0.05}" \
  --lambda_geo_motion "${LAMBDA_GEO_MOTION:-0.05}" \
  --lambda_motion_cosine "${LAMBDA_MOTION_COSINE:-0.10}" \
  --lambda_motion_magnitude "${LAMBDA_MOTION_MAGNITUDE:-0.05}" \
  --horizon_weights "${HORIZON_WEIGHTS}" \
  --rollout_window "${ROLLOUT_WINDOW:-2}" --rollout_overlap "${ROLLOUT_OVERLAP:-1}" \
  --scheduled_context_start "${SCHEDULED_CONTEXT_START:-4000}" \
  --scheduled_context_ramp "${SCHEDULED_CONTEXT_RAMP:-4000}" \
  --scheduled_context_max "${SCHEDULED_CONTEXT_MAX:-0.25}" \
  --aux_warmup_steps "${AUX_WARMUP_STEPS:-1000}" \
  --aux_ramp_steps "${AUX_RAMP_STEPS:-1000}" --aux_t_min "${AUX_T_MIN:-0.60}" \
  --num_workers "${NUM_WORKERS_VALUE}" --shuffle_buffer "${SHUFFLE_BUFFER:-256}" \
  --eval_clips "${EVAL_CLIPS:-64}" --sample_clips "${SAMPLE_CLIPS:-1}" \
  --preview_clips "${PREVIEW_CLIPS}" --preview_fps "${PREVIEW_FPS}" \
  --sample_steps "${SAMPLE_STEPS:-30}" --decode_chunk_size "${DECODE_CHUNK_SIZE:-0}" \
  --guard_motion_ratio_min "${GUARD_MOTION_RATIO_MIN}" \
  --guard_motion_ratio_max "${GUARD_MOTION_RATIO_MAX}" \
  --guard_motion_cosine_chunk3 "${GUARD_MOTION_COSINE_CHUNK3}" \
  --guard_motion_cosine_chunk4 "${GUARD_MOTION_COSINE_CHUNK4}" \
  --guard_expanded_geo_cosine "${GUARD_EXPANDED_GEO_COSINE}" \
  --early_stop_min_steps "${EARLY_STOP_MIN_STEPS:-6000}" \
  --patience "${EARLY_STOP_PATIENCE:-6}" \
  --log_every "${LOG_EVERY:-50}" --eval_every "${EVAL_EVERY:-500}" \
  --save_every "${SAVE_EVERY:-500}" --dtype bf16 "${EXTRA_ARGS[@]}"
