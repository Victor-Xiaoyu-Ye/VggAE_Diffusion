#!/bin/bash
# R7 v2 causal tokenizer wrapper, aligned to the trainer's init/resume,
# extension, early-stop, checkpoint-best, and acceptance-metric CLI.
# Examples:
#   PHASE=codec bash scripts/scale/11_train_causal_tokenizer.sh
#   PHASE=joint bash scripts/scale/11_train_causal_tokenizer.sh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/spatialvid.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"

# -------------------------- centrally editable contract -----------------------
PHASE="${PHASE:-codec}"                       # codec | joint
TEMPORAL_FACTOR="${TEMPORAL_FACTOR:-2}"
GEO_LATENT_DIM="${GEO_LATENT_DIM:-96}"
TEX_LATENT_DIM="${TEX_LATENT_DIM:-96}"
LATENT_DIM=$((GEO_LATENT_DIM + TEX_LATENT_DIM))
R7_NAMESPACE="${R7_NAMESPACE:-r7_t${TEMPORAL_FACTOR}_c${LATENT_DIM}_v2}"
RUN_NAME="${R7_NAMESPACE}/${PHASE}"
OUTPUT_DIR="${SCALE_ROOT}/${RUN_NAME}"
REMOTE_OUTPUT_DIR="${SCALE_REMOTE_ROOT}/${RUN_NAME}"
MIRROR_OUTPUT_DIR="${SCALE_MIRROR_ROOT}/${RUN_NAME}"

DUAL_AE_CKPT="${DUAL_AE_CKPT:-${VGGAE_REF_ROOT}/checkpoints/e5_dual_stream_r5.pt}"
DUAL_AE_CKPT_URL="${DUAL_AE_CKPT_URL:-}"
DUAL_AE_CKPT_MIRROR_URL="${DUAL_AE_CKPT_MIRROR_URL:-}"
# The converging R7 probe uses the same sparse OFT 10K mirror as E7/E9.
SPATIALVID_OFT_ROOT="${SPATIALVID_OFT_ROOT:-obs://yw-ads-training-gy1/data/external/personal/g00833899/y50046448/spatial-vid-hq-oft}"
SPATIALVID_METADATA_URL="${SPATIALVID_OFT_ROOT}/data/train/SpatialVID_HQ_metadata.csv"
SPATIALVID_VIDEO_ROOT="${SPATIALVID_OFT_ROOT}/videos/SpatialVid/HQ/videos"
SPATIALVID_METADATA="${LOCAL_CACHE_ROOT}/metadata/SpatialVID_HQ_metadata_oft.csv"
SPATIALVID_SPLIT_DIR="${RUN_ROOT}/metadata/spatialvid_oft_seed${SPLIT_SEED}"
SPATIALVID_TRAIN_10K_CSV="${SPATIALVID_SPLIT_DIR}/train_10k.csv"
SPATIALVID_EVAL_CSV="${SPATIALVID_SPLIT_DIR}/eval.csv"
SEQ_LEN="${SEQ_LEN:-9}"
BATCH_SIZE="${BATCH_SIZE:-1}"
ACCUM_STEPS="${ACCUM_STEPS:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_CLIPS="${EVAL_CLIPS:-32}"
LOG_EVERY="${LOG_EVERY:-50}"
EVAL_EVERY="${EVAL_EVERY:-500}"
SAVE_EVERY="${SAVE_EVERY:-500}"
AUTO_RESUME="${AUTO_RESUME:-1}"
RESUME="${RESUME:-}"
LEGACY_INIT_CKPT="${LEGACY_INIT_CKPT:-}"
EXTEND="${EXTEND:-0}"
MASTER_PORT="${MASTER_PORT:-29670}"

if [[ "${PHASE}" == "codec" ]]; then
  MAX_STEPS="${MAX_STEPS:-$([[ "${EXTEND}" == 1 ]] && printf 12000 || printf 6000)}"
  LEARNING_RATE="${LEARNING_RATE:-2e-4}"
  PRETRAINED_LR="${PRETRAINED_LR:-5e-5}"
  MIN_STEPS="${MIN_STEPS:-4000}"
  EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-4}"
else
  MAX_STEPS="${MAX_STEPS:-12000}"
  LEARNING_RATE="${LEARNING_RATE:-5e-5}"
  PRETRAINED_LR="${PRETRAINED_LR:-1e-5}"
  MIN_STEPS="${MIN_STEPS:-6000}"
  EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-6}"
fi
WARMUP_STEPS="${WARMUP_STEPS:-200}"
EXTENSION_LR="${EXTENSION_LR:-$([[ "${EXTEND}" == 1 ]] && printf 2e-5 || printf 0)}"
EXTENSION_WARMUP_STEPS="${EXTENSION_WARMUP_STEPS:-200}"
# -----------------------------------------------------------------------------

[[ "${PHASE}" == "codec" || "${PHASE}" == "joint" ]] || {
  echo "PHASE must be codec or joint, got ${PHASE}" >&2; exit 2;
}
[[ "${TEMPORAL_FACTOR}" -eq 2 && "${LATENT_DIM}" -eq 192 && "${SEQ_LEN}" -eq 9 ]] || {
  echo "Production R7 contract is t2/c192/seq9." >&2; exit 2;
}
if [[ "${EXTEND}" == 1 ]]; then
  [[ "${MAX_STEPS}" -eq 12000 ]] || { echo "An extension must target MAX_STEPS=12000." >&2; exit 2; }
  "${PYTHON_BIN}" -c "import sys; sys.exit(0 if float('${EXTENSION_LR}') > 0 else 1)" || {
    echo "EXTEND=1 requires EXTENSION_LR > 0." >&2; exit 2;
  }
elif [[ "${PHASE}" == "codec" && "${MAX_STEPS}" -gt 6000 ]]; then
  echo "The initial codec run is capped at 6000; use EXTEND=1 for 12000." >&2; exit 2
elif [[ "${PHASE}" == "joint" && "${MAX_STEPS}" -gt 12000 ]]; then
  echo "Joint MAX_STEPS may not exceed 12000." >&2; exit 2
fi

configure_modelarts_distributed
require_scale_cluster
require_output_url
ensure_spatialvid_subset_splits
if [[ -n "${DUAL_AE_CKPT_URL}" || -n "${DUAL_AE_CKPT_MIRROR_URL}" ]]; then
  if [[ "${NODE_RANK}" -ne 0 ]]; then rm -f "${DUAL_AE_CKPT}"; fi
  ensure_local_checkpoint "${DUAL_AE_CKPT}" "${DUAL_AE_CKPT_URL}" \
    "dual-stream AE checkpoint" "${DUAL_AE_CKPT_MIRROR_URL}"
fi
require_file "${DUAL_AE_CKPT}" "dual-stream AE checkpoint"
mkdir -p "${OUTPUT_DIR}" "${LOCAL_CACHE_ROOT}/resume"

# Full resume is deliberately searched only inside this exact phase/namespace.
# In particular, the old 3k prototype can never enter through --resume.
if [[ "${AUTO_RESUME}" == 1 || -n "${RESUME}" ]]; then
  RESUME=$(resolve_resume_checkpoint \
    "${RESUME}" "${OUTPUT_DIR}/checkpoint_latest.pt" \
    "${REMOTE_OUTPUT_DIR}/checkpoint_latest.pt" \
    "${LOCAL_CACHE_ROOT}/resume/${R7_NAMESPACE//\//_}_${PHASE}.pt" \
    "${MIRROR_OUTPUT_DIR}/checkpoint_latest.pt")
fi

# Restore append-only metrics before output mirroring starts, so a restarted
# node 0 cannot overwrite the durable history with an empty local directory.
if [[ -n "${RESUME}" && ! -s "${OUTPUT_DIR}/metrics.jsonl" ]]; then
  for source in "${RESUME_METRICS_URL:-${REMOTE_OUTPUT_DIR}/metrics.jsonl}" \
                "${MIRROR_OUTPUT_DIR}/metrics.jsonl"; do
    "${PYTHON_BIN}" "${PROJECT}/scripts/moxing_transfer.py" \
      "${source}" "${OUTPUT_DIR}/metrics.jsonl" 2>/dev/null && break || true
  done
fi

INIT_CKPT=""
ALLOW_LEGACY=0
if [[ -z "${RESUME}" ]]; then
  if [[ -n "${LEGACY_INIT_CKPT}" ]]; then
    INIT_CKPT=$(stage_resume_checkpoint "${LEGACY_INIT_CKPT}" \
      "${LOCAL_CACHE_ROOT}/resume/${R7_NAMESPACE//\//_}_${PHASE}_init.pt")
    ALLOW_LEGACY=1
  elif [[ "${PHASE}" == "codec" ]]; then
    # Formal v2 codec is always a fresh 0->6k training stage: old 3k state is
    # migrated as model weights only. Optimizer, scheduler, global step, RNG,
    # best metrics, and early-stop state are deliberately not resumed.
    LEGACY_RUN="${LEGACY_RUN:-causal_dual_tokenizer_t${TEMPORAL_FACTOR}_c${LATENT_DIM}_codec}"
    INIT_CKPT=$(resolve_resume_checkpoint "" \
      "${SCALE_ROOT}/${LEGACY_RUN}/checkpoint_best.pt" \
      "${SCALE_REMOTE_ROOT}/${LEGACY_RUN}/checkpoint_best.pt" \
      "${LOCAL_CACHE_ROOT}/resume/${R7_NAMESPACE//\//_}_legacy_init.pt" \
      "${SCALE_MIRROR_ROOT}/${LEGACY_RUN}/checkpoint_best.pt")
    if [[ -z "${INIT_CKPT}" ]]; then
      INIT_CKPT=$(resolve_resume_checkpoint "" \
        "${SCALE_ROOT}/${LEGACY_RUN}/checkpoint_latest.pt" \
        "${SCALE_REMOTE_ROOT}/${LEGACY_RUN}/checkpoint_latest.pt" \
        "${LOCAL_CACHE_ROOT}/resume/${R7_NAMESPACE//\//_}_legacy_init.pt" \
        "${SCALE_MIRROR_ROOT}/${LEGACY_RUN}/checkpoint_latest.pt")
    fi
    ALLOW_LEGACY=1
  else
    # Joint is a new optimizer/scheduler stage initialized strictly from codec best.
    CODEC_RUN="${R7_NAMESPACE}/codec"
    INIT_CKPT=$(resolve_resume_checkpoint "" \
      "${SCALE_ROOT}/${CODEC_RUN}/checkpoint_best.pt" \
      "${SCALE_REMOTE_ROOT}/${CODEC_RUN}/checkpoint_best.pt" \
      "${LOCAL_CACHE_ROOT}/resume/${R7_NAMESPACE//\//_}_codec_best.pt" \
      "${SCALE_MIRROR_ROOT}/${CODEC_RUN}/checkpoint_best.pt")
  fi
  [[ -n "${INIT_CKPT}" ]] || {
    echo "No weights initializer found for ${PHASE}; set LEGACY_INIT_CKPT explicitly." >&2; exit 1;
  }
fi
if [[ "${EXTEND}" == 1 && -z "${RESUME}" ]]; then
  echo "EXTEND=1 requires a same-namespace completed full-state checkpoint." >&2; exit 1
fi

EXTRA_ARGS=()
[[ -n "${RESUME}" ]] && EXTRA_ARGS+=(--resume "${RESUME}")
[[ -n "${INIT_CKPT}" ]] && EXTRA_ARGS+=(--init_ckpt "${INIT_CKPT}")
[[ "${ALLOW_LEGACY}" == 1 ]] && EXTRA_ARGS+=(--allow_legacy_checkpoint)

start_output_sync "${OUTPUT_DIR}" "${REMOTE_OUTPUT_DIR}"
trap 'stop_output_sync "${OUTPUT_DIR}" "${REMOTE_OUTPUT_DIR}"' EXIT
run_torchrun "${PROJECT}/train_causal_dual_tokenizer.py" \
  --csv "${SPATIALVID_TRAIN_10K_CSV}" --video_root "${SPATIALVID_VIDEO_ROOT}" \
  --eval_csv "${SPATIALVID_EVAL_CSV}" --encoder_ckpt "${STREAMVGGT_CKPT}" \
  --dual_ae_ckpt "${DUAL_AE_CKPT}" \
  --temporal_factor "${TEMPORAL_FACTOR}" \
  --geo_latent_dim "${GEO_LATENT_DIM}" --tex_latent_dim "${TEX_LATENT_DIM}" \
  --phase "${PHASE}" --seq_len "${SEQ_LEN}" --target_size 518 \
  --clip_duration_seconds "${CLIP_DURATION_SECONDS:-1.0}" \
  --batch_size "${BATCH_SIZE}" --accum_steps "${ACCUM_STEPS}" \
  --max_steps "${MAX_STEPS}" --lr "${LEARNING_RATE}" \
  --pretrained_lr "${PRETRAINED_LR}" --warmup_steps "${WARMUP_STEPS}" \
  --extension_lr "${EXTENSION_LR}" \
  --extension_warmup_steps "${EXTENSION_WARMUP_STEPS}" \
  --num_workers "${NUM_WORKERS}" --eval_clips "${EVAL_CLIPS}" \
  --frames_chunk_size "${FRAMES_CHUNK_SIZE:-0}" \
  --min_steps "${MIN_STEPS}" --early_stop_patience "${EARLY_STOP_PATIENCE}" \
  --early_stop_min_delta "${EARLY_STOP_MIN_DELTA:-0.001}" \
  --gate_psnr "${GATE_PSNR:-23.9}" --gate_lpips "${GATE_LPIPS:-0.13}" \
  --gate_boundary_ratio "${GATE_BOUNDARY_RATIO:-1.10}" \
  --gate_geo_motion_cosine "${GATE_GEO_MOTION_COSINE:-0.95}" \
  --log_every "${LOG_EVERY}" --eval_every "${EVAL_EVERY}" \
  --save_every "${SAVE_EVERY}" --dtype bf16 --output_dir "${OUTPUT_DIR}" \
  "${EXTRA_ARGS[@]}"
