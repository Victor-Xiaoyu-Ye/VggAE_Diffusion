#!/bin/bash
# R7 causal dual-tokenizer training (factor 2/4) on the scale cluster.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/spatialvid.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"

TEMPORAL_FACTOR="${TEMPORAL_FACTOR:-2}"
if [[ "${TEMPORAL_FACTOR}" -eq 2 ]]; then
  GEO_LATENT_DIM="${GEO_LATENT_DIM:-96}"; TEX_LATENT_DIM="${TEX_LATENT_DIM:-96}"
else
  GEO_LATENT_DIM="${GEO_LATENT_DIM:-128}"; TEX_LATENT_DIM="${TEX_LATENT_DIM:-128}"
fi
PHASE="${PHASE:-codec}"
RUN_NAME="causal_dual_tokenizer_t${TEMPORAL_FACTOR}_c$((GEO_LATENT_DIM+TEX_LATENT_DIM))_${PHASE}"
OUTPUT_DIR="${SCALE_ROOT}/${RUN_NAME}"
REMOTE_OUTPUT_DIR="${SCALE_REMOTE_ROOT}/${RUN_NAME}"
DUAL_AE_CKPT="${DUAL_AE_CKPT:-${VGGAE_REF_ROOT}/checkpoints/e5_dual_stream_r5.pt}"
MAX_STEPS="${MAX_STEPS:-3000}"; AUTO_RESUME="${AUTO_RESUME:-1}"; RESUME="${RESUME:-}"
MASTER_PORT="${MASTER_PORT:-29670}"

configure_modelarts_distributed
require_scale_cluster
require_output_url
ensure_spatialvid_subset_splits
require_file "${DUAL_AE_CKPT}" "R5 dual-stream AE checkpoint"
if [[ "${AUTO_RESUME}" -eq 1 || -n "${RESUME}" ]]; then
  RESUME=$(resolve_resume_checkpoint "${RESUME}" "${OUTPUT_DIR}/checkpoint_latest.pt" "${REMOTE_OUTPUT_DIR}/checkpoint_latest.pt" "${LOCAL_CACHE_ROOT}/resume/${RUN_NAME}.pt" "${SCALE_MIRROR_ROOT}/${RUN_NAME}/checkpoint_latest.pt")
fi
EXTRA_ARGS=(); [[ -n "${RESUME}" ]] && EXTRA_ARGS+=(--resume "${RESUME}")
start_output_sync "${OUTPUT_DIR}" "${REMOTE_OUTPUT_DIR}"
trap 'stop_output_sync "${OUTPUT_DIR}" "${REMOTE_OUTPUT_DIR}"' EXIT
run_torchrun "${PROJECT}/train_causal_dual_tokenizer.py" \
  --csv "${SPATIALVID_TRAIN_10K_CSV}" --video_root "${SPATIALVID_VIDEO_ROOT}" \
  --eval_csv "${SPATIALVID_EVAL_CSV}" --encoder_ckpt "${STREAMVGGT_CKPT}" \
  --dual_ae_ckpt "${DUAL_AE_CKPT}" --temporal_factor "${TEMPORAL_FACTOR}" \
  --geo_latent_dim "${GEO_LATENT_DIM}" --tex_latent_dim "${TEX_LATENT_DIM}" \
  --phase "${PHASE}" --seq_len "${SEQ_LEN:-9}" --target_size 518 \
  --clip_duration_seconds "${CLIP_DURATION_SECONDS:-1.0}" \
  --batch_size "${BATCH_SIZE:-1}" --accum_steps "${ACCUM_STEPS:-2}" \
  --max_steps "${MAX_STEPS}" --num_workers "${NUM_WORKERS:-4}" \
  --eval_clips "${EVAL_CLIPS:-16}" --frames_chunk_size "${FRAMES_CHUNK_SIZE:-1}" \
  --log_every "${LOG_EVERY:-50}" --eval_every "${EVAL_EVERY:-500}" \
  --save_every "${SAVE_EVERY:-500}" --dtype bf16 --output_dir "${OUTPUT_DIR}" \
  "${EXTRA_ARGS[@]}"
