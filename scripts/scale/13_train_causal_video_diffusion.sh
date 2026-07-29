#!/bin/bash
# Train the R7 compressed-latent chunk world model.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"; source "${SCRIPT_DIR}/../lib/modelarts.sh"
TEMPORAL_FACTOR="${TEMPORAL_FACTOR:-2}"; LATENT_DIM="${LATENT_DIM:-192}"; RUN_NAME="causal_video_diffusion_t${TEMPORAL_FACTOR}_c${LATENT_DIM}_k${CONTEXT_CHUNKS:-1}m${FUTURE_CHUNKS:-2}"
OUTPUT_DIR="${SCALE_ROOT}/${RUN_NAME}"; REMOTE_OUTPUT_DIR="${SCALE_REMOTE_ROOT}/${RUN_NAME}"; MANIFEST="${R7_MANIFEST:-${LOCAL_CACHE_ROOT}/cache_r7_t${TEMPORAL_FACTOR}_c${LATENT_DIM}/manifest.txt}"; RESUME="${RESUME:-}"; AUTO_RESUME="${AUTO_RESUME:-1}"; MASTER_PORT="${MASTER_PORT:-29680}"
configure_modelarts_distributed; require_scale_cluster; require_output_url; require_file "${MANIFEST}" "R7 latent manifest"
if [[ "${AUTO_RESUME}" -eq 1 || -n "${RESUME}" ]]; then RESUME=$(resolve_resume_checkpoint "${RESUME}" "${OUTPUT_DIR}/checkpoint_latest.pt" "${REMOTE_OUTPUT_DIR}/checkpoint_latest.pt" "${LOCAL_CACHE_ROOT}/resume/${RUN_NAME}.pt" "${SCALE_MIRROR_ROOT}/${RUN_NAME}/checkpoint_latest.pt"); fi
EXTRA=(); [[ -n "${RESUME}" ]] && EXTRA+=(--resume "${RESUME}")
start_output_sync "${OUTPUT_DIR}" "${REMOTE_OUTPUT_DIR}"; trap 'stop_output_sync "${OUTPUT_DIR}" "${REMOTE_OUTPUT_DIR}"' EXIT
run_torchrun "${PROJECT}/train_causal_video_diffusion.py" --manifest "${MANIFEST}" --output_dir "${OUTPUT_DIR}" --latent_dim "${LATENT_DIM}" --context_chunks "${CONTEXT_CHUNKS:-1}" --future_chunks "${FUTURE_CHUNKS:-2}" --model_dim "${MODEL_DIM:-1152}" --spatial_depth "${SPATIAL_DEPTH:-10}" --temporal_depth "${TEMPORAL_DEPTH:-8}" --num_heads "${NUM_HEADS:-16}" --batch_size "${BATCH_SIZE:-2}" --max_steps "${MAX_STEPS:-6000}" --lr "${LEARNING_RATE:-1e-4}" --num_workers "${NUM_WORKERS:-4}" "${EXTRA[@]}"
