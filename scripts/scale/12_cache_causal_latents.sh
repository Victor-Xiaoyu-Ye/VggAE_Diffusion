#!/bin/bash
# Cache accepted R7 temporal latents before world-model training.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"; source "${SCRIPT_DIR}/../lib/spatialvid.sh"; source "${SCRIPT_DIR}/../lib/modelarts.sh"
TEMPORAL_FACTOR="${TEMPORAL_FACTOR:-2}"; LATENT_DIM="${LATENT_DIM:-192}"; RUN="causal_dual_tokenizer_t${TEMPORAL_FACTOR}_c${LATENT_DIM}_${PHASE:-joint}"
R7_CKPT="${R7_CKPT:-${SCALE_ROOT}/${RUN}/checkpoint_latest.pt}"; CACHE_DIR="${R7_CACHE_DIR:-${LOCAL_CACHE_ROOT}/cache_r7_t${TEMPORAL_FACTOR}_c${LATENT_DIM}}"
configure_modelarts_distributed; ensure_spatialvid_subset_splits; require_file "${R7_CKPT}" "R7 checkpoint"
"${PYTHON_BIN}" "${PROJECT}/cache_causal_video_latents.py" --csv "${SPATIALVID_TRAIN_10K_CSV}" --video_root "${SPATIALVID_VIDEO_ROOT}" --encoder_ckpt "${STREAMVGGT_CKPT}" --r7_ckpt "${R7_CKPT}" --output_dir "${CACHE_DIR}" --seq_len "${SEQ_LEN:-9}" --num_workers "${NUM_WORKERS:-4}"
