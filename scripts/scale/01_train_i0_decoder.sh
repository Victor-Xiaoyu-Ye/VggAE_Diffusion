#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/spatialvid.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"

# ----------------------------- editable settings -----------------------------
AUTOENCODER_CKPT="${SCALE_GEOMETRY_AE_CKPT}"
OUTPUT_DIR="${SCALE_ROOT}/i0_decoder"
REMOTE_OUTPUT_DIR="${SCALE_REMOTE_ROOT}/i0_decoder"
RESUME=""
REPRESENTATION_MAX_VIDEOS=1000000

EPOCHS=3
BATCH_SIZE=1
ACCUM_STEPS=8
LEARNING_RATE=1e-4
PRETRAINED_LR_SCALE=0.1
WEIGHT_DECAY=1e-2
WARMUP_STEPS=5000
NUM_WORKERS=8
CLIP_DURATION_SECONDS=1.0
MASTER_PORT=29601
# -----------------------------------------------------------------------------

configure_modelarts_distributed
require_scale_cluster
ensure_spatialvid_scale_splits
require_file "${AUTOENCODER_CKPT}" "scale geometry autoencoder checkpoint"

EXTRA_ARGS=()
if [[ -n "${RESUME}" ]]; then
  RESUME=$(stage_resume_checkpoint \
    "${RESUME}" "${LOCAL_CACHE_ROOT}/resume/i0_decoder.pt")
  EXTRA_ARGS+=(--resume "${RESUME}")
fi

start_output_sync "${OUTPUT_DIR}" "${REMOTE_OUTPUT_DIR}"
trap 'stop_output_sync "${OUTPUT_DIR}" "${REMOTE_OUTPUT_DIR}"' EXIT

run_torchrun "${PROJECT}/train_i0_autoencoder.py" \
  --csv "${SPATIALVID_FULL_TRAIN_CSV}" \
  --video_root "${SPATIALVID_VIDEO_ROOT}" \
  --eval_csv "${SPATIALVID_EVAL_CSV}" \
  --eval_video_root "${SPATIALVID_VIDEO_ROOT}" \
  --encoder_ckpt "${STREAMVGGT_CKPT}" \
  --autoencoder_ckpt "${AUTOENCODER_CKPT}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_videos "${REPRESENTATION_MAX_VIDEOS}" \
  --latent_dim 512 --latent_grid 18 \
  --disable_temporal_mixer \
  --decoder_base_dim 384 --decoder_num_resblocks 2 \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --accum_steps "${ACCUM_STEPS}" \
  --lr "${LEARNING_RATE}" \
  --pretrained_lr_scale "${PRETRAINED_LR_SCALE}" \
  --wd "${WEIGHT_DECAY}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --max_grad_norm 1.0 --lambda_lpips 1.0 \
  --dtype fp16 --seq_len 8 --target_size 518 \
  --max_frame_span 32 \
  --clip_duration_seconds "${CLIP_DURATION_SECONDS}" \
  --num_workers "${NUM_WORKERS}" \
  --log_every 50 --save_every 1 --eval_every 1 \
  "${EXTRA_ARGS[@]}"
