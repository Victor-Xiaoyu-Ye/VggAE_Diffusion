#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT=${PROJECT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
CSV=${CSV:?Set CSV to metadata containing the fixed validation clip}
VIDEO_ROOT=${VIDEO_ROOT:?Set VIDEO_ROOT to the video root}
ENCODER_CKPT=${ENCODER_CKPT:?Set ENCODER_CKPT to StreamVGGT weights}
AUTOENCODER_CKPT=${AUTOENCODER_CKPT:?Set AUTOENCODER_CKPT}
I0_DECODER_CKPT=${I0_DECODER_CKPT:?Set I0_DECODER_CKPT from the overfit decoder}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT}/ckpts/validation/compact_diffusion_overfit}
GPU_ID=${GPU_ID:-0}
RESUME=${RESUME:-}
EXTRA_ARGS=()
if [[ -n "${RESUME}" ]]; then
  EXTRA_ARGS+=(--resume "${RESUME}")
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 "${PROJECT}/train_compact_diffusion.py" \
  --csv "${CSV}" \
  --video_root "${VIDEO_ROOT}" \
  --eval_csv "${CSV}" \
  --eval_video_root "${VIDEO_ROOT}" \
  --encoder_ckpt "${ENCODER_CKPT}" \
  --autoencoder_ckpt "${AUTOENCODER_CKPT}" \
  --i0_decoder_ckpt "${I0_DECODER_CKPT}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_videos 1 --disable_temporal_jitter \
  --latent_dim 512 --latent_grid 18 \
  --model_dim 384 --spatial_depth 4 --temporal_depth 2 --num_heads 6 \
  --time_scale 1000 \
  --i0_condition --i0_residual --levels 4 11 17 23 \
  --decoder_base_dim 384 --decoder_num_resblocks 2 \
  --decoder_pixel_shuffle --decoder_temporal_blocks 2 \
  --no_decoder_aux --rescale \
  --batch_size 1 --accum_steps 1 \
  --epochs 3000 --lr 3e-4 --wd 0 \
  --warmup_steps 100 --ema_decay 0.999 \
  --seq_len 8 --target_size 518 --max_frame_span 32 \
  --num_workers 0 --dtype bf16 \
  --log_every 10 --eval_every 100 --save_every 100 \
  "${EXTRA_ARGS[@]}"
