#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT=${PROJECT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
CSV=${CSV:?Set CSV to the training metadata CSV}
VIDEO_ROOT=${VIDEO_ROOT:?Set VIDEO_ROOT to the video root}
EVAL_CSV=${EVAL_CSV:?Set EVAL_CSV to a held-out metadata CSV}
EVAL_VIDEO_ROOT=${EVAL_VIDEO_ROOT:-${VIDEO_ROOT}}
ENCODER_CKPT=${ENCODER_CKPT:?Set ENCODER_CKPT to StreamVGGT weights}
AUTOENCODER_CKPT=${AUTOENCODER_CKPT:?Set AUTOENCODER_CKPT}
I0_DECODER_CKPT=${I0_DECODER_CKPT:?Set I0_DECODER_CKPT}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT}/ckpts/10k/compact_diffusion}
NUM_GPUS=${NUM_GPUS:-4}
GPU_IDS=${GPU_IDS:-0,1,2,3}
MASTER_PORT=${MASTER_PORT:-29540}
RESUME=${RESUME:-}
EXTRA_ARGS=()
if [[ -n "${RESUME}" ]]; then
  EXTRA_ARGS+=(--resume "${RESUME}")
fi

CUDA_VISIBLE_DEVICES="${GPU_IDS}" torchrun \
  --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" \
  "${PROJECT}/train_compact_diffusion.py" \
  --csv "${CSV}" \
  --video_root "${VIDEO_ROOT}" \
  --eval_csv "${EVAL_CSV}" \
  --eval_video_root "${EVAL_VIDEO_ROOT}" \
  --encoder_ckpt "${ENCODER_CKPT}" \
  --autoencoder_ckpt "${AUTOENCODER_CKPT}" \
  --i0_decoder_ckpt "${I0_DECODER_CKPT}" \
  --output_dir "${OUTPUT_DIR}" \
  --latent_dim 512 --latent_grid 18 \
  --model_dim 768 --spatial_depth 8 --temporal_depth 4 --num_heads 12 \
  --time_scale 1000 \
  --i0_condition --i0_residual \
  --levels 4 11 17 23 \
  --decoder_base_dim 384 --decoder_num_resblocks 2 \
  --decoder_pixel_shuffle --decoder_temporal_blocks 2 \
  --rescale --no_decoder_aux \
  --batch_size 2 --accum_steps 4 \
  --epochs 50 --lr 1e-4 --wd 1e-2 \
  --warmup_steps 1000 --ema_decay 0.9999 \
  --seq_len 8 --target_size 518 --max_frame_span 32 \
  --num_workers 4 --dtype bf16 \
  --log_every 50 --eval_every 5 --save_every 5 \
  "${EXTRA_ARGS[@]}"
