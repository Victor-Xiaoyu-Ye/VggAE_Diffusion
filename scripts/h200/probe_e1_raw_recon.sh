#!/bin/bash
# E1: Raw feature RGB reconstruction ceiling.
#
# Concat all four DPT levels (4,11,17,23) at the native 37x37 grid, project
# to 512-dim, decode with a strong resize-conv decoder. NO tokenizer
# compression. This measures the frozen StreamVGGT feature space's RGB
# reconstruction ceiling.
#
# Decision rule:
#   PSNR >= 28 -> features carry RGB high-freq, two-stream latent worth building
#   PSNR <= 23 -> features do not carry RGB high-freq, use single geometry
#                 latent + decoder-side hallucination
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/h200_env.sh"
ensure_h200_splits

# ----------------------------- editable settings -----------------------------
PROBE_NAME="e1_raw_4lvl_proj512"
OUTPUT_DIR="${VGGAE_H200_RUN_ROOT}/probes/${PROBE_NAME}"
RESUME="${RESUME:-}"

EPOCHS="${EPOCHS:-40}"
EVAL_CLIPS="${EVAL_CLIPS:-32}"
# Memory guide (single card, raw 37-grid probe): ~31G at batch 2.
# Fixed cost (StreamVGGT encoder + LPIPS VGG + weights) is roughly constant,
# activations scale ~linearly with batch. Rough single-card estimate:
#   batch 2 ~ 31G | batch 4 ~ 47G | batch 6 ~ 63G | batch 8 ~ 79G.
# batch*accum = effective batch (16 here); raise BATCH_SIZE and lower
# ACCUM_STEPS together to keep it constant while using more of the GPU.
BATCH_SIZE="${BATCH_SIZE:-4}"
ACCUM_STEPS="${ACCUM_STEPS:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
DECODER_BASE_DIM="${DECODER_BASE_DIM:-384}"
PROJ_DIM="${PROJ_DIM:-512}"
# resize-conv by default to avoid PixelShuffle checkerboard in the ceiling probe
USE_PIXEL_SHUFFLE="${USE_PIXEL_SHUFFLE:-0}"
# Memory knobs (the probe OOMs without these). Defaults: checkpoint on,
# cap last feature map at 296x296 then interp to 518, decode 8 frames in
# chunks of 4. Lower MAX_FEATURE_GRID / FRAMES_CHUNK_SIZE to fit smaller
# cards; raise them for throughput on big cards.
USE_CHECKPOINT="${USE_CHECKPOINT:-1}"
MAX_FEATURE_GRID="${MAX_FEATURE_GRID:-296}"
FRAMES_CHUNK_SIZE="${FRAMES_CHUNK_SIZE:-4}"
# ----------------------------------------------------------------------------

EXTRA_ARGS=()
if [[ -n "${RESUME}" && -s "${RESUME}" ]]; then
  EXTRA_ARGS+=(--resume "${RESUME}")
elif [[ -s "${OUTPUT_DIR}/checkpoint_latest.pt" ]]; then
  EXTRA_ARGS+=(--resume "${OUTPUT_DIR}/checkpoint_latest.pt")
fi

mkdir -p "${OUTPUT_DIR}/logs"
echo "[E1] raw feature ceiling -> ${OUTPUT_DIR}"

h200_torchrun "${VGGAE_PROJECT}/probe_feature_recon.py" \
  --mode raw \
  --probe_name "${PROBE_NAME}" \
  --csv "${VGGAE_TRAIN_10K_CSV}" \
  --video_root "${VGGAE_VIDEO_ROOT}" \
  --eval_csv "${VGGAE_EVAL_CSV}" \
  --eval_video_root "${VGGAE_VIDEO_ROOT}" \
  --encoder_ckpt "${VGGAE_ENCODER_CKPT}" \
  --levels 4 11 17 23 \
  --proj_dim "${PROJ_DIM}" \
  --decoder_base_dim "${DECODER_BASE_DIM}" \
  --decoder_num_resblocks 2 \
  --decoder_use_pixel_shuffle "${USE_PIXEL_SHUFFLE}" \
  --num_temporal_blocks 1 \
  --max_feature_grid "${MAX_FEATURE_GRID}" \
  --use_checkpoint "${USE_CHECKPOINT}" \
  --frames_chunk_size "${FRAMES_CHUNK_SIZE}" \
  --lambda_l1 1.0 --lambda_mse 0.5 --lambda_lpips 1.0 \
  --lambda_grad 0.05 --lambda_temporal 0.05 \
  --batch_size "${BATCH_SIZE}" --accum_steps "${ACCUM_STEPS}" \
  --epochs "${EPOCHS}" --lr "${LEARNING_RATE}" --wd 1e-2 \
  --warmup_steps 200 --ema_decay 0.999 \
  --seq_len 8 --target_size 518 --clip_duration_seconds 1.0 \
  --num_workers 8 --dtype bf16 \
  --output_dir "${OUTPUT_DIR}" \
  --eval_every 2 --save_every 5 --log_every 50 \
  --eval_clips "${EVAL_CLIPS}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee -a "${OUTPUT_DIR}/logs/train.log"
