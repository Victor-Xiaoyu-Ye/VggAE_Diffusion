#!/bin/bash
# E3: Grid artifact isolation.
#
# Same compressed pipeline (GenerativeTokenizer + CompactDecoder), three
# decoder/upsample variants. Identifies which component produces the grid
# texture seen in current reconstructions.
#
# Variants:
#   (a) pixel_shuffle  -> current default, PixelShuffle upsample
#   (b) resize_conv    -> bilinear + conv (resize-conv, checkerboard-free)
#
# Both use the SAME tokenizer, latent_dim, latent_grid, data, and seed so
# the only variable is the decoder upsample path. Compare PSNR + visual
# grid texture in samples/.
#
# Note: the third hypothesis (adaptive_avg_pool 37->18 odd/even
# misalignment) is tested separately by re-running with --latent_grid 19
# or by bypassing spatial compression in the probe; that requires a
# tokenizer change and is left as a follow-up if (a) vs (b) do not
# isolate the grid.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/h200_env.sh"
ensure_h200_splits

# ----------------------------- editable settings -----------------------------
EPOCHS="${EPOCHS:-30}"
EVAL_CLIPS="${EVAL_CLIPS:-32}"
BATCH_SIZE="${BATCH_SIZE:-2}"
ACCUM_STEPS="${ACCUM_STEPS:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
DECODER_BASE_DIM="${DECODER_BASE_DIM:-384}"
LATENT_DIM="${LATENT_DIM:-512}"
LATENT_GRID="${LATENT_GRID:-18}"
# ----------------------------------------------------------------------------

run_variant() {
  local variant="$1"   # pixel_shuffle | resize_conv
  local use_ps="$2"    # 1 | 0
  local PROBE_NAME="e3_${variant}"
  local OUTPUT_DIR="${VGGAE_H200_RUN_ROOT}/probes/${PROBE_NAME}"
  local RESUME_ARGS=()
  if [[ -s "${OUTPUT_DIR}/checkpoint_latest.pt" ]]; then
    RESUME_ARGS+=(--resume "${OUTPUT_DIR}/checkpoint_latest.pt")
  fi
  mkdir -p "${OUTPUT_DIR}/logs"
  echo "[E3] variant=${variant} use_pixel_shuffle=${use_ps} -> ${OUTPUT_DIR}"

  h200_torchrun "${VGGAE_PROJECT}/probe_feature_recon.py" \
    --mode compressed \
    --probe_name "${PROBE_NAME}" \
    --csv "${VGGAE_TRAIN_10K_CSV}" \
    --video_root "${VGGAE_VIDEO_ROOT}" \
    --eval_csv "${VGGAE_EVAL_CSV}" \
    --eval_video_root "${VGGAE_VIDEO_ROOT}" \
    --encoder_ckpt "${VGGAE_ENCODER_CKPT}" \
    --levels 4 11 17 23 \
    --latent_dim "${LATENT_DIM}" \
    --latent_grid "${LATENT_GRID}" \
    --decoder_base_dim "${DECODER_BASE_DIM}" \
    --decoder_num_resblocks 2 \
    --decoder_use_pixel_shuffle "${use_ps}" \
    --num_temporal_blocks 1 \
    --lambda_l1 1.0 --lambda_mse 0.5 --lambda_lpips 1.0 \
    --lambda_grad 0.05 --lambda_temporal 0.05 \
    --lambda_latent_reg 0.01 \
    --batch_size "${BATCH_SIZE}" --accum_steps "${ACCUM_STEPS}" \
    --epochs "${EPOCHS}" --lr "${LEARNING_RATE}" --wd 1e-2 \
    --warmup_steps 200 --ema_decay 0.999 \
    --seq_len 8 --target_size 518 --clip_duration_seconds 1.0 \
    --num_workers 8 --dtype bf16 \
    --output_dir "${OUTPUT_DIR}" \
    --eval_every 2 --save_every 5 --log_every 50 \
    --eval_clips "${EVAL_CLIPS}" \
    "${RESUME_ARGS[@]}" \
    2>&1 | tee -a "${OUTPUT_DIR}/logs/train.log"
}

run_variant pixel_shuffle 1
run_variant resize_conv 0

echo "[E3] done. Compare samples/ grids between:"
echo "  ${VGGAE_H200_RUN_ROOT}/probes/e3_pixel_shuffle/samples/"
echo "  ${VGGAE_H200_RUN_ROOT}/probes/e3_resize_conv/samples/"
