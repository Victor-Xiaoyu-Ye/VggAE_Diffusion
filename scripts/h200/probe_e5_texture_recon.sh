#!/bin/bash
# E5: Dual-stream reconstruction probe (z_geo + TextureEncoder z_tex).
#
# Tests whether an explicit per-frame appearance latent can break the ~20 PSNR
# VGGT-only ceiling from E1/E4. Decoder sees (z_geo, z_tex) only — no RGB skip.
#
#   TEX_MODE=oracle (default): TextureEncoder(RGB) -> z_tex  (reconstruction ceiling)
#   TEX_MODE=zero:             z_tex = 0                   (geo-only ablation)
#
# Run:
#   bash scripts/h200/probe_e5_texture_recon.sh
#   TEX_MODE=zero bash scripts/h200/probe_e5_texture_recon.sh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/h200_env.sh"
ensure_h200_splits

# E5 uses all 4 H200 cards.
export VGGAE_NUM_GPUS=4
export VGGAE_GPU_IDS=0,1,2,3
# Avoid colliding with a leftover E4 master port.
export VGGAE_MASTER_PORT="${VGGAE_MASTER_PORT:-29550}"

# ----------------------------- editable settings -----------------------------
TEX_MODE="${TEX_MODE:-oracle}"
if [[ "${TEX_MODE}" == "zero" ]]; then
  PROBE_NAME="e5_dual_stream_zero"
else
  PROBE_NAME="e5_dual_stream_oracle"
fi
OUTPUT_DIR="${VGGAE_H200_RUN_ROOT}/probes/${PROBE_NAME}"
RESUME="${RESUME:-}"

EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-2}"
ACCUM_STEPS="${ACCUM_STEPS:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
GEO_DIM="${GEO_DIM:-256}"
TEX_DIM="${TEX_DIM:-256}"
TEX_BASE_CH="${TEX_BASE_CH:-64}"
LATENT_GRID="${LATENT_GRID:-18}"
DECODER_BASE_DIM="${DECODER_BASE_DIM:-384}"
LPIPS_WEIGHT="${LPIPS_WEIGHT:-0.1}"
USE_CHECKPOINT="${USE_CHECKPOINT:-1}"
FRAMES_CHUNK_SIZE="${FRAMES_CHUNK_SIZE:-4}"
# ----------------------------------------------------------------------------

EXTRA_ARGS=()
if [[ -n "${RESUME}" && -s "${RESUME}" ]]; then
  EXTRA_ARGS+=(--resume "${RESUME}")
elif [[ -s "${OUTPUT_DIR}/checkpoint_latest.pt" ]]; then
  EXTRA_ARGS+=(--resume "${OUTPUT_DIR}/checkpoint_latest.pt")
fi

mkdir -p "${OUTPUT_DIR}/logs"
echo "[E5] tex_mode=${TEX_MODE} -> ${OUTPUT_DIR} (${VGGAE_NUM_GPUS} GPUs)"

h200_torchrun "${VGGAE_PROJECT}/probe_e5_texture_recon.py" \
  --tex_mode "${TEX_MODE}" \
  --probe_name "${PROBE_NAME}" \
  --csv "${VGGAE_TRAIN_10K_CSV}" \
  --video_root "${VGGAE_VIDEO_ROOT}" \
  --eval_csv "${VGGAE_EVAL_CSV}" \
  --eval_video_root "${VGGAE_VIDEO_ROOT}" \
  --encoder_ckpt "${VGGAE_ENCODER_CKPT}" \
  --levels 4 11 17 23 \
  --geo_dim "${GEO_DIM}" \
  --tex_dim "${TEX_DIM}" \
  --tex_base_ch "${TEX_BASE_CH}" \
  --latent_grid "${LATENT_GRID}" \
  --decoder_base_dim "${DECODER_BASE_DIM}" \
  --lambda_l1 1.0 --lambda_lpips "${LPIPS_WEIGHT}" --lambda_tex_reg 0.01 \
  --batch_size "${BATCH_SIZE}" --accum_steps "${ACCUM_STEPS}" \
  --epochs "${EPOCHS}" --lr "${LEARNING_RATE}" --wd 1e-2 \
  --warmup_steps 200 --ema_decay 0.999 \
  --use_checkpoint "${USE_CHECKPOINT}" \
  --frames_chunk_size "${FRAMES_CHUNK_SIZE}" \
  --seq_len 8 --target_size 518 --clip_duration_seconds 1.0 \
  --num_workers 8 --dtype bf16 \
  --output_dir "${OUTPUT_DIR}" \
  --eval_every 2 --save_every 5 --log_every 50 \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee -a "${OUTPUT_DIR}/logs/train.log"
