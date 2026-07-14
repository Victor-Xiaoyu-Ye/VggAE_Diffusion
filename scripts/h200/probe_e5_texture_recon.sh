#!/bin/bash
# E5: Dual-stream reconstruction probe (z_geo + TextureEncoder z_tex).
#
# Tests whether an explicit per-frame appearance latent can break the ~20 PSNR
# VGGT-only ceiling from E1/E4. Decoder sees (z_geo, z_tex) only — no RGB skip.
#
#   TEX_MODE=oracle (default): TextureEncoder(RGB) -> z_tex (recon ceiling)
#   TEX_MODE=zero:             z_tex = 0            (geo-only ablation)
#   TEX_MODE=tex_only:         z_geo = 0            (proves geo load-bearing)
#
# Ablation knobs (2026-07 survey; see PROJECT_CONTEXT.md):
#   TEX_PACK=s2d          space-to-depth packing instead of avg-pool
#   TEX_REG_MODE=match_geo SVG-style stat alignment to z_geo instead of N(0,1)
#   FEAT_WEIGHT=0.5       VGGT feature-consistency loss (MIRA P-DINO analogue)
#
# Run matrix (see scripts/h200/README.md):
#   bash scripts/h200/probe_e5_texture_recon.sh
#   TEX_MODE=zero bash scripts/h200/probe_e5_texture_recon.sh
#   TEX_PACK=s2d TEX_REG_MODE=match_geo bash scripts/h200/probe_e5_texture_recon.sh
#   TEX_MODE=tex_only TEX_PACK=s2d bash scripts/h200/probe_e5_texture_recon.sh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# Capture user-provided overrides BEFORE h200_env.sh / the E5 defaults below
# hard-assign them (needed for single-card smoke runs).
USER_MASTER_PORT="${VGGAE_MASTER_PORT:-}"
USER_NUM_GPUS="${VGGAE_NUM_GPUS:-}"
USER_GPU_IDS="${VGGAE_GPU_IDS:-}"
source "${SCRIPT_DIR}/h200_env.sh"
ensure_h200_splits

# E5 default: all 4 H200 cards. Env overrides (captured above) win, e.g.
#   VGGAE_NUM_GPUS=1 VGGAE_GPU_IDS=0 EPOCHS=1 bash .../probe_e5_texture_recon.sh
export VGGAE_NUM_GPUS="${USER_NUM_GPUS:-4}"
export VGGAE_GPU_IDS="${USER_GPU_IDS:-0,1,2,3}"
# Avoid colliding with a leftover E4 master port (h200_env default 29540).
export VGGAE_MASTER_PORT="${USER_MASTER_PORT:-29550}"

# ----------------------------- editable settings -----------------------------
TEX_MODE="${TEX_MODE:-oracle}"
TEX_PACK="${TEX_PACK:-avgpool}"       # avgpool | s2d
TEX_REG_MODE="${TEX_REG_MODE:-n01}"   # n01 | match_geo
FEAT_WEIGHT="${FEAT_WEIGHT:-0}"       # >0 enables VGGT feature-consistency loss
FEAT_FRAMES="${FEAT_FRAMES:-2}"

PROBE_NAME="e5_${TEX_MODE}_${TEX_PACK}"
if [[ "${TEX_REG_MODE}" != "n01" ]]; then
  PROBE_NAME="${PROBE_NAME}_${TEX_REG_MODE}"
fi
if [[ "${FEAT_WEIGHT}" != "0" ]]; then
  PROBE_NAME="${PROBE_NAME}_feat${FEAT_WEIGHT}"
fi
# Set PROBE_SUFFIX=smoke for throwaway runs so their checkpoints never get
# auto-resumed by a later real run with the same knob combination.
if [[ -n "${PROBE_SUFFIX:-}" ]]; then
  PROBE_NAME="${PROBE_NAME}_${PROBE_SUFFIX}"
fi
OUTPUT_DIR="${VGGAE_H200_RUN_ROOT}/probes/${PROBE_NAME}"
RESUME="${RESUME:-}"

EPOCHS="${EPOCHS:-40}"
EVAL_CLIPS="${EVAL_CLIPS:-32}"
MAX_VIDEOS="${MAX_VIDEOS:-0}"   # 0 = full train csv; set small (e.g. 64) for smoke
# Memory guide (H200 141G, measured 2026-07-14): batch 2 + feat loss ~ 75G on
# one card, so without feat loss batch 4 fits comfortably and is the default.
# With FEAT_WEIGHT>0 (R5) drop to BATCH_SIZE=2 ACCUM_STEPS=4. Keep
# batch*accum = 8 per card (global 32 on 4 cards) so LR needs no retuning.
BATCH_SIZE="${BATCH_SIZE:-4}"
ACCUM_STEPS="${ACCUM_STEPS:-2}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
GEO_DIM="${GEO_DIM:-256}"
TEX_DIM="${TEX_DIM:-256}"
TEX_BASE_CH="${TEX_BASE_CH:-64}"
LATENT_GRID="${LATENT_GRID:-18}"
DECODER_BASE_DIM="${DECODER_BASE_DIM:-384}"
LPIPS_WEIGHT="${LPIPS_WEIGHT:-0.1}"
TEX_REG_WEIGHT="${TEX_REG_WEIGHT:-0.01}"
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
echo "[E5] tex_mode=${TEX_MODE} pack=${TEX_PACK} reg=${TEX_REG_MODE}" \
     "feat=${FEAT_WEIGHT} -> ${OUTPUT_DIR} (${VGGAE_NUM_GPUS} GPUs)"

h200_torchrun "${VGGAE_PROJECT}/probe_e5_texture_recon.py" \
  --tex_mode "${TEX_MODE}" \
  --tex_pack "${TEX_PACK}" \
  --tex_reg_mode "${TEX_REG_MODE}" \
  --probe_name "${PROBE_NAME}" \
  --csv "${VGGAE_TRAIN_10K_CSV}" \
  --video_root "${VGGAE_VIDEO_ROOT}" \
  --eval_csv "${VGGAE_EVAL_CSV}" \
  --eval_video_root "${VGGAE_VIDEO_ROOT}" \
  --encoder_ckpt "${VGGAE_ENCODER_CKPT}" \
  --max_videos "${MAX_VIDEOS}" \
  --levels 4 11 17 23 \
  --geo_dim "${GEO_DIM}" \
  --tex_dim "${TEX_DIM}" \
  --tex_base_ch "${TEX_BASE_CH}" \
  --latent_grid "${LATENT_GRID}" \
  --decoder_base_dim "${DECODER_BASE_DIM}" \
  --lambda_l1 1.0 --lambda_lpips "${LPIPS_WEIGHT}" \
  --lambda_feat "${FEAT_WEIGHT}" --feat_frames "${FEAT_FRAMES}" \
  --lambda_tex_reg "${TEX_REG_WEIGHT}" \
  --batch_size "${BATCH_SIZE}" --accum_steps "${ACCUM_STEPS}" \
  --epochs "${EPOCHS}" --lr "${LEARNING_RATE}" --wd 1e-2 \
  --warmup_steps 200 --ema_decay 0.999 \
  --use_checkpoint "${USE_CHECKPOINT}" \
  --frames_chunk_size "${FRAMES_CHUNK_SIZE}" \
  --seq_len 8 --target_size 518 --clip_duration_seconds 1.0 \
  --num_workers 8 --dtype bf16 \
  --output_dir "${OUTPUT_DIR}" \
  --eval_every 2 --save_every 5 --log_every 50 \
  --eval_clips "${EVAL_CLIPS}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee -a "${OUTPUT_DIR}/logs/train.log"
