#!/bin/bash
# E4: DPT-style reconstruction probe.
#
# Tests whether the proven VGGT-native DPT decoder (as in 4DLangRecon) works in
# our pipeline, and how much a compact-latent bottleneck (needed for diffusion)
# costs.
#
#   BOTTLENECK=1 (default, production path):
#     frozen VGGT -> CompactCompressor -> z[cdim,g,g] -> DPTLatentDecoder -> RGB
#   BOTTLENECK=0 (control / ceiling on our data):
#     frozen VGGT -> DPTHead (output_dim=4, sigmoid) -> RGB   (reproduces
#     4DLangRecon's decoder directly on SpatialVID, no compact latent)
#
# Run both to isolate the bottleneck cost:
#   bash scripts/h200/probe_e4_dpt_recon.sh                 # bottleneck=1
#   BOTTLENECK=0 bash scripts/h200/probe_e4_dpt_recon.sh    # control
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/h200_env.sh"
ensure_h200_splits

# E4 uses all 4 H200 cards regardless of the single-card default in h200_env.sh.
# Edit here to change; these override the sourced topology.
export VGGAE_NUM_GPUS=4
export VGGAE_GPU_IDS=0,1,2,3

# ----------------------------- editable settings -----------------------------
BOTTLENECK="${BOTTLENECK:-1}"
if [[ "${BOTTLENECK}" -eq 0 ]]; then
  PROBE_NAME="e4_dpt_bottleneck0_control"
else
  PROBE_NAME="e4_dpt_bottleneck1"
fi
OUTPUT_DIR="${VGGAE_H200_RUN_ROOT}/probes/${PROBE_NAME}"
RESUME="${RESUME:-}"

EPOCHS="${EPOCHS:-40}"
EVAL_CLIPS="${EVAL_CLIPS:-32}"
BATCH_SIZE="${BATCH_SIZE:-2}"
ACCUM_STEPS="${ACCUM_STEPS:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
CDIM="${CDIM:-256}"
LATENT_GRID="${LATENT_GRID:-18}"
DPT_FEATURES="${DPT_FEATURES:-256}"
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
echo "[E4] bottleneck=${BOTTLENECK} -> ${OUTPUT_DIR} (${VGGAE_NUM_GPUS} GPUs)"

h200_torchrun "${VGGAE_PROJECT}/probe_e4_dpt_recon.py" \
  --bottleneck "${BOTTLENECK}" \
  --probe_name "${PROBE_NAME}" \
  --csv "${VGGAE_TRAIN_10K_CSV}" \
  --video_root "${VGGAE_VIDEO_ROOT}" \
  --eval_csv "${VGGAE_EVAL_CSV}" \
  --eval_video_root "${VGGAE_VIDEO_ROOT}" \
  --encoder_ckpt "${VGGAE_ENCODER_CKPT}" \
  --levels 4 11 17 23 \
  --cdim "${CDIM}" \
  --latent_grid "${LATENT_GRID}" \
  --dpt_features "${DPT_FEATURES}" \
  --lambda_l1 1.0 --lambda_lpips "${LPIPS_WEIGHT}" \
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
