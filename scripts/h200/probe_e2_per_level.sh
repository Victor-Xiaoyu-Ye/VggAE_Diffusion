#!/bin/bash
# E2: Per-level reconstruction.
#
# Run the probe once per VGGT level (4, 11, 17, 23) with the same decoder.
# Identifies which level carries the most reconstructable information and
# tests the "deep LayerNorm flattens high-frequency" hypothesis.
#
# Outputs are written to:
#   $VGGAE_H200_RUN_ROOT/probes/e2_level{L}/
#
# Compare per-level PSNR in metrics.jsonl after all four finish.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/h200_env.sh"
ensure_h200_splits

# ----------------------------- editable settings -----------------------------
LEVELS_TO_PROBE="${LEVELS_TO_PROBE:-4 11 17 23}"
EPOCHS="${EPOCHS:-30}"
EVAL_CLIPS="${EVAL_CLIPS:-32}"
BATCH_SIZE="${BATCH_SIZE:-2}"
ACCUM_STEPS="${ACCUM_STEPS:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
DECODER_BASE_DIM="${DECODER_BASE_DIM:-384}"
LATENT_DIM="${LATENT_DIM:-512}"
# Compress 37 -> 18 to match the production tokenizer grid; set to 37 to
# skip compression and measure the per-level raw ceiling at full resolution.
LATENT_GRID="${LATENT_GRID:-18}"
USE_PIXEL_SHUFFLE="${USE_PIXEL_SHUFFLE:-0}"
# Memory knobs (see probe_e1_raw_recon.sh for rationale).
USE_CHECKPOINT="${USE_CHECKPOINT:-1}"
MAX_FEATURE_GRID="${MAX_FEATURE_GRID:-296}"
FRAMES_CHUNK_SIZE="${FRAMES_CHUNK_SIZE:-4}"
# ----------------------------------------------------------------------------

for LVL in ${LEVELS_TO_PROBE}; do
  PROBE_NAME="e2_level${LVL}"
  OUTPUT_DIR="${VGGAE_H200_RUN_ROOT}/probes/${PROBE_NAME}"
  RESUME_ARGS=()
  if [[ -s "${OUTPUT_DIR}/checkpoint_latest.pt" ]]; then
    RESUME_ARGS+=(--resume "${OUTPUT_DIR}/checkpoint_latest.pt")
  fi
  mkdir -p "${OUTPUT_DIR}/logs"
  echo "[E2] level ${LVL} -> ${OUTPUT_DIR}"

  h200_torchrun "${VGGAE_PROJECT}/probe_feature_recon.py" \
    --mode per_level \
    --probe_name "${PROBE_NAME}" \
    --csv "${VGGAE_TRAIN_10K_CSV}" \
    --video_root "${VGGAE_VIDEO_ROOT}" \
    --eval_csv "${VGGAE_EVAL_CSV}" \
    --eval_video_root "${VGGAE_VIDEO_ROOT}" \
    --encoder_ckpt "${VGGAE_ENCODER_CKPT}" \
    --levels "${LVL}" \
    --per_level "${LVL}" \
    --latent_dim "${LATENT_DIM}" \
    --latent_grid "${LATENT_GRID}" \
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
    "${RESUME_ARGS[@]}" \
    2>&1 | tee -a "${OUTPUT_DIR}/logs/train.log"
done

echo "[E2] done. Compare per-level PSNR:"
for LVL in ${LEVELS_TO_PROBE}; do
  PROBE_NAME="e2_level${LVL}"
  OUTPUT_DIR="${VGGAE_H200_RUN_ROOT}/probes/${PROBE_NAME}"
  if [[ -s "${OUTPUT_DIR}/metrics.jsonl" ]]; then
    echo "  level ${LVL}:"
    tail -1 "${OUTPUT_DIR}/metrics.jsonl" | \
      "${VGGAE_PYTHON_BIN}" -c "import sys,json; d=json.loads(sys.stdin.read()); print(f'    psnr={d.get(\"psnr\",\"?\"):.2f} l1={d.get(\"l1\",\"?\"):.4f} lpips={d.get(\"lpips\",\"?\"):.4f}')" 2>/dev/null || true
  fi
done
