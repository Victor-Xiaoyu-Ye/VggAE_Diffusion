#!/bin/bash
# H200/local factor-2 vs factor-4 codec probes using the real R7 trainer.
# Required env: TRAIN_CSV, EVAL_CSV, VIDEO_ROOT, ENCODER_CKPT, DUAL_AE_CKPT.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT=$(cd "${SCRIPT_DIR}/../.." && pwd)
PYTHON_BIN="${PYTHON_BIN:-python}"
: "${TRAIN_CSV:?set TRAIN_CSV}"; : "${EVAL_CSV:?set EVAL_CSV}"
: "${VIDEO_ROOT:?set VIDEO_ROOT}"; : "${ENCODER_CKPT:?set ENCODER_CKPT}"
: "${DUAL_AE_CKPT:?set DUAL_AE_CKPT}"
for FACTOR in 2 4; do
  if [[ "${FACTOR}" -eq 2 ]]; then GEO=96; TEX=96; else GEO=128; TEX=128; fi
  "${PYTHON_BIN}" "${PROJECT}/train_causal_dual_tokenizer.py" \
    --csv "${TRAIN_CSV}" --eval_csv "${EVAL_CSV}" --video_root "${VIDEO_ROOT}" \
    --encoder_ckpt "${ENCODER_CKPT}" --dual_ae_ckpt "${DUAL_AE_CKPT}" \
    --temporal_factor "${FACTOR}" --geo_latent_dim "${GEO}" \
    --tex_latent_dim "${TEX}" --seq_len "${SEQ_LEN:-9}" --phase codec \
    --max_steps "${MAX_STEPS:-3000}" --batch_size "${BATCH_SIZE:-1}" \
    --accum_steps "${ACCUM_STEPS:-2}" --num_workers "${NUM_WORKERS:-4}" \
    --throughput_divisor "${THROUGHPUT_DIVISOR:-20}" \
    --output_dir "${OUTPUT_ROOT:-outputs/h200}/r7_t${FACTOR}_c$((GEO+TEX))_codec"
done
