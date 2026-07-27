#!/bin/bash
# Stage 10 chain: R6 bottleneck -> PSNR gate -> compressed diffusion.
#
# One command runs lever B end to end on the cluster:
#   1. Stage 09 (R6): finetune the 512->COMP_DIM bottleneck (~3k steps).
#   2. Gate: final eval PSNR must be >= GATE_DB (R5 24.41 - 0.5 allowance).
#   3. Stage 07' (E9): from-scratch DiT on the COMPRESSED latent with the
#      MIRA-informed width (MODEL_DIM 1152 on 128ch = 9x width ratio) and
#      the RAE dimension-dependent time shift.
#
#   bash scripts/scale/10_bottleneck_then_diffusion.sh
#   STAGES=diffusion bash scripts/scale/10_bottleneck_then_diffusion.sh  # skip R6
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/spatialvid.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"

COMP_DIM="${COMP_DIM:-128}"
GATE_DB="${GATE_DB:-23.9}"
STAGES="${STAGES:-bottleneck,diffusion}"
BOTTLENECK_DIR="${SCALE_ROOT}/latent_bottleneck_c${COMP_DIM}"
BOTTLENECK_REMOTE="${SCALE_REMOTE_ROOT}/latent_bottleneck_c${COMP_DIM}"
BOTTLENECK_MIRROR="${SCALE_MIRROR_ROOT}/latent_bottleneck_c${COMP_DIM}"
R6_CKPT="${BOTTLENECK_DIR}/checkpoint_latest.pt"

configure_modelarts_distributed

want() { [[ ",${STAGES}," == *",$1,"* ]]; }

# ---------------------------------------------------------------- stage 09 --
if want bottleneck; then
  echo "=== [chain] stage 09: R6 bottleneck (comp_dim=${COMP_DIM}) ==="
  COMP_DIM="${COMP_DIM}" bash "${SCRIPT_DIR}/09_train_latent_bottleneck.sh"
  # Nonzero nodes leave stage 09 before node 0 finishes its final synchronous
  # OBS writes. Wait until that publication is complete before any node reads.
  MASTER_PORT=29661 run_distributed_barrier
fi

# /cache is node-local: only node 0 owns the R6 file written by global rank 0.
# Every other node stages the published artifact using the standard current-
# output then persistent-mirror fallback before its local torchrun starts.
if [[ "${NODE_RANK}" -ne 0 ]]; then
  rm -f "${R6_CKPT}"
fi
ensure_local_checkpoint \
  "${R6_CKPT}" "${BOTTLENECK_REMOTE}/checkpoint_latest.pt" \
  "R6 bottleneck checkpoint" \
  "${BOTTLENECK_MIRROR}/checkpoint_latest.pt"

# ------------------------------------------------------------------- gate ---
# A diffusion-only restart reuses an already gated R6 checkpoint. Its fresh
# node-local output directory intentionally has no R6 metrics or samples, so
# do not gate on artifacts that belong to the previous bottleneck run.
if want bottleneck; then
  R6_PSNR=$("${PYTHON_BIN}" - "${BOTTLENECK_DIR}/metrics.jsonl" <<'PY'
import json, sys
psnr = None
try:
    with open(sys.argv[1]) as f:
        for line in f:
            row = json.loads(line)
            if 'psnr' in row:
                psnr = row['psnr']
except FileNotFoundError:
    pass
print(f'{psnr:.3f}' if psnr is not None else 'NONE')
PY
  )
  if [[ "${R6_PSNR}" == "NONE" ]]; then
    # metrics may live only on OBS after a node restart
    "${PYTHON_BIN}" "${PROJECT}/scripts/moxing_transfer.py" \
      "${BOTTLENECK_REMOTE}/metrics.jsonl" \
      "${BOTTLENECK_DIR}/metrics.jsonl" 2>/dev/null \
    || "${PYTHON_BIN}" "${PROJECT}/scripts/moxing_transfer.py" \
      "${BOTTLENECK_MIRROR}/metrics.jsonl" \
      "${BOTTLENECK_DIR}/metrics.jsonl" 2>/dev/null || true
    R6_PSNR=$("${PYTHON_BIN}" - "${BOTTLENECK_DIR}/metrics.jsonl" <<'PY'
import json, sys
psnr = None
try:
    with open(sys.argv[1]) as f:
        for line in f:
            row = json.loads(line)
            if 'psnr' in row:
                psnr = row['psnr']
except FileNotFoundError:
    pass
print(f'{psnr:.3f}' if psnr is not None else 'NONE')
PY
    )
  fi
  echo "=== [chain] R6 final PSNR = ${R6_PSNR} dB (gate ${GATE_DB}) ==="
  if [[ "${R6_PSNR}" == "NONE" ]] || \
     ! "${PYTHON_BIN}" -c "exit(0 if float('${R6_PSNR}') >= ${GATE_DB} else 1)"; then
    echo "[chain] GATE FAILED: bottleneck loses too much reconstruction." \
         "Inspect ${BOTTLENECK_REMOTE}/samples, then either raise COMP_DIM" \
         "(192) or extend MAX_STEPS and rerun stage bottleneck." >&2
    exit 1
  fi
else
  echo "=== [chain] reusing pre-gated R6 checkpoint; skip bottleneck gate ==="
fi

# ---------------------------------------------------------------- stage 10 --
if want diffusion; then
  echo "=== [chain] stage 10: compressed diffusion (E9-v3 spatiotemporal motion) ==="
  # E9-v3 combines clean z0, interleaved axial attention, and trajectory losses
  # while keeping the accepted R6 latent and DiT capacity fixed.
  DUAL_AE_CKPT="${R6_CKPT}" \
  DUAL_AE_CKPT_URL="${BOTTLENECK_REMOTE}/checkpoint_latest.pt" \
  DUAL_AE_CKPT_MIRROR_URL="${BOTTLENECK_MIRROR}/checkpoint_latest.pt" \
  PROBE_SUFFIX="${PROBE_SUFFIX_OVERRIDE:-c${COMP_DIM}_cf0_stmotion}" \
  CLEAN_FRAME0=1 \
  BLOCK_SCHEDULE=interleaved \
  LAMBDA_MOTION="${LAMBDA_MOTION:-0.10}" \
  LAMBDA_ACCEL="${LAMBDA_ACCEL:-0.05}" \
  LAMBDA_GEO_MOTION="${LAMBDA_GEO_MOTION:-0.05}" \
  AUX_WARMUP_STEPS="${AUX_WARMUP_STEPS:-1000}" \
  AUX_RAMP_STEPS="${AUX_RAMP_STEPS:-1000}" \
  AUX_T_MIN="${AUX_T_MIN:-0.60}" \
  AUTO_RESUME="${AUTO_RESUME:-1}" \
  MODEL_DIM="${MODEL_DIM:-1152}" \
  SPATIAL_DEPTH="${SPATIAL_DEPTH:-10}" \
  TEMPORAL_DEPTH="${TEMPORAL_DEPTH:-6}" \
  NUM_HEADS="${NUM_HEADS:-16}" \
  TIME_SHIFT_ALPHA="${TIME_SHIFT_ALPHA:-1.5}" \
  WARMUP_STEPS="${WARMUP_STEPS:-300}" \
  NUM_WORKERS="${NUM_WORKERS:-4}" \
  STAT_BATCHES="${STAT_BATCHES:-16}" \
  EVAL_CLIPS="${EVAL_CLIPS:-16}" \
  SAMPLE_CLIPS="${SAMPLE_CLIPS:-4}" \
  SAMPLE_STEPS="${SAMPLE_STEPS:-30}" \
  EVAL_EVERY="${EVAL_EVERY:-500}" \
  SAVE_EVERY="${SAVE_EVERY:-500}" \
  LOG_EVERY="${LOG_EVERY:-50}" \
  BATCH_SIZE="${BATCH_SIZE:-2}" \
  MAX_STEPS="${MAX_STEPS_DIFF:-6000}" \
  bash "${SCRIPT_DIR}/07_train_dual_diffusion.sh"
fi

echo "=== [chain] done ==="
