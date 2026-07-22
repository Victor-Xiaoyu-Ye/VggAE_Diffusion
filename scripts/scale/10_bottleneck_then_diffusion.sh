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
R6_CKPT="${BOTTLENECK_DIR}/checkpoint_latest.pt"

want() { [[ ",${STAGES}," == *",$1,"* ]]; }

# ---------------------------------------------------------------- stage 09 --
if want bottleneck; then
  echo "=== [chain] stage 09: R6 bottleneck (comp_dim=${COMP_DIM}) ==="
  COMP_DIM="${COMP_DIM}" bash "${SCRIPT_DIR}/09_train_latent_bottleneck.sh"
fi

# If this node skipped stage 09 (or after a restart), stage the R6 artifact.
if [[ ! -s "${R6_CKPT}" ]]; then
  mkdir -p "${BOTTLENECK_DIR}"
  "${PYTHON_BIN}" "${PROJECT}/scripts/moxing_transfer.py" \
    "${BOTTLENECK_REMOTE}/checkpoint_latest.pt" "${R6_CKPT}" 2>/dev/null \
  || "${PYTHON_BIN}" "${PROJECT}/scripts/moxing_transfer.py" \
    "${SCALE_MIRROR_ROOT}/latent_bottleneck_c${COMP_DIM}/checkpoint_latest.pt" \
    "${R6_CKPT}" 2>/dev/null || true
fi
if [[ ! -s "${R6_CKPT}" ]]; then
  echo "[chain] no R6 checkpoint found locally or on OBS — run stage" \
       "bottleneck first" >&2
  exit 1
fi

# ------------------------------------------------------------------- gate ---
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

# ---------------------------------------------------------------- stage 10 --
if want diffusion; then
  echo "=== [chain] stage 10: compressed diffusion (E9) ==="
  # E9 = stage-07 harness on the R6 checkpoint: latent_dim auto-detects 128
  # via has_bottleneck; MIRA-informed width + RAE time shift via env.
  DUAL_AE_CKPT="${R6_CKPT}" \
  PROBE_SUFFIX="c${COMP_DIM}" \
  MODEL_DIM="${MODEL_DIM:-1152}" \
  SPATIAL_DEPTH="${SPATIAL_DEPTH:-10}" \
  TEMPORAL_DEPTH="${TEMPORAL_DEPTH:-6}" \
  NUM_HEADS="${NUM_HEADS:-16}" \
  TIME_SHIFT_ALPHA="${TIME_SHIFT_ALPHA:-1.5}" \
  BATCH_SIZE="${BATCH_SIZE:-2}" \
  MAX_STEPS="${MAX_STEPS_DIFF:-6000}" \
  bash "${SCRIPT_DIR}/07_train_dual_diffusion.sh"
fi

echo "=== [chain] done ==="
