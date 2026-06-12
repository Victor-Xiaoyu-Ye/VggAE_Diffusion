#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"

I0_CKPT="${OVERFIT_I0_DECODER_CKPT}"
LOG_DIR="${RUN_ROOT}/validation/compact_diffusion_overfit"
mkdir -p "${LOG_DIR}"

echo "[wait] Waiting for I0 overfit checkpoint: ${I0_CKPT}"
while [[ ! -f "${I0_CKPT}" ]]; do
  sleep 60
done
echo "[wait] Found ${I0_CKPT}, launching diffusion overfit..."
bash "${SCRIPT_DIR}/overfit_compact_diffusion.sh" 2>&1 | tee "${LOG_DIR}/train.log"
