#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"

# ----------------------------- editable settings -----------------------------
REPORT_DIR="${RUN_ROOT}/reports"
# -----------------------------------------------------------------------------

python3 "${PROJECT}/collect_experiment_results.py" \
  --latent_contract "${RUN_ROOT}/diagnostics/latent_contract.json" \
  --run "geometry_autoencoder=${RUN_ROOT}/10k/geometry_autoencoder" \
  --run "i0_decoder_overfit=${RUN_ROOT}/validation/i0_decoder_overfit" \
  --run "diffusion_overfit=${RUN_ROOT}/validation/compact_diffusion_overfit" \
  --run "i0_decoder_10k=${RUN_ROOT}/10k/i0_decoder" \
  --run "diffusion_10k=${RUN_ROOT}/10k/compact_diffusion" \
  --output_dir "${REPORT_DIR}"
