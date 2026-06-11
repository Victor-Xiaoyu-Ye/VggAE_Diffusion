#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Set any stage to 0 to skip it on the next launch.
RUN_LATENT_DIAGNOSTIC=1
RUN_I0_OVERFIT=1
RUN_DIFFUSION_OVERFIT=1

if [[ "${RUN_LATENT_DIAGNOSTIC}" == "1" ]]; then
  bash "${SCRIPT_DIR}/diagnose_latent_contract.sh"
fi
if [[ "${RUN_I0_OVERFIT}" == "1" ]]; then
  bash "${SCRIPT_DIR}/overfit_i0_autoencoder.sh"
fi
if [[ "${RUN_DIFFUSION_OVERFIT}" == "1" ]]; then
  bash "${SCRIPT_DIR}/overfit_compact_diffusion.sh"
fi
bash "${SCRIPT_DIR}/collect_results.sh"
