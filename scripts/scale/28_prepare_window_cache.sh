#!/bin/bash
# Reuse audited R7 cache infrastructure only; not stages 25/26 or their trainer.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
AE_VARIANT="${AE_VARIANT:-t2v2}"
case "${AE_VARIANT}" in
  t2v2) export R7_NAMESPACE=r7_t2_c192_v2 TEMPORAL_FACTOR=2 PROBE_CONTRACT=0 ;;
  t1geo112) export R7_NAMESPACE=r7_t1_c192_geo112_tex80_probe_v1 TEMPORAL_FACTOR=1 PROBE_CONTRACT=1 ;;
  t1equal) export R7_NAMESPACE=r7_t1_c192_probe_v1 TEMPORAL_FACTOR=1 PROBE_CONTRACT=1 ;;
  t2v3) export R7_NAMESPACE=r7_t2_c192_v3 TEMPORAL_FACTOR=2 PROBE_CONTRACT=0 ;;
  *) echo "Unknown AE_VARIANT=${AE_VARIANT}" >&2; exit 2 ;;
esac
# New namespace states this is the user's accepted historical-AE experiment,
# not a claim that the old geometry gate passed. Never rewrite old gate files.
export ALLOW_DIAGNOSTIC_CACHE=1
if [[ "${AE_VARIANT}" == t2v2 ]]; then DEFAULT_NORM=legacy; else DEFAULT_NORM=framewise; fi
export WINDOW_AE_NORM="${WINDOW_AE_NORM:-${DEFAULT_NORM}}"
export R7_CACHE_VERSION="${R7_CACHE_VERSION:-r7_window_${AE_VARIANT}_${WINDOW_AE_NORM}_diag_v2}"
export SAMPLES_PER_TAR="${SAMPLES_PER_TAR:-256}"
export STORE_I0=1
export INDEPENDENT_ANCHOR=1
if [[ "${MODE:-train}" == eval ]]; then export STORE_RGB=1; else export STORE_RGB=0; fi
exec bash "${SCRIPT_DIR}/12_cache_causal_latents.sh"
