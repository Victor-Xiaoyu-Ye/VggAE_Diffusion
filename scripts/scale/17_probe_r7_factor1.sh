#!/bin/bash
# Factor-1 R7 codec/cache ceiling probe. Uses independent namespaces and never
# overwrites the accepted t2/c192 v3 artifacts.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STAGES="${STAGES:-codec,joint,gate}"
R7_NAMESPACE="${R7_NAMESPACE:-r7_t1_c192_probe_v2}"
CACHE_VERSION="${R7_CACHE_VERSION:-${R7_NAMESPACE}_seq9_frame_channel_v1}"

want() { [[ ",${STAGES}," == *",$1,"* ]]; }

if want codec; then
  PROBE_CONTRACT=1 TEMPORAL_FACTOR=1 R7_NAMESPACE="${R7_NAMESPACE}" \
    PHASE=codec GATE_PSNR="${GATE_PSNR:-24.5}" \
    GATE_LPIPS="${GATE_LPIPS:-0.12}" \
    GATE_GEO_MOTION_COSINE="${GATE_GEO_MOTION_COSINE:-0.95}" \
    bash "${SCRIPT_DIR}/11_train_causal_tokenizer.sh"
fi

if want joint; then
  PROBE_CONTRACT=1 TEMPORAL_FACTOR=1 R7_NAMESPACE="${R7_NAMESPACE}" \
    PHASE=joint GATE_PSNR="${GATE_PSNR:-24.5}" \
    GATE_LPIPS="${GATE_LPIPS:-0.12}" \
    GATE_GEO_MOTION_COSINE="${GATE_GEO_MOTION_COSINE:-0.95}" \
    bash "${SCRIPT_DIR}/11_train_causal_tokenizer.sh"
fi

if want gate; then
  PROBE_CONTRACT=1 TEMPORAL_FACTOR=1 R7_NAMESPACE="${R7_NAMESPACE}" \
    GATE_PSNR="${GATE_PSNR:-24.5}" GATE_LPIPS="${GATE_LPIPS:-0.12}" \
    GATE_GEO_MOTION_COSINE="${GATE_GEO_MOTION_COSINE:-0.95}" \
    STAGES=gate bash "${SCRIPT_DIR}/14_r7_recon_then_diffusion.sh"
fi

if want cache_train || want cache_eval || want cache_merge; then
  PROBE_CONTRACT=1 TEMPORAL_FACTOR=1 R7_NAMESPACE="${R7_NAMESPACE}" \
    GATE_PSNR="${GATE_PSNR:-24.5}" GATE_LPIPS="${GATE_LPIPS:-0.12}" \
    GATE_GEO_MOTION_COSINE="${GATE_GEO_MOTION_COSINE:-0.95}" \
    STAGES=gate bash "${SCRIPT_DIR}/14_r7_recon_then_diffusion.sh"
fi

if want cache_train; then
  PROBE_CONTRACT=1 TEMPORAL_FACTOR=1 R7_NAMESPACE="${R7_NAMESPACE}" \
    R7_CACHE_VERSION="${CACHE_VERSION}" MODE=train \
    bash "${SCRIPT_DIR}/12_cache_causal_latents.sh"
fi

if want cache_eval; then
  PROBE_CONTRACT=1 TEMPORAL_FACTOR=1 R7_NAMESPACE="${R7_NAMESPACE}" \
    R7_CACHE_VERSION="${CACHE_VERSION}" MODE=eval \
    bash "${SCRIPT_DIR}/12_cache_causal_latents.sh"
fi

if want cache_merge; then
  PROBE_CONTRACT=1 TEMPORAL_FACTOR=1 R7_NAMESPACE="${R7_NAMESPACE}" \
    R7_CACHE_VERSION="${CACHE_VERSION}" MODE=merge \
    bash "${SCRIPT_DIR}/12_cache_causal_latents.sh"
fi
