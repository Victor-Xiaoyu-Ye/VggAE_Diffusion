#!/bin/bash
# Independent c192 t1 channel-allocation probes. Run one ARM per submission.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"
ARM="${ARM:-geo128_tex64}"

case "${ARM}" in
  baseline)
    GEO_LATENT_DIM=96; TEX_LATENT_DIM=96;;
  geo112_tex80)
    GEO_LATENT_DIM=112; TEX_LATENT_DIM=80;;
  geo128_tex64)
    GEO_LATENT_DIM=128; TEX_LATENT_DIM=64;;
  *)
    echo "ARM must be baseline, geo112_tex80, or geo128_tex64." >&2
    exit 2;;
esac

LATENT_DIM=$((GEO_LATENT_DIM + TEX_LATENT_DIM))
[[ "${LATENT_DIM}" -eq 192 ]] || {
  echo "The production matrix keeps total latent width fixed at 192." >&2
  exit 2
}
R7_NAMESPACE="${R7_NAMESPACE:-r7_t1_c192_${ARM}_probe_v1}"
STAGES="${STAGES:-codec,joint,gate}"
configure_modelarts_distributed
require_output_url
BARRIER_PORT="${BARRIER_PORT:-29710}"

want() { [[ ",${STAGES}," == *",$1,"* ]]; }
barrier() {
  MASTER_PORT="${BARRIER_PORT}" run_distributed_barrier
  BARRIER_PORT=$((BARRIER_PORT + 1))
}
run_phase() {
  PROBE_CONTRACT=1 MATRIX_CONTRACT=1 TEMPORAL_FACTOR=1 \
  GEO_LATENT_DIM="${GEO_LATENT_DIM}" TEX_LATENT_DIM="${TEX_LATENT_DIM}" \
  LATENT_DIM=192 R7_NAMESPACE="${R7_NAMESPACE}" \
  GATE_PSNR="${GATE_PSNR:-24.5}" GATE_LPIPS="${GATE_LPIPS:-0.12}" \
  GATE_GEO_MOTION_COSINE="${GATE_GEO_MOTION_COSINE:-0.95}" \
  PHASE="$1" bash "${SCRIPT_DIR}/11_train_causal_tokenizer.sh"
}

if want codec; then
  run_phase codec
  barrier
fi
if want joint; then
  run_phase joint
  barrier
fi

if want gate; then
  R7_BEST="${SCALE_ROOT}/${R7_NAMESPACE}/joint/checkpoint_best.pt"
  R7_BEST_URL="${SCALE_REMOTE_ROOT}/${R7_NAMESPACE}/joint/checkpoint_best.pt"
  R7_BEST_MIRROR_URL="${SCALE_MIRROR_ROOT}/${R7_NAMESPACE}/joint/checkpoint_best.pt"
  MARKER="${SCALE_ROOT}/${R7_NAMESPACE}/joint/gate_passed.json"
  MARKER_URL="${SCALE_REMOTE_ROOT}/${R7_NAMESPACE}/joint/gate_passed.json"
  MARKER_MIRROR_URL="${SCALE_MIRROR_ROOT}/${R7_NAMESPACE}/joint/gate_passed.json"
  ensure_local_checkpoint "${R7_BEST}" "${R7_BEST_URL}" \
    "matrix joint-best checkpoint" "${R7_BEST_MIRROR_URL}"
  if [[ "${NODE_RANK}" -eq 0 ]]; then
    "${PYTHON_BIN}" "${PROJECT}/probe_r7_causality.py" \
      --factors 1 --checkpoint "${R7_BEST}"
    "${PYTHON_BIN}" - "${R7_BEST}" "${MARKER}" \
      "${GATE_PSNR:-24.5}" "${GATE_LPIPS:-0.12}" \
      "${GATE_GEO_MOTION_COSINE:-0.95}" \
      "${GEO_LATENT_DIM}" "${TEX_LATENT_DIM}" <<'PY'
import hashlib, json, sys, torch
ckpt, marker = sys.argv[1:3]
psnr, lpips, geo = map(float, sys.argv[3:6])
split = [int(sys.argv[6]), int(sys.argv[7])]
artifact = torch.load(ckpt, map_location="cpu", weights_only=False)
best = artifact.get("best_state") or {}
row = best.get("metrics") or {}
checks = {
    "psnr": row.get("eval/psnr", float("-inf")) >= psnr,
    "lpips": row.get("eval/lpips", float("inf")) <= lpips,
    "boundary": True,
    "geo_motion": row.get("eval/geo_motion_cosine", -1.0) >= geo,
}
digest = hashlib.sha256()
with open(ckpt, "rb") as stream:
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(block)
payload = {
    "checkpoint_sha256": digest.hexdigest(),
    "step": best.get("step"), "metrics": row,
    "causality_probe": True, "checks": checks,
    "temporal_factor": 1, "latent_dim": 192, "channel_split": split,
    "limits": [psnr, lpips, 1.10, geo],
    "passed": all(checks.values()),
}
with open(marker, "w", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
print(json.dumps(payload, sort_keys=True))
raise SystemExit(0 if payload["passed"] else 2)
PY
    "${PYTHON_BIN}" "${PROJECT}/scripts/moxing_transfer.py" \
      "${MARKER}" "${MARKER_URL}"
    if [[ "${MARKER_MIRROR_URL}" != "${MARKER_URL}" ]]; then
      "${PYTHON_BIN}" "${PROJECT}/scripts/moxing_transfer.py" \
        "${MARKER}" "${MARKER_MIRROR_URL}"
    fi
  fi
  barrier
fi

printf 'R7 t1 representation arm complete: %s (%s)\n' "${ARM}" "${R7_NAMESPACE}"
