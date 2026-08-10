#!/bin/bash
# R7 reconstruction -> diffusion chain. Every invocation is run by all nodes.
# Examples:
#   STAGES=codec,joint,gate,cache,diffusion,sample bash scripts/scale/14_r7_recon_then_diffusion.sh
#   STAGES=gate,cache,diffusion bash scripts/scale/14_r7_recon_then_diffusion.sh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/spatialvid.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"

STAGES="${STAGES:-codec,joint,gate,cache,diffusion,sample}"
TEMPORAL_FACTOR="${TEMPORAL_FACTOR:-2}"
LATENT_DIM="${LATENT_DIM:-192}"
R7_NAMESPACE="${R7_NAMESPACE:-r7_t${TEMPORAL_FACTOR}_c${LATENT_DIM}_v3}"
JOINT_DIR="${SCALE_ROOT}/${R7_NAMESPACE}/joint"
JOINT_REMOTE="${SCALE_REMOTE_ROOT}/${R7_NAMESPACE}/joint"
JOINT_MIRROR="${SCALE_MIRROR_ROOT}/${R7_NAMESPACE}/joint"
R7_BEST="${R7_CKPT:-${JOINT_DIR}/checkpoint_best.pt}"
R7_BEST_URL="${R7_CKPT_URL:-${JOINT_REMOTE}/checkpoint_best.pt}"
R7_BEST_MIRROR_URL="${R7_CKPT_MIRROR_URL:-${JOINT_MIRROR}/checkpoint_best.pt}"
GATE_METRICS="${JOINT_DIR}/metrics.jsonl"
GATE_MARKER="${JOINT_DIR}/gate_passed.json"
GATE_MARKER_URL="${JOINT_REMOTE}/gate_passed.json"
GATE_MARKER_MIRROR_URL="${JOINT_MIRROR}/gate_passed.json"
GATE_PSNR="${GATE_PSNR:-23.9}"
GATE_LPIPS="${GATE_LPIPS:-0.13}"
GATE_BOUNDARY_RATIO="${GATE_BOUNDARY_RATIO:-1.10}"
GATE_GEO_MOTION_COSINE="${GATE_GEO_MOTION_COSINE:-0.95}"
BARRIER_PORT="${BARRIER_PORT:-29690}"
CACHE_NUM_PARTITIONS="${CACHE_NUM_PARTITIONS:-1}"
ALLOW_DIAGNOSTIC_DIFFUSION="${ALLOW_DIAGNOSTIC_DIFFUSION:-0}"
DIAGNOSTIC_MIN_GEO_MOTION_COSINE="${DIAGNOSTIC_MIN_GEO_MOTION_COSINE:-0.82}"
DIAGNOSTIC_DIFFUSION_NAMESPACE="${DIAGNOSTIC_DIFFUSION_NAMESPACE:-r7_diffusion_t2_c192_ctx1_fut4_v3_diag}"

configure_modelarts_distributed
require_scale_cluster
require_output_url
want() { [[ ",${STAGES}," == *",$1,"* ]]; }
barrier() { MASTER_PORT="${BARRIER_PORT}" run_distributed_barrier; BARRIER_PORT=$((BARRIER_PORT + 1)); }

stage_r7_checkpoint() {
  local local_path=$1 remote_path=$2 mirror_path=$3 label=$4
  # Force all non-publisher nodes to consume the just-published durable object.
  if [[ "${NODE_RANK}" -ne 0 ]]; then rm -f "${local_path}"; fi
  ensure_local_checkpoint "${local_path}" "${remote_path}" "${label}" "${mirror_path}"
}

stage_r7_best() {
  stage_r7_checkpoint "${R7_BEST}" "${R7_BEST_URL}" \
    "${R7_BEST_MIRROR_URL}" "accepted R7 best checkpoint"
}

verify_gate_marker() {
  stage_r7_best
  # Always fetch the durable publication rather than trusting node-local state.
  rm -f "${GATE_MARKER}"
  ensure_local_checkpoint "${GATE_MARKER}" "${GATE_MARKER_URL}" \
    "R7 passed-gate marker" "${GATE_MARKER_MIRROR_URL}"
  "${PYTHON_BIN}" - "${GATE_MARKER}" "${R7_BEST}" \
      "${GATE_PSNR}" "${GATE_LPIPS}" "${GATE_BOUNDARY_RATIO}" \
      "${GATE_GEO_MOTION_COSINE}" <<'PY'
import hashlib, json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    marker = json.load(stream)
assert marker.get("passed") is True, marker
expected_limits = tuple(map(float, sys.argv[3:7]))
assert tuple(marker.get("limits", ())) == expected_limits, (
    "gate thresholds changed; rerun STAGES=gate", marker.get("limits"),
    expected_limits)
digest = hashlib.sha256()
with open(sys.argv[2], "rb") as stream:
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(block)
assert marker.get("checkpoint_sha256") == digest.hexdigest(), (
    "gate marker belongs to different checkpoint content",
    marker.get("checkpoint_sha256"), digest.hexdigest())
PY
}

verify_diagnostic_checkpoint() {
  [[ "${ALLOW_DIAGNOSTIC_DIFFUSION}" == 1 ]] || {
    echo "Diagnostic diffusion requires ALLOW_DIAGNOSTIC_DIFFUSION=1." >&2
    return 1
  }
  stage_r7_best
  # This is deliberately not a passed-gate marker. It permits only an explicitly
  # named diagnostic diffusion from a causal, RGB-qualified checkpoint.
  "${PYTHON_BIN}" "${PROJECT}/probe_r7_causality.py" \
    --factors "${TEMPORAL_FACTOR}" --checkpoint "${R7_BEST}"
  "${PYTHON_BIN}" - "${R7_BEST}" "${GATE_PSNR}" "${GATE_LPIPS}" \
      "${GATE_BOUNDARY_RATIO}" "${DIAGNOSTIC_MIN_GEO_MOTION_COSINE}" <<'PY'
import sys, torch
path = sys.argv[1]
psnr, lpips, boundary, geo = map(float, sys.argv[2:6])
artifact = torch.load(path, map_location="cpu", weights_only=False)
best = artifact.get("best_state", {})
row = best.get("metrics") or {}
required = ("eval/psnr", "eval/lpips", "eval/boundary_ratio",
            "eval/geo_motion_cosine")
missing = [key for key in required if key not in row]
if missing:
    raise SystemExit(f"diagnostic checkpoint metrics missing: {missing}")
checks = {
    "psnr": row["eval/psnr"] >= psnr,
    "lpips": row["eval/lpips"] <= lpips,
    "boundary": row["eval/boundary_ratio"] <= boundary,
    "diagnostic_geo_motion": row["eval/geo_motion_cosine"] >= geo,
}
print("diagnostic checkpoint checks:", checks, row)
if not all(checks.values()):
    raise SystemExit(2)
PY
}

if want codec; then
  echo "=== R7 codec (formal v2, <=6k) ==="
  PHASE=codec R7_NAMESPACE="${R7_NAMESPACE}" \
    bash "${SCRIPT_DIR}/11_train_causal_tokenizer.sh"
  barrier
  stage_r7_checkpoint \
    "${SCALE_ROOT}/${R7_NAMESPACE}/codec/checkpoint_best.pt" \
    "${SCALE_REMOTE_ROOT}/${R7_NAMESPACE}/codec/checkpoint_best.pt" \
    "${SCALE_MIRROR_ROOT}/${R7_NAMESPACE}/codec/checkpoint_best.pt" \
    "published codec-best checkpoint"
  barrier
fi

if want joint; then
  # 11 stages codec best on each node; the barrier prevents reading it before
  # node 0 has finished the synchronous OBS/mirror publication.
  echo "=== R7 joint (codec-best init, <=12k) ==="
  PHASE=joint R7_NAMESPACE="${R7_NAMESPACE}" \
    bash "${SCRIPT_DIR}/11_train_causal_tokenizer.sh"
  barrier
  stage_r7_best
  barrier
fi

if want gate; then
  echo "=== R7 acceptance gate ==="
  stage_r7_best
  gate_status_file="${JOINT_DIR}/gate_status_node0.txt"
  rm -f "${gate_status_file}"
  gate_status=0
  if [[ "${NODE_RANK}" -eq 0 ]]; then
    set +e
    if [[ ! -s "${GATE_METRICS}" ]]; then
      mkdir -p "${JOINT_DIR}"
      for source in "${JOINT_REMOTE}/metrics.jsonl" "${JOINT_MIRROR}/metrics.jsonl"; do
        "${PYTHON_BIN}" "${PROJECT}/scripts/moxing_transfer.py" \
          "${source}" "${GATE_METRICS}" 2>/dev/null && break || true
      done
    fi
    # Causality/contract validation is an additional mandatory gate. It cannot
    # substitute for any missing or failed reconstruction-quality metric.
    "${PYTHON_BIN}" "${PROJECT}/probe_r7_causality.py" \
      --factors "${TEMPORAL_FACTOR}" --checkpoint "${R7_BEST}"
    probe_status=$?
    # checkpoint_best corresponds to best_state.metrics, not necessarily the
    # last metrics row. Prefer it; fall back to the row at best_state.step.
    "${PYTHON_BIN}" - "${R7_BEST}" "${GATE_METRICS}" "${GATE_MARKER}" \
        "${GATE_PSNR}" "${GATE_LPIPS}" "${GATE_BOUNDARY_RATIO}" \
        "${GATE_GEO_MOTION_COSINE}" <<'PY'
import hashlib, json, sys, torch
ckpt, metrics_path, marker = sys.argv[1:4]
limits = tuple(map(float, sys.argv[4:8]))
artifact = torch.load(ckpt, map_location="cpu", weights_only=False)
best = artifact.get("best_state", {})
row = best.get("metrics") or {}
step = best.get("step")
if not row and step is not None:
    try:
        with open(metrics_path, encoding="utf-8") as stream:
            for line in stream:
                candidate = json.loads(line)
                if candidate.get("step") == step and "eval/psnr" in candidate:
                    row = candidate
    except FileNotFoundError:
        pass
required = ("eval/psnr", "eval/lpips", "eval/boundary_ratio",
            "eval/geo_motion_cosine")
missing = [key for key in required if key not in row]
if missing:
    print(f"quality gate metrics missing for checkpoint_best: {missing}", file=sys.stderr)
    raise SystemExit(3)
checks = {
    "psnr": row["eval/psnr"] >= limits[0],
    "lpips": row["eval/lpips"] <= limits[1],
    "boundary": row["eval/boundary_ratio"] <= limits[2],
    "geo_motion": row["eval/geo_motion_cosine"] >= limits[3],
}
digest = hashlib.sha256()
with open(ckpt, "rb") as stream:
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(block)
payload = {"checkpoint_sha256": digest.hexdigest(), "step": step,
           "metrics": row, "causality_probe": True, "checks": checks,
           "limits": list(limits), "passed": all(checks.values())}
with open(marker, "w", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
print(json.dumps(payload, sort_keys=True))
raise SystemExit(0 if payload["passed"] else 2)
PY
    metric_status=$?
    gate_status=$(( probe_status != 0 ? probe_status : metric_status ))
    if [[ "${gate_status}" -ne 0 ]]; then
      printf '{"passed":false,"status":%d}\n' "${gate_status}" > "${GATE_MARKER}"
    fi
    "${PYTHON_BIN}" "${PROJECT}/scripts/moxing_transfer.py" \
      "${GATE_MARKER}" "${GATE_MARKER_URL}"
    publish_status=$?
    if [[ "${publish_status}" -eq 0 && \
          "${GATE_MARKER_MIRROR_URL}" != "${GATE_MARKER_URL}" ]]; then
      "${PYTHON_BIN}" "${PROJECT}/scripts/moxing_transfer.py" \
        "${GATE_MARKER}" "${GATE_MARKER_MIRROR_URL}"
      publish_status=$?
    fi
    gate_status=$(( gate_status != 0 ? gate_status : publish_status ))
    set -e
  fi
  # Every node reaches the barrier; all then validate the same durable marker.
  barrier
  verify_gate_marker
  barrier
fi

if want cache; then
  if [[ "${ALLOW_DIAGNOSTIC_DIFFUSION}" == 1 ]]; then
    verify_diagnostic_checkpoint
  else
    verify_gate_marker
  fi
  echo "=== R7 durable train/eval cache ==="
  # Run every train partition serially. All nodes participate in each torchrun;
  # a barrier separates durable partition publication before the next launch.
  for ((partition_id=0; partition_id<CACHE_NUM_PARTITIONS; partition_id++)); do
    echo "=== R7 train cache partition ${partition_id}/${CACHE_NUM_PARTITIONS} ==="
    MODE=train R7_NAMESPACE="${R7_NAMESPACE}" R7_CKPT="${R7_BEST}" \
      R7_CKPT_URL="${R7_BEST_URL}" R7_CKPT_MIRROR_URL="${R7_BEST_MIRROR_URL}" \
      CACHE_PARTITION_ID="${partition_id}" \
      CACHE_NUM_PARTITIONS="${CACHE_NUM_PARTITIONS}" \
      bash "${SCRIPT_DIR}/12_cache_causal_latents.sh"
    barrier
  done
  MODE=eval R7_NAMESPACE="${R7_NAMESPACE}" R7_CKPT="${R7_BEST}" \
    R7_CKPT_URL="${R7_BEST_URL}" R7_CKPT_MIRROR_URL="${R7_BEST_MIRROR_URL}" \
    bash "${SCRIPT_DIR}/12_cache_causal_latents.sh"
  barrier
  MODE=merge R7_NAMESPACE="${R7_NAMESPACE}" \
    CACHE_NUM_PARTITIONS="${CACHE_NUM_PARTITIONS}" \
    bash "${SCRIPT_DIR}/12_cache_causal_latents.sh"
  barrier
fi

if want diffusion; then
  if [[ "${ALLOW_DIAGNOSTIC_DIFFUSION}" == 1 ]]; then
    verify_diagnostic_checkpoint
    echo "=== R7 diagnostic diffusion (not acceptance-promoted) ==="
    MODE=train R7_NAMESPACE="${R7_NAMESPACE}" R7_CKPT="${R7_BEST}" \
      R7_CKPT_URL="${R7_BEST_URL}" R7_CKPT_MIRROR_URL="${R7_BEST_MIRROR_URL}" \
      DIFFUSION_NAMESPACE="${DIAGNOSTIC_DIFFUSION_NAMESPACE}" \
      bash "${SCRIPT_DIR}/13_train_causal_video_diffusion.sh"
  else
    verify_gate_marker
    echo "=== R7 production diffusion ==="
    MODE=train R7_NAMESPACE="${R7_NAMESPACE}" R7_CKPT="${R7_BEST}" \
      R7_CKPT_URL="${R7_BEST_URL}" R7_CKPT_MIRROR_URL="${R7_BEST_MIRROR_URL}" \
      bash "${SCRIPT_DIR}/13_train_causal_video_diffusion.sh"
  fi
  barrier
fi

if want sample; then
  if [[ "${ALLOW_DIAGNOSTIC_DIFFUSION}" == 1 ]]; then
    verify_diagnostic_checkpoint
    echo "=== R7 diagnostic best-EMA sample ==="
    MODE=sample R7_NAMESPACE="${R7_NAMESPACE}" R7_CKPT="${R7_BEST}" \
      R7_CKPT_URL="${R7_BEST_URL}" R7_CKPT_MIRROR_URL="${R7_BEST_MIRROR_URL}" \
      DIFFUSION_NAMESPACE="${DIAGNOSTIC_DIFFUSION_NAMESPACE}" \
      bash "${SCRIPT_DIR}/13_train_causal_video_diffusion.sh"
  else
    verify_gate_marker
    echo "=== R7 best-EMA sample ==="
    MODE=sample R7_NAMESPACE="${R7_NAMESPACE}" R7_CKPT="${R7_BEST}" \
      R7_CKPT_URL="${R7_BEST_URL}" R7_CKPT_MIRROR_URL="${R7_BEST_MIRROR_URL}" \
      bash "${SCRIPT_DIR}/13_train_causal_video_diffusion.sh"
  fi
  barrier
fi

echo "=== R7 chain done: ${STAGES} ==="
