#!/bin/bash
# Precompute UMT5-xxl text embeddings for the R7 Wan T2V diffusion stage.
# Node 0, single process. Builds annotation_index.json from the OFT mirror's
# annotations tree, encodes every train_10k/eval caption plus the empty CFG
# prompt with Wan's native T5, and publishes the sidecar to durable OBS.
# Example:
#   bash scripts/scale/15_precompute_wan_text_embeddings.sh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/spatialvid.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"

# The OFT 10k mirror: videos are the subset, metadata is the full CSV, and the
# annotations tree matches the local A100/H200 copies used by
# scripts/build_annotation_index.sh.
SPATIALVID_OFT_ROOT="${SPATIALVID_OFT_ROOT:-obs://yw-ads-training-gy1/data/external/personal/g00833899/y50046448/spatial-vid-hq-oft}"
SPATIALVID_METADATA_URL="${SPATIALVID_OFT_ROOT}/data/train/SpatialVID_HQ_metadata.csv"
SPATIALVID_VIDEO_ROOT="${SPATIALVID_OFT_ROOT}/videos/SpatialVid/HQ/videos"
SPATIALVID_METADATA="${LOCAL_CACHE_ROOT}/metadata/SpatialVID_HQ_metadata_oft.csv"
SPATIALVID_SPLIT_DIR="${RUN_ROOT}/metadata/spatialvid_oft_seed${SPLIT_SEED}"
SPATIALVID_TRAIN_10K_CSV="${SPATIALVID_SPLIT_DIR}/train_10k.csv"
SPATIALVID_EVAL_CSV="${SPATIALVID_SPLIT_DIR}/eval.csv"
SPATIALVID_ANNO_URL="${SPATIALVID_ANNO_URL:-${SPATIALVID_OFT_ROOT}/annotations/SpatialVID/annotations}"
ANNOTATION_INDEX="${ANNOTATION_INDEX:-${RUN_ROOT}/metadata/annotation_index_oft.json}"
# Durable prebuilt index (preferred): one small JSON instead of copying tens of
# thousands of caption.json files. Built once here, or uploaded manually from
# the A100 box (`ckpts/annotation_index.json` from build_annotation_index.sh).
ANNOTATION_INDEX_URL="${ANNOTATION_INDEX_URL:-${PERSISTENT_OBS_ROOT}/metadata/annotation_index_oft.json}"

WAN_T2V_13B_DIR="${WAN_T2V_13B_DIR:-${VGGAE_REF_ROOT}/Wan2.1-T2V-1.3B}"
TEXT_EMBEDDING_VERSION="${TEXT_EMBEDDING_VERSION:-umt5xxl_spatialvid_10k_v1}"
TEXT_EMBEDDING_OBS_DIR="${TEXT_EMBEDDING_OBS_DIR:-${PERSISTENT_OBS_ROOT}/text_embeddings/${TEXT_EMBEDDING_VERSION}}"
TEXT_LEN="${TEXT_LEN:-512}"
BATCH_SIZE="${BATCH_SIZE:-16}"

configure_modelarts_distributed
require_output_url
if [[ "${NODE_RANK}" -ne 0 ]]; then
  echo "Text-embedding precompute runs on node 0 only; node ${NODE_RANK} idle."
  exit 0
fi

# Idempotent chain stage: the sidecar is durable, keyed by its _SUCCESS
# marker. Re-running the chain must not redo the full UMT5 encode.
sidecar_published() {
  PYTHONPATH="${PROJECT}" "${PYTHON_BIN}" - \
    "${TEXT_EMBEDDING_OBS_DIR}/_SUCCESS" <<'PY'
import sys
from utils.moxing_io import remote_exists
sys.exit(0 if remote_exists(sys.argv[1]) else 1)
PY
}
if [[ "${FORCE_TEXT_EMBED:-0}" != 1 ]] && sidecar_published; then
  echo "Text-embedding sidecar already published: ${TEXT_EMBEDDING_OBS_DIR}"
  echo "Set FORCE_TEXT_EMBED=1 to rebuild."
  exit 0
fi
ensure_spatialvid_subset_splits

require_dir "${WAN_T2V_13B_DIR}" "Wan2.1-T2V-1.3B checkpoint directory"
for asset in "models_t5_umt5-xxl-enc-bf16.pth" "google/umt5-xxl"; do
  [[ -e "${WAN_T2V_13B_DIR}/${asset}" ]] || {
    echo "Wan T5 asset missing: ${WAN_T2V_13B_DIR}/${asset}" >&2
    echo "The staged snapshot must include the umt5-xxl encoder + tokenizer." >&2
    exit 1
  }
done

if [[ ! -s "${ANNOTATION_INDEX}" ]]; then
  # Try the durable prebuilt index first; else build it by reading the
  # caption.json objects directly from OBS for exactly the split video ids.
  # The tree is never copied: copy_parallel over tens of thousands of small
  # objects proved unreliable (silent partial/no-op), keyed reads are not.
  mkdir -p "$(dirname "${ANNOTATION_INDEX}")"
  "${PYTHON_BIN}" "${PROJECT}/scripts/moxing_transfer.py" \
    "${ANNOTATION_INDEX_URL}" "${ANNOTATION_INDEX}" 2>/dev/null || true
fi
if [[ ! -s "${ANNOTATION_INDEX}" ]]; then
  PYTHONPATH="${PROJECT}" "${PYTHON_BIN}" "${PROJECT}/data/annotation_index.py" \
    --csv_path "${SPATIALVID_TRAIN_10K_CSV}" \
    --csv_path "${SPATIALVID_EVAL_CSV}" \
    --anno_dir "${SPATIALVID_ANNO_URL}" \
    --out_path "${ANNOTATION_INDEX}" \
    --workers "${ANNOTATION_FETCH_WORKERS:-16}"
  # Publish the built index durably so later runs skip the OBS reads.
  "${PYTHON_BIN}" "${PROJECT}/scripts/moxing_transfer.py" \
    "${ANNOTATION_INDEX}" "${ANNOTATION_INDEX_URL}" || true
fi
require_file "${ANNOTATION_INDEX}" "annotation index"

PYTHONPATH="${PROJECT}" "${PYTHON_BIN}" \
  "${PROJECT}/precompute_wan_text_embeddings.py" \
  --wan_ckpt_dir "${WAN_T2V_13B_DIR}" \
  --annotation_index "${ANNOTATION_INDEX}" \
  --csv "${SPATIALVID_TRAIN_10K_CSV}" \
  --csv "${SPATIALVID_EVAL_CSV}" \
  --output_dir "${TEXT_EMBEDDING_OBS_DIR}" \
  --text_len "${TEXT_LEN}" \
  --batch_size "${BATCH_SIZE}"

echo "Text-embedding sidecar is durable at ${TEXT_EMBEDDING_OBS_DIR}"
