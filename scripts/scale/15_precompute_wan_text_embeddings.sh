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
SPATIALVID_ANNO_DIR="${SPATIALVID_ANNO_DIR:-${LOCAL_CACHE_ROOT}/annotations/spatialvid_oft}"
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
  # Preferred path: fetch the durable prebuilt index from OBS.
  mkdir -p "$(dirname "${ANNOTATION_INDEX}")"
  "${PYTHON_BIN}" "${PROJECT}/scripts/moxing_transfer.py" \
    "${ANNOTATION_INDEX_URL}" "${ANNOTATION_INDEX}" 2>/dev/null || true
fi
if [[ ! -s "${ANNOTATION_INDEX}" ]]; then
  # Fallback: stage the annotations tree and build the index here.
  # copy_parallel on a missing OBS prefix is a silent no-op, so validate that
  # the staged directory actually contains group subdirectories.
  if [[ ! -d "${SPATIALVID_ANNO_DIR}" ]] \
      || [[ -z "$(ls -A "${SPATIALVID_ANNO_DIR}" 2>/dev/null)" ]]; then
    echo "Staging annotations from ${SPATIALVID_ANNO_URL}"
    "${PYTHON_BIN}" "${PROJECT}/scripts/moxing_transfer.py" --directory \
      "${SPATIALVID_ANNO_URL}" "${SPATIALVID_ANNO_DIR}"
  fi
  if [[ ! -d "${SPATIALVID_ANNO_DIR}" ]] \
      || [[ -z "$(ls -A "${SPATIALVID_ANNO_DIR}" 2>/dev/null)" ]]; then
    echo "Annotation staging produced no files from ${SPATIALVID_ANNO_URL}." >&2
    echo "Either the OBS mirror lacks the annotations tree, or the prefix" >&2
    echo "differs. Fix SPATIALVID_ANNO_URL, or build annotation_index.json" >&2
    echo "on the A100 box (scripts/build_annotation_index.sh) and upload it" >&2
    echo "to ${ANNOTATION_INDEX_URL}." >&2
    exit 1
  fi
  "${PYTHON_BIN}" "${PROJECT}/data/annotation_index.py" \
    --csv_path "${SPATIALVID_METADATA}" \
    --anno_dir "${SPATIALVID_ANNO_DIR}" \
    --out_path "${ANNOTATION_INDEX}"
  # Publish the built index durably so later runs skip the tree copy.
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
