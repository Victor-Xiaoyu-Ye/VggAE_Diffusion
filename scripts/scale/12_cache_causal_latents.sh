#!/bin/bash
# Durable R7 t2/c192/seq9 cache wrapper.
# Actual cache CLI: partition/resume/tar options plus strict checkpoint/source
# representation inputs; each partition emits manifest.txt + stats.pt for merge.
# Examples:
#   MODE=train CACHE_PARTITION_ID=0 CACHE_NUM_PARTITIONS=4 bash scripts/scale/12_cache_causal_latents.sh
#   MODE=eval bash scripts/scale/12_cache_causal_latents.sh
#   MODE=merge CACHE_NUM_PARTITIONS=4 bash scripts/scale/12_cache_causal_latents.sh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/spatialvid.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"

MODE="${MODE:-train}"                         # train | eval | merge
TEMPORAL_FACTOR="${TEMPORAL_FACTOR:-2}"
LATENT_DIM="${LATENT_DIM:-192}"
SEQ_LEN="${SEQ_LEN:-9}"
R7_NAMESPACE="${R7_NAMESPACE:-r7_t${TEMPORAL_FACTOR}_c${LATENT_DIM}_v3}"
R7_ACCEPTED_PHASE="${R7_ACCEPTED_PHASE:-joint}"
CACHE_VERSION="${R7_CACHE_VERSION:-${R7_NAMESPACE}_seq${SEQ_LEN}_frame_channel_v1}"
R7_CACHE_OBS_ROOT="${R7_CACHE_OBS_ROOT:-${PERSISTENT_OBS_ROOT}/cache_latents/${CACHE_VERSION}}"
R7_CKPT="${R7_CKPT:-${SCALE_ROOT}/${R7_NAMESPACE}/${R7_ACCEPTED_PHASE}/checkpoint_best.pt}"
R7_CKPT_URL="${R7_CKPT_URL:-${SCALE_REMOTE_ROOT}/${R7_NAMESPACE}/${R7_ACCEPTED_PHASE}/checkpoint_best.pt}"
R7_CKPT_MIRROR_URL="${R7_CKPT_MIRROR_URL:-${SCALE_MIRROR_ROOT}/${R7_NAMESPACE}/${R7_ACCEPTED_PHASE}/checkpoint_best.pt}"
DUAL_AE_CKPT="${DUAL_AE_CKPT:-${VGGAE_REF_ROOT}/checkpoints/e5_dual_stream_r5.pt}"
DUAL_AE_CKPT_URL="${DUAL_AE_CKPT_URL:-}"
DUAL_AE_CKPT_MIRROR_URL="${DUAL_AE_CKPT_MIRROR_URL:-}"
SPATIALVID_OFT_ROOT="${SPATIALVID_OFT_ROOT:-obs://yw-ads-training-gy1/data/external/personal/g00833899/y50046448/spatial-vid-hq-oft}"
SPATIALVID_METADATA_URL="${SPATIALVID_OFT_ROOT}/data/train/SpatialVID_HQ_metadata.csv"
SPATIALVID_VIDEO_ROOT="${SPATIALVID_OFT_ROOT}/videos/SpatialVid/HQ/videos"
SPATIALVID_METADATA="${LOCAL_CACHE_ROOT}/metadata/SpatialVID_HQ_metadata_oft.csv"
SPATIALVID_SPLIT_DIR="${RUN_ROOT}/metadata/spatialvid_oft_seed${SPLIT_SEED}"
SPATIALVID_TRAIN_10K_CSV="${SPATIALVID_SPLIT_DIR}/train_10k.csv"
SPATIALVID_EVAL_CSV="${SPATIALVID_SPLIT_DIR}/eval.csv"
CACHE_PARTITION_ID="${CACHE_PARTITION_ID:-0}"
CACHE_NUM_PARTITIONS="${CACHE_NUM_PARTITIONS:-1}"
SAMPLES_PER_TAR="${SAMPLES_PER_TAR:-512}"
MAX_FAILURE_RATE="${MAX_FAILURE_RATE:-0.01}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
STORE_I0="${STORE_I0:-1}"
RESUME_CACHE="${RESUME_CACHE:-1}"
MASTER_PORT="${MASTER_PORT:-29675}"

[[ "${MODE}" == "train" || "${MODE}" == "eval" || "${MODE}" == "merge" ]] || {
  echo "MODE must be train, eval, or merge." >&2; exit 2;
}
[[ "${TEMPORAL_FACTOR}" -eq 2 && "${LATENT_DIM}" -eq 192 && "${SEQ_LEN}" -eq 9 ]] || {
  echo "Production cache contract is t2/c192/seq9." >&2; exit 2;
}
configure_modelarts_distributed
require_scale_cluster
require_output_url
ensure_spatialvid_subset_splits

if [[ "${MODE}" == "merge" ]]; then
  # merge_latent_cache works directly against durable OBS cache roots.
  [[ "${NODE_RANK}" -eq 0 ]] || exit 0
  LOG_DIR="${SCALE_ROOT}/cache_generation/${CACHE_VERSION}/merge"
  REMOTE_LOG_DIR="${SCALE_REMOTE_ROOT}/cache_generation/${CACHE_VERSION}/merge"
  start_output_sync "${LOG_DIR}" "${REMOTE_LOG_DIR}"
  trap 'stop_output_sync "${LOG_DIR}" "${REMOTE_LOG_DIR}"' EXIT
  exec > >(tee -a "${STAGE_LOG_FILE}") 2>&1
  "${PYTHON_BIN}" "${PROJECT}/merge_latent_cache.py" \
    --cache_dir "${R7_CACHE_OBS_ROOT}/train" \
    --expected_partitions "${CACHE_NUM_PARTITIONS}" \
    --max_failure_rate "${MAX_FAILURE_RATE}"
  "${PYTHON_BIN}" "${PROJECT}/merge_latent_cache.py" \
    --cache_dir "${R7_CACHE_OBS_ROOT}/eval" \
    --expected_partitions 1 --max_failure_rate "${MAX_FAILURE_RATE}"
  exit 0
fi

# The checkpoint is node-local after staging; every node performs this step.
if [[ "${NODE_RANK}" -ne 0 ]]; then rm -f "${R7_CKPT}"; fi
ensure_local_checkpoint "${R7_CKPT}" "${R7_CKPT_URL}" \
  "accepted R7 checkpoint" "${R7_CKPT_MIRROR_URL}"
if [[ -n "${DUAL_AE_CKPT_URL}" || -n "${DUAL_AE_CKPT_MIRROR_URL}" ]]; then
  if [[ "${NODE_RANK}" -ne 0 ]]; then rm -f "${DUAL_AE_CKPT}"; fi
  ensure_local_checkpoint "${DUAL_AE_CKPT}" "${DUAL_AE_CKPT_URL}" \
    "source dual-stream AE checkpoint" "${DUAL_AE_CKPT_MIRROR_URL}"
fi
require_file "${DUAL_AE_CKPT}" "source dual-stream AE checkpoint"

if [[ "${MODE}" == "train" ]]; then
  CSV="${SPATIALVID_TRAIN_10K_CSV}"
  CACHE_DIR="${R7_CACHE_OBS_ROOT}/train"
  PARTITION_ID="${CACHE_PARTITION_ID}"
  NUM_PARTITIONS="${CACHE_NUM_PARTITIONS}"
  CLIPS_PER_VIDEO="${CLIPS_PER_VIDEO:-1}"
else
  CSV="${SPATIALVID_EVAL_CSV}"
  CACHE_DIR="${R7_CACHE_OBS_ROOT}/eval"
  PARTITION_ID=0
  NUM_PARTITIONS=1
  CLIPS_PER_VIDEO=1
fi
LOG_DIR="${SCALE_ROOT}/cache_generation/${CACHE_VERSION}/${MODE}"
REMOTE_LOG_DIR="${SCALE_REMOTE_ROOT}/cache_generation/${CACHE_VERSION}/${MODE}"
start_output_sync "${LOG_DIR}" "${REMOTE_LOG_DIR}"
trap 'stop_output_sync "${LOG_DIR}" "${REMOTE_LOG_DIR}"' EXIT

EXTRA_ARGS=()
[[ "${RESUME_CACHE}" == 1 ]] && EXTRA_ARGS+=(--resume_cache)
[[ "${STORE_I0}" == 1 ]] && EXTRA_ARGS+=(--store_i0_rgb)
run_torchrun "${PROJECT}/cache_causal_video_latents.py" \
  --csv "${CSV}" --video_root "${SPATIALVID_VIDEO_ROOT}" \
  --encoder_ckpt "${STREAMVGGT_CKPT}" --r7_ckpt "${R7_CKPT}" \
  --dual_ae_ckpt "${DUAL_AE_CKPT}" \
  --output_dir "${CACHE_DIR}" --split "${MODE}" \
  --index_shard_id "${PARTITION_ID}" --index_num_shards "${NUM_PARTITIONS}" \
  --partition_id "${PARTITION_ID}" --num_partitions "${NUM_PARTITIONS}" \
  --samples_per_tar "${SAMPLES_PER_TAR}" --clips_per_video "${CLIPS_PER_VIDEO}" \
  --seq_len "${SEQ_LEN}" --target_size 518 \
  --batch_size "${BATCH_SIZE}" --num_workers "${NUM_WORKERS}" --dtype fp16 \
  "${EXTRA_ARGS[@]}"

if [[ "${NODE_RANK}" -eq 0 ]]; then
  echo "R7 ${MODE} cache is durable at ${CACHE_DIR}"
fi
