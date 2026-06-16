#!/bin/bash

# Local 10K experiment defaults. Source this after scripts/spatialvid_config.sh
# and before scripts/lib/spatialvid.sh.

LOCAL_SPATIALVID_ROOT="${LOCAL_SPATIALVID_ROOT:-/public2/LiZhen/yexiaoyu/dataset/spatial-vid-hq-oft}"
SPATIALVID_METADATA="${LOCAL_SPATIALVID_ROOT}/data/train/SpatialVID_HQ_metadata.csv"
SPATIALVID_METADATA_URL="${SPATIALVID_METADATA}"
SPATIALVID_VIDEO_ROOT="${LOCAL_SPATIALVID_ROOT}/videos/SpatialVID/videos"
SPATIALVID_DEPTH_ROOT="${LOCAL_SPATIALVID_ROOT}/depths/SpatialVID/depths"

RUN_ROOT="${LOCAL_RUN_ROOT:-${PROJECT}/outputs}"
REMOTE_RUN_ROOT="${RUN_ROOT}"
PERSISTENT_RUN_ROOT="${RUN_ROOT}"
MIRROR_RUN_ROOT="${RUN_ROOT}"

STREAMVGGT_CKPT="${LOCAL_STREAMVGGT_CKPT:-/home/yexiaoyu/work/4DLangVGGT/ckpt/streamvggt/checkpoints.pth}"
GEOMETRY_AE_CKPT="${LOCAL_GEOMETRY_AE_CKPT:-${RUN_ROOT}/10k/geometry_autoencoder/checkpoint_latest.pt}"
I0_DECODER_CKPT="${LOCAL_I0_DECODER_CKPT:-${RUN_ROOT}/10k/i0_decoder/checkpoint_latest.pt}"
OVERFIT_I0_DECODER_CKPT="${RUN_ROOT}/validation/i0_decoder_overfit/checkpoint_latest.pt"
OVERFIT_DIFFUSION_CKPT="${RUN_ROOT}/validation/compact_diffusion_overfit/checkpoint_latest.pt"
DIFFUSION_CKPT="${LOCAL_DIFFUSION_CKPT:-${RUN_ROOT}/10k/compact_diffusion/checkpoint_latest.pt}"

SPATIALVID_SPLIT_DIR="${RUN_ROOT}/metadata/spatialvid_seed${SPLIT_SEED}"
SPATIALVID_TRAIN_10K_CSV="${SPATIALVID_SPLIT_DIR}/train_10k.csv"
SPATIALVID_EVAL_CSV="${SPATIALVID_SPLIT_DIR}/eval.csv"
SPATIALVID_OVERFIT_CSV="${SPATIALVID_SPLIT_DIR}/overfit.csv"
SPATIALVID_FULL_TRAIN_CSV="${SPATIALVID_SPLIT_DIR}/train_full.csv"

CUDA_DEVICE_IDS="${CUDA_DEVICE_IDS:-0,1,2,3,4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-$(awk -F, '{print NF}' <<< "${CUDA_DEVICE_IDS}")}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"

configure_modelarts_distributed() {
  NUM_NPUS="${NUM_GPUS}"
  NNODES=1
  NODE_RANK=0
  MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  MASTER_PORT="${MASTER_PORT:-29500}"
  WORLD_SIZE="${NUM_GPUS}"
  export NUM_NPUS NNODES NODE_RANK MASTER_ADDR MASTER_PORT WORLD_SIZE
}

start_output_sync() {
  local local_dir=$1
  mkdir -p "${local_dir}/logs"
  export STAGE_LOG_FILE="${local_dir}/logs/train_local.log"
}

stop_output_sync() {
  true
}

run_torchrun() {
  local launcher=("${TORCHRUN_BIN}")
  if ! command -v "${TORCHRUN_BIN}" >/dev/null 2>&1; then
    launcher=("${PYTHON_BIN}" -m torch.distributed.run)
  fi
  local command=(
    "${launcher[@]}"
    "--nnodes=1"
    "--node_rank=0"
    "--nproc_per_node=${NUM_GPUS}"
    "--master_addr=${MASTER_ADDR:-127.0.0.1}"
    "--master_port=${MASTER_PORT:-29500}"
    "$@"
  )
  if [[ -n "${STAGE_LOG_FILE:-}" ]]; then
    mkdir -p "$(dirname "${STAGE_LOG_FILE}")"
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_IDS}" \
      OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
      "${command[@]}" 2>&1 | tee -a "${STAGE_LOG_FILE}"
    return "${PIPESTATUS[0]}"
  fi
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_IDS}" \
    OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
    "${command[@]}"
}
