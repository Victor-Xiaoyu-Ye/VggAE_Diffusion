#!/bin/bash

# Local 10K experiment defaults. Source this after scripts/spatialvid_config.sh
# and before scripts/lib/spatialvid.sh.

LOCAL_SPATIALVID_ROOT="${LOCAL_SPATIALVID_ROOT:-/public2/LiZhen/yexiaoyu/dataset/spatial-vid-hq-oft}"
SPATIALVID_METADATA="${LOCAL_SPATIALVID_ROOT}/data/train/SpatialVID_HQ_metadata.csv"
SPATIALVID_METADATA_URL="${SPATIALVID_METADATA}"
SPATIALVID_VIDEO_ROOT="${LOCAL_SPATIALVID_ROOT}/videos/SpatialVID/videos"
SPATIALVID_DEPTH_ROOT="${LOCAL_SPATIALVID_ROOT}/depths/SpatialVID/depths"

RUN_ROOT="${LOCAL_RUN_ROOT:-${PROJECT}/outputs/spatialvid_runs}"
REMOTE_RUN_ROOT="${RUN_ROOT}"
PERSISTENT_RUN_ROOT="${RUN_ROOT}"
MIRROR_RUN_ROOT="${RUN_ROOT}"

STREAMVGGT_CKPT="${LOCAL_STREAMVGGT_CKPT:-/home/yexiaoyu/work/4DLangVGGT/ckpt/streamvggt/checkpoints.pth}"
if [[ -n "${LOCAL_GEOMETRY_AE_CKPT:-}" ]]; then
  GEOMETRY_AE_CKPT="${LOCAL_GEOMETRY_AE_CKPT}"
elif [[ -s "${RUN_ROOT}/10k/geometry_autoencoder/checkpoint_latest.pt" ]]; then
  GEOMETRY_AE_CKPT="${RUN_ROOT}/10k/geometry_autoencoder/checkpoint_latest.pt"
elif [[ -s "${RUN_ROOT}/10k/geometry_autoencoder/checkpoint_final.pt" ]]; then
  GEOMETRY_AE_CKPT="${RUN_ROOT}/10k/geometry_autoencoder/checkpoint_final.pt"
else
  GEOMETRY_AE_CKPT="${PROJECT}/ckpts/autoencoder/exp-1-big/checkpoint_final.pt"
fi
if [[ -n "${LOCAL_I0_DECODER_CKPT:-}" ]]; then
  I0_DECODER_CKPT="${LOCAL_I0_DECODER_CKPT}"
elif [[ -s "${RUN_ROOT}/10k/i0_decoder/checkpoint_latest.pt" ]]; then
  I0_DECODER_CKPT="${RUN_ROOT}/10k/i0_decoder/checkpoint_latest.pt"
else
  I0_DECODER_CKPT="${RUN_ROOT}/10k/i0_decoder/checkpoint_final.pt"
fi
if [[ -s "${RUN_ROOT}/validation/i0_decoder_overfit/checkpoint_latest.pt" ]]; then
  OVERFIT_I0_DECODER_CKPT="${RUN_ROOT}/validation/i0_decoder_overfit/checkpoint_latest.pt"
else
  OVERFIT_I0_DECODER_CKPT="${RUN_ROOT}/validation/i0_decoder_overfit/checkpoint_final.pt"
fi
OVERFIT_DIFFUSION_CKPT="${RUN_ROOT}/validation/compact_diffusion_overfit/checkpoint_latest.pt"
DIFFUSION_CKPT="${LOCAL_DIFFUSION_CKPT:-${RUN_ROOT}/10k/compact_diffusion/checkpoint_latest.pt}"

SPATIALVID_SPLIT_DIR="${RUN_ROOT}/metadata/spatialvid_seed${SPLIT_SEED}"
SPATIALVID_TRAIN_10K_CSV="${SPATIALVID_SPLIT_DIR}/train_10k.csv"
SPATIALVID_EVAL_CSV="${SPATIALVID_SPLIT_DIR}/eval.csv"
SPATIALVID_OVERFIT_CSV="${SPATIALVID_SPLIT_DIR}/overfit.csv"
SPATIALVID_FULL_TRAIN_CSV="${SPATIALVID_SPLIT_DIR}/train_full.csv"

CUDA_DEVICE_IDS="${CUDA_DEVICE_IDS:-0,1,2,3,4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-$(awk -F, '{print NF}' <<< "${CUDA_DEVICE_IDS}")}"
if [[ ( -z "${PYTHON_BIN:-}" || "${PYTHON_BIN}" == "python" ) && -x "/home/yexiaoyu/miniconda3/envs/rae/bin/python" ]]; then
  PYTHON_BIN="/home/yexiaoyu/miniconda3/envs/rae/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi
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

count_csv_rows() {
  local csv_path=$1
  if [[ ! -s "${csv_path}" ]]; then
    echo 0
    return
  fi
  local total_lines
  total_lines=$(wc -l < "${csv_path}" | tr -d ' ')
  if (( total_lines <= 0 )); then
    echo 0
  else
    echo $((total_lines - 1))
  fi
}

cap_warmup_steps() {
  local csv_path=$1
  local batch_size=$2
  local accum_steps=$3
  local epochs=$4
  local requested_warmup=$5
  local rows
  rows=$(count_csv_rows "${csv_path}")
  if (( rows <= 0 || batch_size <= 0 || accum_steps <= 0 || epochs <= 0 )); then
    echo "${requested_warmup}"
    return
  fi

  # Match DistributedSampler + drop_last=True conservatively: each rank only
  # sees roughly rows / NUM_GPUS examples, then DataLoader drops incomplete
  # per-rank batches.
  local samples_per_rank=$((rows / NUM_GPUS))
  local batches_per_epoch=$((samples_per_rank / batch_size))
  if (( batches_per_epoch < 1 )); then
    batches_per_epoch=1
  fi
  local steps_per_epoch=$(((batches_per_epoch + accum_steps - 1) / accum_steps))
  if (( steps_per_epoch < 1 )); then
    steps_per_epoch=1
  fi
  local total_steps=$((steps_per_epoch * epochs))
  local max_warmup=$((total_steps > 1 ? total_steps - 1 : 0))
  if (( requested_warmup > max_warmup )); then
    echo "${max_warmup}"
  else
    echo "${requested_warmup}"
  fi
}

maybe_cap_warmup_steps() {
  local csv_path=$1
  local batch_size=$2
  local accum_steps=$3
  local epochs=$4
  local requested_warmup=$5
  local capped
  capped=$(cap_warmup_steps "${csv_path}" "${batch_size}" "${accum_steps}" "${epochs}" "${requested_warmup}")
  if [[ "${capped}" != "${requested_warmup}" ]]; then
    echo "[local_cuda] Capping warmup_steps ${requested_warmup} -> ${capped} for $(count_csv_rows "${csv_path}") rows, ${NUM_GPUS} GPUs, batch=${batch_size}, accum=${accum_steps}, epochs=${epochs}" >&2
  fi
  echo "${capped}"
}
