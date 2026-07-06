#!/bin/bash
# H200 (4x H200) experiment environment for the reconstruction-first phase.
#
# This machine is the local 4-GPU H200 box used for the E1/E2/E3 feature-space
# probes and the subsequent reconstruction rewrite. It is NOT the ModelArts
# Ascend scale cluster (scripts/scale/) nor the local A100 10k box
# (scripts/10k/). Keep paths and launch topology separate from both.
#
# Source this before the probe / trainer launchers:
#   source scripts/h200/h200_env.sh
#
# Override any variable before sourcing to customize per run.

# Repo root (auto-detect from this file's location).
export VGGAE_PROJECT="${VGGAE_PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

# Dataset (same layout as the legacy 4-card H200 train_decoder_dpt_gan.sh).
export VGGAE_DATASET_ROOT="${VGGAE_DATASET_ROOT:-/home/yexiaoyu/data/spatial-vid-hq-oft}"
export VGGAE_METADATA_CSV="${VGGAE_METADATA_CSV:-${VGGAE_DATASET_ROOT}/data/train/SpatialVID_HQ_metadata.csv}"
export VGGAE_VIDEO_ROOT="${VGGAE_VIDEO_ROOT:-${VGGAE_DATASET_ROOT}/videos/SpatialVID/videos}"
export VGGAE_DEPTH_ROOT="${VGGAE_DEPTH_ROOT:-${VGGAE_DATASET_ROOT}/depths/SpatialVID/depths}"

# Frozen StreamVGGT encoder checkpoint.
export VGGAE_ENCODER_CKPT="${VGGAE_ENCODER_CKPT:-/home/yexiaoyu/data/StreamVGGT/checkpoints.pth}"

# Output root for H200 experiments.
export VGGAE_H200_RUN_ROOT="${VGGAE_H200_RUN_ROOT:-${VGGAE_PROJECT}/outputs/h200}"

# GPU topology.
export VGGAE_NUM_GPUS="${VGGAE_NUM_GPUS:-4}"
export VGGAE_GPU_IDS="${VGGAE_GPU_IDS:-0,1,2,3}"
export VGGAE_MASTER_PORT="${VGGAE_MASTER_PORT:-29540}"

# Python binary (prefer the rae env if present, else python).
if [[ -x /home/yexiaoyu/miniconda3/envs/rae/bin/python ]]; then
  export VGGAE_PYTHON_BIN="${VGGAE_PYTHON_BIN:-/home/yexiaoyu/miniconda3/envs/rae/bin/python}"
else
  export VGGAE_PYTHON_BIN="${VGGAE_PYTHON_BIN:-python}"
fi
export VGGAE_TORCHRUN_BIN="${VGGAE_TORCHRUN_BIN:-torchrun}"

# SpatialVID deterministic 10K split (created by prepare_spatialvid_splits.py).
# Reuse the same seed as scripts/10k so eval sets align across machines.
export VGGAE_SPLIT_SEED="${VGGAE_SPLIT_SEED:-42}"
export VGGAE_SPLIT_DIR="${VGGAE_SPLIT_DIR:-${VGGAE_H200_RUN_ROOT}/metadata/spatialvid_seed${VGGAE_SPLIT_SEED}}"
export VGGAE_TRAIN_10K_CSV="${VGGAE_TRAIN_10K_CSV:-${VGGAE_SPLIT_DIR}/train_10k.csv}"
export VGGAE_EVAL_CSV="${VGGAE_EVAL_CSV:-${VGGAE_SPLIT_DIR}/eval.csv}"
export VGGAE_OVERFIT_CSV="${VGGAE_OVERFIT_CSV:-${VGGAE_SPLIT_DIR}/overfit.csv}"

mkdir -p "${VGGAE_H200_RUN_ROOT}" "${VGGAE_SPLIT_DIR}"

# Ensure the 10k/eval/overfit splits exist on this machine.
# Args verified against prepare_spatialvid_splits.py:
#   required: --csv, --video_root, --output_dir
#   optional: --train_count (def 10000), --eval_count (def 64),
#             --overfit_count (def 1), --min_frames (def 8),
#             --seed (def 42), --candidate_multiplier (def 2),
#             --skip_file_check (flag), --write_full_train (flag), --force (flag)
ensure_h200_splits() {
  if [[ -s "${VGGAE_TRAIN_10K_CSV}" && -s "${VGGAE_EVAL_CSV}" ]]; then
    return 0
  fi
  echo "[h200_env] Building SpatialVID 10k/eval/overfit splits under ${VGGAE_SPLIT_DIR}" >&2
  "${VGGAE_PYTHON_BIN}" "${VGGAE_PROJECT}/prepare_spatialvid_splits.py" \
    --csv "${VGGAE_METADATA_CSV}" \
    --video_root "${VGGAE_VIDEO_ROOT}" \
    --output_dir "${VGGAE_SPLIT_DIR}" \
    --seed "${VGGAE_SPLIT_SEED}" \
    ${VGGAE_SPLIT_EXTRA_ARGS:-}
}

# Standard torchrun launcher for the 4-card H200 topology.
h200_torchrun() {
  local launcher=("${VGGAE_TORCHRUN_BIN}")
  if ! command -v "${VGGAE_TORCHRUN_BIN}" >/dev/null 2>&1; then
    launcher=("${VGGAE_PYTHON_BIN}" -m torch.distributed.run)
  fi
  CUDA_VISIBLE_DEVICES="${VGGAE_GPU_IDS}" \
    OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
    "${launcher[@]}" \
    --nnodes=1 --node_rank=0 \
    --nproc_per_node="${VGGAE_NUM_GPUS}" \
    --master_addr=127.0.0.1 \
    --master_port="${VGGAE_MASTER_PORT}" \
    "$@"
}
