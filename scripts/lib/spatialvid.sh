#!/bin/bash

require_file() {
  local path=$1
  local label=$2
  if [[ ! -f "${path}" ]]; then
    echo "Missing ${label}: ${path}" >&2
    echo "Edit scripts/spatialvid_config.sh before launching." >&2
    exit 1
  fi
}

require_dir() {
  local path=$1
  local label=$2
  if [[ ! -d "${path}" ]]; then
    echo "Missing ${label}: ${path}" >&2
    echo "Edit scripts/spatialvid_config.sh before launching." >&2
    exit 1
  fi
}

validate_spatialvid_config() {
  require_file "${SPATIALVID_METADATA}" "SpatialVID metadata CSV"
  require_dir "${SPATIALVID_VIDEO_ROOT}" "SpatialVID video root"
  validate_model_config
  mkdir -p "${RUN_ROOT}" "${SPATIALVID_SPLIT_DIR}"
}

validate_model_config() {
  require_file "${STREAMVGGT_CKPT}" "StreamVGGT checkpoint"
  mkdir -p "${RUN_ROOT}"
}

ensure_spatialvid_splits() {
  validate_spatialvid_config
  "${PYTHON_BIN}" "${PROJECT}/prepare_spatialvid_splits.py" \
    --csv "${SPATIALVID_METADATA}" \
    --video_root "${SPATIALVID_VIDEO_ROOT}" \
    --output_dir "${SPATIALVID_SPLIT_DIR}" \
    --train_count "${TRAIN_10K_VIDEOS}" \
    --eval_count "${EVAL_VIDEOS}" \
    --overfit_count "${OVERFIT_VIDEOS}" \
    --min_frames "${MIN_VIDEO_FRAMES}" \
    --seed "${SPLIT_SEED}"
}

ensure_spatialvid_scale_splits() {
  validate_spatialvid_config
  "${PYTHON_BIN}" "${PROJECT}/prepare_spatialvid_splits.py" \
    --csv "${SPATIALVID_METADATA}" \
    --video_root "${SPATIALVID_VIDEO_ROOT}" \
    --output_dir "${SPATIALVID_SPLIT_DIR}" \
    --train_count "${TRAIN_10K_VIDEOS}" \
    --eval_count "${EVAL_VIDEOS}" \
    --overfit_count "${OVERFIT_VIDEOS}" \
    --min_frames "${MIN_VIDEO_FRAMES}" \
    --seed "${SPLIT_SEED}" \
    --write_full_train
}
