#!/bin/bash

# ============================================================================
# EDIT THIS BLOCK ON EACH CLUSTER
# ============================================================================

# SpatialVID layout. Usually only SPATIALVID_ROOT needs to change.
SPATIALVID_ROOT="/public2/LiZhen/yexiaoyu/dataset/spatial-vid-hq-oft"
SPATIALVID_METADATA="${SPATIALVID_ROOT}/data/train/SpatialVID_HQ_metadata.csv"
SPATIALVID_VIDEO_ROOT="${SPATIALVID_ROOT}/videos/SpatialVID/videos"
SPATIALVID_DEPTH_ROOT="${SPATIALVID_ROOT}/depths/SpatialVID/depths"

# Frozen StreamVGGT checkpoint.
STREAMVGGT_CKPT="/home/yexiaoyu/work/4DLangVGGT/ckpt/streamvggt/checkpoints.pth"

# All generated splits, checkpoints, previews, and reports are kept here.
RUN_ROOT="/home/yexiaoyu/work/VggAE-Diffusion/outputs/spatialvid_runs"

# Use the project environment explicitly so scripts do not depend on the
# interactive shell's currently activated Conda environment.
PYTHON_BIN="/home/yexiaoyu/miniconda3/envs/rae/bin/python"
TORCHRUN_BIN="/home/yexiaoyu/miniconda3/envs/rae/bin/torchrun"

# Automatic deterministic split sizes.
SPLIT_SEED=42
TRAIN_10K_VIDEOS=10000
EVAL_VIDEOS=64
OVERFIT_VIDEOS=1
MIN_VIDEO_FRAMES=8

# Checkpoints consumed by downstream stages. Change these when selecting a
# different experiment checkpoint.
GEOMETRY_AE_CKPT="/home/yexiaoyu/work/VggAE-Diffusion/ckpts/autoencoder/exp-1-big/checkpoint_final.pt"
I0_DECODER_CKPT="${RUN_ROOT}/10k/i0_decoder/checkpoint_final.pt"
OVERFIT_I0_DECODER_CKPT="${RUN_ROOT}/validation/i0_decoder_overfit/checkpoint_final.pt"
OVERFIT_DIFFUSION_CKPT="${RUN_ROOT}/validation/compact_diffusion_overfit/checkpoint_final.pt"
DIFFUSION_CKPT="${RUN_ROOT}/10k/compact_diffusion/checkpoint_final.pt"

# Large-scale stage checkpoints.
SCALE_GEOMETRY_AE_CKPT="${RUN_ROOT}/scale/geometry_autoencoder/checkpoint_final.pt"
SCALE_I0_DECODER_CKPT="${RUN_ROOT}/scale/i0_decoder/checkpoint_final.pt"
SCALE_DIFFUSION_CKPT="${RUN_ROOT}/scale/compact_dit/checkpoint_final.pt"

# Reference image or video used by sampling scripts.
I0_PATH="/path/to/reference.png"

# ============================================================================
# DERIVED PATHS - normally do not edit
# ============================================================================

CONFIG_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT=$(cd "${CONFIG_DIR}/.." && pwd)
SPATIALVID_SPLIT_DIR="${RUN_ROOT}/metadata/spatialvid_seed${SPLIT_SEED}"
SPATIALVID_TRAIN_10K_CSV="${SPATIALVID_SPLIT_DIR}/train_10k.csv"
SPATIALVID_EVAL_CSV="${SPATIALVID_SPLIT_DIR}/eval.csv"
SPATIALVID_OVERFIT_CSV="${SPATIALVID_SPLIT_DIR}/overfit.csv"
SPATIALVID_FULL_TRAIN_CSV="${SPATIALVID_SPLIT_DIR}/train_full.csv"
SCALE_ROOT="${RUN_ROOT}/scale"
SCALE_CSV_SHARD_ROOT="${SCALE_ROOT}/csv_shards"
SCALE_TRAIN_CACHE_DIR="${SCALE_ROOT}/latent_cache/train"
SCALE_EVAL_CACHE_DIR="${SCALE_ROOT}/latent_cache/eval"
