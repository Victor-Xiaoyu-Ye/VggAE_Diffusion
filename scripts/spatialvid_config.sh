#!/bin/bash

CONFIG_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT=$(cd "${CONFIG_DIR}/.." && pwd)

# SpatialVID-HQ stays on OBS. Video files are staged on demand into the
# bounded node-local cache below; the full dataset is never copied locally.
SPATIALVID_OBS_ROOT="obs://yw-pixelgeek-training-data-gy1/01.USERS/z00546255/data/yexiaoyu/dataset/SpatialVID-HQ"
SPATIALVID_METADATA_URL="${SPATIALVID_OBS_ROOT}/data/train/SpatialVID_HQ_metadata.csv"
SPATIALVID_VIDEO_ROOT="${SPATIALVID_OBS_ROOT}/videos/SpatialVID/videos"
SPATIALVID_DEPTH_ROOT="${SPATIALVID_OBS_ROOT}/depths/SpatialVID/depths"

# The cluster provides about 3 TB under /cache. Keep all temporary downloads,
# latent shard staging, checkpoints, and logs below it.
LOCAL_CACHE_ROOT="${LOCAL_CACHE_ROOT:-/cache/vggae}"
RUN_ROOT="${LOCAL_CACHE_ROOT}/outputs"
SPATIALVID_METADATA="${LOCAL_CACHE_ROOT}/metadata/SpatialVID_HQ_metadata.csv"

# Model code and reference checkpoints are pulled with the repository. These
# can still be overridden by exporting the variable before running the script.
VEGGIE_REF_ROOT="${VEGGIE_REF_ROOT:-${PROJECT}/veggie_ref}"
STREAMVGGT_CKPT="${STREAMVGGT_CKPT:-${VEGGIE_REF_ROOT}/streamvggt/checkpoints.pth}"
GEOMETRY_AE_CKPT="${GEOMETRY_AE_CKPT:-${VEGGIE_REF_ROOT}/checkpoints/geometry_autoencoder.pt}"

# ModelArts normally injects OUTPUT_URL. All rank-0 outputs are periodically
# mirrored here. Set it manually only outside ModelArts.
OUTPUT_URL="${OUTPUT_URL:-}"
REMOTE_RUN_ROOT="${OUTPUT_URL%/}"

# The cluster runtime prepares PATH before invoking bash.
PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"

# Bounded on-demand caches per node. Values are soft limits in GiB.
export MOX_VIDEO_CACHE_DIR="${LOCAL_CACHE_ROOT}/cache/videos"
export MOX_DEPTH_CACHE_DIR="${LOCAL_CACHE_ROOT}/cache/depth"
export MOX_LATENT_CACHE_DIR="${LOCAL_CACHE_ROOT}/cache/latent_shards"
export MOX_METADATA_CACHE_DIR="${LOCAL_CACHE_ROOT}/cache/metadata"
export MOX_CACHE_WRITER_DIR="${LOCAL_CACHE_ROOT}/cache/latent_writer"
export MOX_VIDEO_CACHE_GB="${MOX_VIDEO_CACHE_GB:-1400}"
export MOX_LATENT_CACHE_GB="${MOX_LATENT_CACHE_GB:-1400}"
export MOX_DOWNLOAD_RETRIES=4
export OUTPUT_SYNC_SECONDS=300

# Automatic deterministic split sizes.
SPLIT_SEED=42
TRAIN_10K_VIDEOS=10000
EVAL_VIDEOS=64
OVERFIT_VIDEOS=1
MIN_VIDEO_FRAMES=8

# Local checkpoints consumed by downstream stages.
I0_DECODER_CKPT="${RUN_ROOT}/10k/i0_decoder/checkpoint_final.pt"
OVERFIT_I0_DECODER_CKPT="${RUN_ROOT}/validation/i0_decoder_overfit/checkpoint_final.pt"
OVERFIT_DIFFUSION_CKPT="${RUN_ROOT}/validation/compact_diffusion_overfit/checkpoint_final.pt"
DIFFUSION_CKPT="${RUN_ROOT}/10k/compact_diffusion/checkpoint_final.pt"
SCALE_GEOMETRY_AE_CKPT="${RUN_ROOT}/scale/geometry_autoencoder/checkpoint_final.pt"
SCALE_I0_DECODER_CKPT="${RUN_ROOT}/scale/i0_decoder/checkpoint_final.pt"
SCALE_DIFFUSION_CKPT="${RUN_ROOT}/scale/compact_dit/checkpoint_final.pt"

# ============================================================================
# DERIVED PATHS - normally do not edit
# ============================================================================

SPATIALVID_SPLIT_DIR="${RUN_ROOT}/metadata/spatialvid_seed${SPLIT_SEED}"
SPATIALVID_TRAIN_10K_CSV="${SPATIALVID_SPLIT_DIR}/train_10k.csv"
SPATIALVID_EVAL_CSV="${SPATIALVID_SPLIT_DIR}/eval.csv"
SPATIALVID_OVERFIT_CSV="${SPATIALVID_SPLIT_DIR}/overfit.csv"
SPATIALVID_FULL_TRAIN_CSV="${SPATIALVID_SPLIT_DIR}/train_full.csv"

SCALE_ROOT="${RUN_ROOT}/scale"
SCALE_CSV_SHARD_ROOT="${SCALE_ROOT}/csv_shards"

# Large latent caches live under OUTPUT_URL and are streamed through the local
# MOX_LATENT_CACHE_DIR.
SCALE_REMOTE_ROOT="${REMOTE_RUN_ROOT}/scale"
SCALE_TRAIN_CACHE_DIR="${SCALE_REMOTE_ROOT}/latent_cache/train"
SCALE_EVAL_CACHE_DIR="${SCALE_REMOTE_ROOT}/latent_cache/eval"
