#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT=${PROJECT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
CSV_SHARD_DIR=${CSV_SHARD_DIR:?Set CSV_SHARD_DIR from stage 02}
VIDEO_ROOT=${VIDEO_ROOT:?Set VIDEO_ROOT to the SpatialVID video root}
ENCODER_CKPT=${ENCODER_CKPT:?Set ENCODER_CKPT to StreamVGGT weights}
AUTOENCODER_CKPT=${AUTOENCODER_CKPT:?Set AUTOENCODER_CKPT from stage 00}
CACHE_DIR=${CACHE_DIR:?Set CACHE_DIR to shared high-throughput storage}

INDEX_SHARD_ID=${INDEX_SHARD_ID:?Set INDEX_SHARD_ID from the array-job index}
INDEX_NUM_SHARDS=${INDEX_NUM_SHARDS:-256}
NUM_GPUS=${NUM_GPUS:-8}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29602}
SAMPLES_PER_TAR=${SAMPLES_PER_TAR:-512}
printf -v CSV_PART "part-%05d.csv" "${INDEX_SHARD_ID}"

torchrun --nnodes="${NNODES}" --node_rank="${NODE_RANK}" \
  --nproc_per_node="${NUM_GPUS}" --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${PROJECT}/cache_compact_latents.py" \
  --csv "${CSV_SHARD_DIR}/${CSV_PART}" \
  --video_root "${VIDEO_ROOT}" \
  --encoder_ckpt "${ENCODER_CKPT}" \
  --autoencoder_ckpt "${AUTOENCODER_CKPT}" \
  --output_dir "${CACHE_DIR}" \
  --partition_id "${INDEX_SHARD_ID}" \
  --num_partitions "${INDEX_NUM_SHARDS}" \
  --samples_per_tar "${SAMPLES_PER_TAR}" \
  --latent_dim 512 --latent_grid 18 \
  --levels 4 11 17 23 \
  --seq_len 8 --target_size 518 \
  --max_frame_span 32 \
  --batch_size 1 --num_workers 8
