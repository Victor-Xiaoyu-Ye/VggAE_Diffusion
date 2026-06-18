#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"

# Two optimizer steps on the held-out cache. This validates OBS tar streaming,
# normalization, NPU forward/backward, EMA, checkpointing, and RGB preview.
OUTPUT_DIR="${SCALE_ROOT}/smoke_compact_dit"
REMOTE_OUTPUT_DIR="${SCALE_REMOTE_ROOT}/smoke_compact_dit"
I0_CKPT="${SCALE_I0_DECODER_CKPT}"
MASTER_PORT=29605

configure_modelarts_distributed
require_scale_cluster
require_output_url

ensure_local_checkpoint \
  "${I0_CKPT}" "${SCALE_I0_DECODER_CKPT_URL}" \
  "scale I0 decoder checkpoint" \
  "${SCALE_I0_DECODER_MIRROR_CKPT_URL}"

# Merge only the held-out partition. Node 0 writes the shared OBS metadata;
# the other nodes wait before entering torchrun.
if [[ "${NODE_RANK}" -eq 0 ]]; then
  "${PYTHON_BIN}" "${PROJECT}/merge_latent_cache.py" \
    --cache_dir "${SCALE_EVAL_CACHE_DIR}" \
    --expected_partitions 1 \
    --max_failure_rate 0.01
else
  "${PYTHON_BIN}" "${PROJECT}/scripts/wait_for_path.py" \
    "${SCALE_EVAL_CACHE_DIR}/manifest.txt"
  "${PYTHON_BIN}" "${PROJECT}/scripts/wait_for_path.py" \
    "${SCALE_EVAL_CACHE_DIR}/stats.pt"
fi

start_output_sync "${OUTPUT_DIR}" "${REMOTE_OUTPUT_DIR}"
trap 'stop_output_sync "${OUTPUT_DIR}" "${REMOTE_OUTPUT_DIR}"' EXIT

run_torchrun "${PROJECT}/train_cached_compact_diffusion.py" \
  --manifest "${SCALE_EVAL_CACHE_DIR}/manifest.txt" \
  --stats "${SCALE_EVAL_CACHE_DIR}/stats.pt" \
  --eval_manifest "${SCALE_EVAL_CACHE_DIR}/manifest.txt" \
  --eval_stats "${SCALE_EVAL_CACHE_DIR}/stats.pt" \
  --i0_decoder_ckpt "${I0_CKPT}" \
  --output_dir "${OUTPUT_DIR}" \
  --latent_dim "${SCALE_LATENT_DIM}" \
  --latent_grid "${SCALE_LATENT_GRID}" --seq_len 7 \
  --model_dim 640 --spatial_depth 8 --temporal_depth 4 --num_heads 10 \
  --time_scale 1000 \
  --batch_size 1 --accum_steps 1 --max_steps 2 \
  --lr 1e-4 --wd 1e-2 --warmup_steps 1 \
  --ema_decay 0.9999 --max_grad_norm 1.0 \
  --num_workers 0 --shuffle_buffer 1 \
  --log_every 1 --save_every 1 --eval_every 1 \
  --sample_steps 2 --dtype fp16
