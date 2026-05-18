#!/bin/bash
# Reconstruct real videos through encoder → tokens → decoder
# Usage: bash scripts/reconstruct.sh /path/to/video.mp4
set -e

PROJECT=/home/yexiaoyu/work/VggAE-Diffusion

VIDEO_PATH=${1:?Usage: bash scripts/reconstruct.sh <video_path>}
OUT_DIR=${PROJECT}/reconstruct_out/$(basename ${VIDEO_PATH} .mp4)

ASCEND_RT_VISIBLE_DEVICES=3 python ${PROJECT}/reconstruct.py \
    --video_path "${VIDEO_PATH}" \
    --encoder_ckpt /home/yexiaoyu/work/4DLangVGGT/ckpt/streamvggt/checkpoints.pth \
    --decoder_ckpt ${PROJECT}/ckpts/decoder_dpt/exp-5-dpt/decoder_final.pt \
    --token_stats ${PROJECT}/ckpts/token_stats.pt \
    --out_dir ${OUT_DIR} \
    --num_workers 4 \
    --sample_fps 24
