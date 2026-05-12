#!/bin/bash
set -e

PROJECT=/home/yexiaoyu/work/VggAE-Diffusion
DATASET=/public2/LiZhen/yexiaoyu/dataset/spatial-vid-hq-oft

mkdir -p ${PROJECT}/ckpts

cd ${PROJECT}
CUDA_VISIBLE_DEVICES=0 python compute_token_stats.py \
    --csv_path ${DATASET}/data/train/SpatialVID_HQ_metadata.csv \
    --video_root ${DATASET}/videos/SpatialVID/videos \
    --streamvggt_ckpt /home/yexiaoyu/work/4DLangVGGT/ckpt/streamvggt/checkpoints.pth \
    --out_path ${PROJECT}/ckpts/token_stats.pt \
    --seq_len 8 \
    --max_batches 2000
