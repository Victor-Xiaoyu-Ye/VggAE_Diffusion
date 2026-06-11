#!/bin/bash
set -e

PROJECT=/home/yexiaoyu/work/VggAE-Diffusion
DATASET=/public2/LiZhen/yexiaoyu/dataset/spatial-vid-hq-oft

cd ${PROJECT}
CUDA_VISIBLE_DEVICES=0 proxychains python ${PROJECT}/visualize_latent_space.py \
    --csv ${DATASET}/data/train/SpatialVID_HQ_metadata.csv \
    --video_root ${DATASET}/videos/SpatialVID/videos \
    --annotation_index ${PROJECT}/ckpts/annotation_index.json \
    --encoder_ckpt /home/yexiaoyu/work/4DLangVGGT/ckpt/streamvggt/checkpoints.pth \
    --token_stats ${PROJECT}/ckpts/token_stats.pt \
    --levels 11 \
    --decoder_ckpt ${PROJECT}/ckpts/decoder_gld/exp-0/decoder_epoch45.pt \
    --decoder_keep_levels 11 \
    --with_dino \
    --max_videos 128 \
    --seq_len 8 \
    --pool clip \
    --out_dir ${PROJECT}/analysis/latent_vis_level11
