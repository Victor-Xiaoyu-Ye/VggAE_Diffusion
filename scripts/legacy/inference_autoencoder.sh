#!/bin/bash
# Autoencoder inference: reconstruct videos and compute PSNR
set -e

PROJECT=/home/yexiaoyu/work/VggAE-Diffusion
DATASET=/public2/LiZhen/yexiaoyu/dataset/spatial-vid-hq-oft

CKPT=${PROJECT}/ckpts/autoencoder/exp-1/checkpoint_final.pt
OUTPUT_DIR=${PROJECT}/outputs/autoencoder_inference/exp-1

mkdir -p ${OUTPUT_DIR}

# Batch inference on dataset
CUDA_VISIBLE_DEVICES=0 python ${PROJECT}/inference_autoencoder.py \
    --checkpoint ${CKPT} \
    --encoder_ckpt /home/yexiaoyu/work/4DLangVGGT/ckpt/streamvggt/checkpoints.pth \
    --csv ${DATASET}/data/train/SpatialVID_HQ_metadata.csv \
    --video_root ${DATASET}/videos/SpatialVID/videos \
    --output_dir ${OUTPUT_DIR} \
    --num_videos 20 \
    --seq_len 8 --target_size 518 \
    --latent_dim 512 --latent_grid 18 \
    --levels 4 11 17 23 \
    --compute_psnr
