#!/bin/bash
# Autoencoder inference (big version, 4-card data paths)
set -e

PROJECT=/home/yexiaoyu/work/VggAE-Diffusion
DATASET=/home/yexiaoyu/data/spatial-vid-hq-oft

CKPT=${PROJECT}/ckpts/autoencoder/exp-1-big/checkpoint_final.pt
OUTPUT_DIR=${PROJECT}/outputs/autoencoder_inference/exp-1-big

mkdir -p ${OUTPUT_DIR}

CUDA_VISIBLE_DEVICES=0 python ${PROJECT}/inference_autoencoder.py \
    --checkpoint ${CKPT} \
    --encoder_ckpt /home/yexiaoyu/data/StreamVGGT/checkpoints.pth \
    --csv ${DATASET}/data/train/SpatialVID_HQ_metadata.csv \
    --video_root ${DATASET}/videos/SpatialVID/videos \
    --output_dir ${OUTPUT_DIR} \
    --num_videos 20 \
    --seq_len 8 --target_size 518 \
    --latent_dim 512 --latent_grid 18 \
    --levels 4 11 17 23 \
    --decoder_base_dim 384 --decoder_num_resblocks 2 \
    --compute_psnr
