#!/bin/bash
# Phase 1: Train Generative Tokenizer A + Compact Decoder G
# Creates the compact generative latent space z_g
set -e

PROJECT=/home/yexiaoyu/work/VggAE-Diffusion
DATASET=/public2/LiZhen/yexiaoyu/dataset/spatial-vid-hq-oft

mkdir -p ${PROJECT}/ckpts/autoencoder

NUM_GPUS=8
GPU_IDS=0,1,2,3,4,5,6,7
CUDA_VISIBLE_DEVICES=${GPU_IDS} torchrun --nproc_per_node=${NUM_GPUS} --master_port=29510 \
  ${PROJECT}/train_autoencoder.py \
    --csv ${DATASET}/data/train/SpatialVID_HQ_metadata.csv \
    --video_root ${DATASET}/videos/SpatialVID/videos \
    --encoder_ckpt /home/yexiaoyu/work/4DLangVGGT/ckpt/streamvggt/checkpoints.pth \
    --output_dir ${PROJECT}/ckpts/autoencoder/exp-1 \
    --latent_dim 512 --latent_grid 18 \
    --levels 4 11 17 23 \
    --decoder_base_dim 256 \
    --latent_noise_std 0.1 --latent_noise_warmup 1000 \
    --lambda_l1 1.0 --lambda_grad 0.1 --lambda_temporal 0.2 --lambda_latent_reg 0.01 \
    --batch_size 3 --accum_steps 4 \
    --epochs 50 --lr 2e-4 --wd 1e-2 \
    --warmup_steps 500 --ema_decay 0.999 \
    --seq_len 8 --target_size 518 \
    --num_workers 4 --dtype bf16 \
    --eval_every 5 --save_every 5
