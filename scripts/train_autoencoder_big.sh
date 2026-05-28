#!/bin/bash
# Phase 1 (Big): Train Generative Tokenizer A + High-Capacity Compact Decoder G
# Uses wider decoder (base_dim=384), pixel-shuffle upsampling, LPIPS loss.
# Card config: 4 GPUs, matching train_decoder_dpt_mean_ds.sh
set -e

PROJECT=/home/yexiaoyu/work/VggAE-Diffusion
DATASET=/home/yexiaoyu/data/spatial-vid-hq-oft

mkdir -p ${PROJECT}/ckpts/autoencoder

NUM_GPUS=4
GPU_IDS=0,1,2,3
CUDA_VISIBLE_DEVICES=${GPU_IDS} torchrun --nproc_per_node=${NUM_GPUS} --master_port=29510 \
  ${PROJECT}/train_autoencoder.py \
    --csv ${DATASET}/data/train/SpatialVID_HQ_metadata.csv \
    --video_root ${DATASET}/videos/SpatialVID/videos \
    --encoder_ckpt /home/yexiaoyu/data/StreamVGGT/checkpoints.pth \
    --output_dir ${PROJECT}/ckpts/autoencoder/exp-1-big \
    --latent_dim 512 --latent_grid 18 \
    --levels 4 11 17 23 \
    --decoder_base_dim 384 --decoder_num_resblocks 2 \
    --latent_noise_std 0.05 --latent_noise_warmup 1000 \
    --lambda_l1 1.0 --lambda_lpips 1.0 --lambda_grad 0.05 --lambda_temporal 0.05 --lambda_latent_reg 0.01 \
    --batch_size 10 --accum_steps 4 \
    --epochs 120 --lr 1e-4 --wd 1e-2 \
    --warmup_steps 500 --ema_decay 0.999 \
    --seq_len 8 --target_size 518 \
    --num_workers 10 --dtype bf16 \
    --eval_every 5 --save_every 5
