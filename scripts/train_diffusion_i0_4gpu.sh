#!/bin/bash
# Phase 2 (4-GPU): Train flow matching on z_geo with rescale + I_0 decoder aux
set -e

PROJECT=/home/yexiaoyu/work/VggAE-Diffusion
DATASET=/home/yexiaoyu/data/spatial-vid-hq-oft

# Autoencoder checkpoints
AE_CKPT=${PROJECT}/ckpts/autoencoder/exp-1-big/checkpoint_final.pt
I0_CKPT=${PROJECT}/ckpts/i0_autoencoder/exp-1/checkpoint_final.pt

mkdir -p ${PROJECT}/ckpts/diffusion_i0

NUM_GPUS=4
GPU_IDS=0,1,2,3
CUDA_VISIBLE_DEVICES=${GPU_IDS} torchrun --nproc_per_node=${NUM_GPUS} --master_port=29540 \
    ${PROJECT}/train_compact_diffusion.py \
    --csv ${DATASET}/data/train/SpatialVID_HQ_metadata.csv \
    --video_root ${DATASET}/videos/SpatialVID/videos \
    --encoder_ckpt /home/yexiaoyu/data/StreamVGGT/checkpoints.pth \
    --autoencoder_ckpt ${AE_CKPT} \
    --i0_decoder_ckpt ${I0_CKPT} \
    --output_dir ${PROJECT}/ckpts/diffusion_i0/exp-1 \
    --latent_dim 512 --latent_grid 18 \
    --model_dim 768 --spatial_depth 8 --temporal_depth 4 --num_heads 12 \
    --levels 4 11 17 23 \
    --decoder_base_dim 384 --decoder_num_resblocks 2 \
    --decoder_pixel_shuffle --decoder_temporal_blocks 2 \
    --rescale \
    --batch_size 2 --accum_steps 4 \
    --epochs 50 --lr 1e-4 --wd 1e-2 \
    --warmup_steps 1000 --ema_decay 0.9999 \
    --seq_len 8 --target_size 518 \
    --num_workers 4 --dtype bf16 \
    --eval_every 5 --save_every 5
