#!/bin/bash
# Train I_0-conditioned autoencoder (cross-frame appearance + geometry)
# 7 GPUs DDP, GPU 0 reserved for diffusion sampling
set -e

PROJECT=/home/yexiaoyu/work/VggAE-Diffusion
DATASET=/public2/LiZhen/yexiaoyu/dataset/spatial-vid-hq-oft

# Frozen Tokenizer A from exp-1-big
AE_CKPT=${PROJECT}/ckpts/autoencoder/exp-1-big/checkpoint_final.pt

mkdir -p ${PROJECT}/ckpts/i0_autoencoder

NUM_GPUS=5
GPU_IDS=3,4,5,6,7
CUDA_VISIBLE_DEVICES=${GPU_IDS} torchrun --nproc_per_node=${NUM_GPUS} --master_port=29530 \
    ${PROJECT}/train_i0_autoencoder.py \
    --csv ${DATASET}/data/train/SpatialVID_HQ_metadata.csv \
    --video_root ${DATASET}/videos/SpatialVID/videos \
    --encoder_ckpt /home/yexiaoyu/work/4DLangVGGT/ckpt/streamvggt/checkpoints.pth \
    --autoencoder_ckpt ${AE_CKPT} \
    --output_dir ${PROJECT}/ckpts/i0_autoencoder/exp-1 \
    --latent_dim 512 --latent_grid 18 \
    --decoder_base_dim 384 --decoder_num_resblocks 2 \
    --epochs 50 --batch_size 1 --accum_steps 4 --lr 1e-4 --wd 1e-2 \
    --warmup_steps 500 --max_grad_norm 1.0 \
    --dtype bf16 --seq_len 8 --target_size 518 --num_workers 2 \
    --cross_frame_gap 4 \
    --save_every 5 --eval_every 5
