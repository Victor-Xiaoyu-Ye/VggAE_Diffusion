#!/bin/bash
set -e

PROJECT=/home/yexiaoyu/work/VggAE-Diffusion
DATASET=/public2/LiZhen/yexiaoyu/dataset/spatial-vid-hq-oft

mkdir -p ${PROJECT}/ckpts

NUM_GPUS=5
GPU_IDS=0,1,2,3,4
CUDA_VISIBLE_DEVICES=${GPU_IDS} torchrun --nproc_per_node=${NUM_GPUS} --master_port=29502 \
  ${PROJECT}/train_decoder.py \
    --data_csv ${DATASET}/data/train/SpatialVID_HQ_metadata.csv \
    --video_root ${DATASET}/videos/SpatialVID/videos \
    --encoder_ckpt /home/yexiaoyu/work/4DLangVGGT/ckpt/streamvggt/checkpoints.pth \
    --token_stats ${PROJECT}/ckpts/token_stats.pt \
    --output_dir ${PROJECT}/ckpts/decoder_vit/exp-6-big \
    --seq_len 8 \
    --batch_size 4 --accum_steps 4 \
    --epochs 60 --lr 1e-4 --wd 1e-2 \
    --decoder_dim 1024 --vit_depth 20 --vit_heads 8 \
    --lpips_weight 1.0 --temporal_weight 0.05 --grad_weight 0.05 \
    --token_noise_std 0.02 --level_dropout 0.15 \
    --boundary_level 11 --boundary_only_prob 0.25 \
    --warmup_steps 500 \
    --ema_decay 0.999 --eval_every 5 --save_every 5 \
    --num_workers 4 --dtype bf16 \
    --output_depth --depth_root ${DATASET}/depths/SpatialVID/depths --depth_weight 0.1
