#!/bin/bash
# Experiment B: DPTHead baseline + GAN + multi-scale + 200 epochs
set -e

PROJECT=/home/yexiaoyu/work/VggAE-Diffusion
DATASET=/home/yexiaoyu/data/spatial-vid-hq-oft

mkdir -p ${PROJECT}/ckpts

NUM_GPUS=4
GPU_IDS=0,1,2,3
ASCEND_RT_VISIBLE_DEVICES=${GPU_IDS} torchrun --nproc_per_node=${NUM_GPUS} --master_port=29510 \
  ${PROJECT}/train_decoder_dpt.py \
    --data_csv ${DATASET}/data/train/SpatialVID_HQ_metadata.csv \
    --video_root ${DATASET}/videos/SpatialVID/videos \
    --encoder_ckpt /home/yexiaoyu/data/StreamVGGT/checkpoints.pth \
    --token_stats /home/yexiaoyu/data/token_stats.pt \
    --output_dir ${PROJECT}/ckpts/decoder_dpt/exp-7-gan \
    --seq_len 8 \
    --batch_size 10 --accum_steps 4 \
    --epochs 200 --lr 1e-4 --wd 1e-2 \
    --features 256 --multi_scale \
    --lpips_weight 1.0 --temporal_weight 0.05 --grad_weight 0.05 \
    --token_noise_std 0.02 --level_dropout 0.15 \
    --boundary_level 11 --boundary_only_prob 0.25 \
    --warmup_steps 500 \
    --ema_decay 0.999 --eval_every 5 --save_every 5 \
    --num_workers 32 --dtype bf16 \
    --output_depth --depth_root ${DATASET}/depths/SpatialVID/depths --depth_weight 0.1 \
    --gan \
    --resume --decoder_ckpt ${PROJECT}/ckpts/decoder_dpt/exp-7-gan/decoder_epoch5.pt
