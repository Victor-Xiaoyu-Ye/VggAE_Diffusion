#!/bin/bash
# exp-8: DPTHead + multi-layer mean (RAEv2-style: avg levels 4,11,17,23)
set -e

PROJECT=/home/yexiaoyu/work/VggAE-Diffusion
DATASET=/public2/LiZhen/yexiaoyu/dataset/spatial-vid-hq-oft

mkdir -p ${PROJECT}/ckpts

NUM_GPUS=8
GPU_IDS=0,1,2,3,4,5,6,7
CUDA_VISIBLE_DEVICES=${GPU_IDS} torchrun --nproc_per_node=${NUM_GPUS} --master_port=29504 \
  ${PROJECT}/train_decoder_dpt.py \
    --data_csv ${DATASET}/data/train/SpatialVID_HQ_metadata.csv \
    --video_root ${DATASET}/videos/SpatialVID/videos \
    --encoder_ckpt /home/yexiaoyu/work/4DLangVGGT/ckpt/streamvggt/checkpoints.pth \
    --token_stats ${PROJECT}/ckpts/token_stats.pt \
    --output_dir ${PROJECT}/ckpts/decoder_dpt/exp-8-mean \
    --seq_len 8 \
    --batch_size 5 --accum_steps 4 \
    --epochs 120 --lr 1e-4 --wd 1e-2 \
    --features 256 --multi_layer_mean \
    --lpips_weight 1.0 --temporal_weight 0.05 --grad_weight 0.05 \
    --token_noise_std 0.02 --level_dropout 0.15 \
    --boundary_level 11 --boundary_only_prob 0.25 \
    --warmup_steps 500 \
    --ema_decay 0.999 --eval_every 5 --save_every 5 \
    --num_workers 10 --dtype bf16 \
    --output_depth --depth_root ${DATASET}/depths/SpatialVID/depths --depth_weight 0.1 \
    --resume --decoder_ckpt ${PROJECT}/ckpts/decoder_dpt/exp-8-mean/decoder_epoch60.pt
