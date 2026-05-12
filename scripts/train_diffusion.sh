#!/bin/bash
set -e

PROJECT=/home/yexiaoyu/work/VggAE-Diffusion
DATASET=/public2/LiZhen/yexiaoyu/dataset/spatial-vid-hq-oft

mkdir -p ${PROJECT}/ckpts

NUM_GPUS=5
GPU_IDS=0,4,5,6,7
CUDA_VISIBLE_DEVICES=${GPU_IDS} torchrun --nproc_per_node=${NUM_GPUS} --master_port=29503 \
    ${PROJECT}/train_diffusion.py \
    --csv ${DATASET}/data/train/SpatialVID_HQ_metadata.csv \
    --video_root ${DATASET}/videos/SpatialVID/videos \
    --encoder_ckpt /home/yexiaoyu/work/4DLangVGGT/ckpt/streamvggt/checkpoints.pth \
    --token_stats ${PROJECT}/ckpts/token_stats.pt \
    --annotation_index ${PROJECT}/ckpts/annotation_index.json \
    --output_dir ${PROJECT}/ckpts/diffusion_level11_dpt\
    --select_levels 11 --seq_len 8 \
    --hidden_dim 768 --num_layers 12 --num_heads 12 \
    --decoder_ckpt ${PROJECT}/ckpts/decoder_dpt/exp-5-dpt/decoder_final.pt \
    --recon_weight 0.05 --recon_every 1 \
    --recon_num_frames 4 --recon_t_min 0.25 --recon_grad_weight 0.05 \
    --input_noise 0.005 \
    --batch_size 10 --accum_steps 4 \
    --epochs 100 --lr 5e-5 --wd 1e-2 \
    --warmup_steps 2000 --ema_decay 0.9999 \
    --eval_every 10 --save_every 5 \
    --num_workers 10 --use_checkpoint
