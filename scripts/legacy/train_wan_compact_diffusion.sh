#!/bin/bash
# Phase 2 (Wan backbone): Flow matching on compact latent z_g using Wan 1.3B
# DUAL time conditioning: concat input + adaLN blocks
# Trainable: modulation + time_emb + QKV (~215M params)
# Uses exp-1-big autoencoder (512-dim, PSNR 20.6 dB)
set -e

PROJECT=/home/yexiaoyu/work/VggAE-Diffusion
DATASET=/public2/LiZhen/yexiaoyu/dataset/spatial-vid-hq-oft

AE_CKPT=${PROJECT}/ckpts/autoencoder/exp-1-big/checkpoint_final.pt
WAN_CKPT_DIR=/home/yexiaoyu/work/VggAE-Diffusion/Wan2.1/checkpoints/Wan2.1-T2V-1.3B

mkdir -p ${PROJECT}/ckpts/diffusion_wan_compact

NUM_GPUS=8
GPU_IDS=0,1,2,3,4,5,6,7
CUDA_VISIBLE_DEVICES=${GPU_IDS} torchrun --nproc_per_node=${NUM_GPUS} --master_port=29520 \
  ${PROJECT}/train_wan_compact_diffusion.py \
    --csv ${DATASET}/data/train/SpatialVID_HQ_metadata.csv \
    --video_root ${DATASET}/videos/SpatialVID/videos \
    --encoder_ckpt /home/yexiaoyu/work/4DLangVGGT/ckpt/streamvggt/checkpoints.pth \
    --autoencoder_ckpt ${AE_CKPT} \
    --wan_ckpt_dir ${WAN_CKPT_DIR} \
    --output_dir ${PROJECT}/ckpts/diffusion_wan_compact/exp-1 \
    --latent_dim 512 --latent_grid 18 \
    --levels 4 11 17 23 \
    --decoder_base_dim 384 --decoder_num_resblocks 2 \
    --decoder_pixel_shuffle --decoder_temporal_blocks 2 --decoder_version v2 \
    --text_cond --cfg_dropout 0.1 \
    --annotation_index ${PROJECT}/ckpts/annotation_index.json \
    --decoder_aux --recon_weight 0.05 --recon_every 1 \
    --batch_size 5 --accum_steps 4 \
    --epochs 50 --lr 1e-4 --wd 1e-2 \
    --warmup_steps 1000 --ema_decay 0.9999 \
    --seq_len 8 --target_size 518 \
    --num_workers 5 --dtype bf16 \
    --eval_every 2 --save_every 2 \
    --resume ${PROJECT}/ckpts/diffusion_wan_compact/exp-1/checkpoint_epoch0001.pt
