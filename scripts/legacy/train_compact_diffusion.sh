#!/bin/bash
# Phase 2: Train flow matching diffusion on compact latent z_g (with rescale)
# z_g * scale → std≈1, fixes signal-to-noise ratio for flow matching
set -e

PROJECT=/home/yexiaoyu/work/VggAE-Diffusion
DATASET=/public2/LiZhen/yexiaoyu/dataset/spatial-vid-hq-oft

# exp-1-big autoencoder (PSNR 20.6 dB, best available)
AE_CKPT=${PROJECT}/ckpts/autoencoder/exp-1-big/checkpoint_final.pt

mkdir -p ${PROJECT}/ckpts/diffusion_compact

NUM_GPUS=7
GPU_IDS=1,2,3,4,5,6,7
CUDA_VISIBLE_DEVICES=${GPU_IDS} torchrun --nproc_per_node=${NUM_GPUS} --master_port=29514 \
  ${PROJECT}/train_compact_diffusion.py \
    --csv ${DATASET}/data/train/SpatialVID_HQ_metadata.csv \
    --video_root ${DATASET}/videos/SpatialVID/videos \
    --encoder_ckpt /home/yexiaoyu/work/4DLangVGGT/ckpt/streamvggt/checkpoints.pth \
    --autoencoder_ckpt ${AE_CKPT} \
    --output_dir ${PROJECT}/ckpts/diffusion_compact/exp-3-rescale \
    --latent_dim 512 --latent_grid 18 \
    --model_dim 768 --spatial_depth 8 --temporal_depth 4 --num_heads 12 \
    --levels 4 11 17 23 \
    --decoder_base_dim 384 --decoder_num_resblocks 2 \
    --decoder_pixel_shuffle --decoder_temporal_blocks 2 \
    --text_cond --cfg_dropout 0.1 \
    --annotation_index ${PROJECT}/ckpts/annotation_index.json \
    --decoder_aux --recon_weight 0.05 --recon_every 1 \
    --rescale \
    --batch_size 5 --accum_steps 4 \
    --epochs 50 --lr 1e-4 --wd 1e-2 \
    --warmup_steps 1000 --ema_decay 0.9999 \
    --seq_len 8 --target_size 518 \
    --num_workers 4 --dtype bf16 \
    --eval_every 2 --save_every 2
