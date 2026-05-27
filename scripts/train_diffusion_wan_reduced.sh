#!/bin/bash
# Diffusion with dimensionality reduction: channel 2048→256 + spatial 37→18
# Latent: 22M → 660K dims (34x reduction), Wan backbone with 324 tokens/frame
set -e

PROJECT=/home/yexiaoyu/work/VggAE-Diffusion
DATASET=/public2/LiZhen/yexiaoyu/dataset/spatial-vid-hq-oft

mkdir -p ${PROJECT}/ckpts

NUM_GPUS=8
GPU_IDS=0,1,2,3,4,5,6,7
CUDA_VISIBLE_DEVICES=${GPU_IDS} torchrun --nproc_per_node=${NUM_GPUS} --master_port=29508 \
  ${PROJECT}/train_diffusion_wan_reduced.py \
    --csv ${DATASET}/data/train/SpatialVID_HQ_metadata.csv \
    --video_root ${DATASET}/videos/SpatialVID/videos \
    --encoder_ckpt /home/yexiaoyu/work/4DLangVGGT/ckpt/streamvggt/checkpoints.pth \
    --token_stats ${PROJECT}/ckpts/token_stats.pt \
    --wan_ckpt_dir ${PROJECT}/Wan2.1/checkpoints/Wan2.1-T2V-1.3B \
    --output_dir ${PROJECT}/ckpts/diffusion_wan/exp-4-reduced \
    --select_levels 11 --seq_len 8 \
    --lora_rank 64 --lora_alpha 128 \
    --text_cond --cfg_dropout 0.1 \
    --decoder_ckpt ${PROJECT}/ckpts/decoder_dpt/exp-6-big/decoder_final.pt \
    --recon_weight 0.05 --recon_every 1 \
    --recon_num_frames 4 --recon_t_min 0.25 --recon_grad_weight 0.05 \
    --input_noise 0.005 \
    --batch_size 2 --accum_steps 4 \
    --epochs 50 --lr 5e-5 --wd 1e-2 \
    --warmup_steps 1000 --ema_decay 0.9999 \
    --eval_every 10 --save_every 5 \
    --num_workers 4 --use_checkpoint
