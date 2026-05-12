#!/bin/bash
set -e

PROJECT=/home/yexiaoyu/work/VggAE-Diffusion
DATASET=/public2/LiZhen/yexiaoyu/dataset/spatial-vid-hq-oft

mkdir -p ${PROJECT}/ckpts

NUM_GPUS=5
GPU_IDS=0,4,5,6,7
CUDA_VISIBLE_DEVICES=${GPU_IDS} torchrun --nproc_per_node=${NUM_GPUS} --master_port=29506 \
  ${PROJECT}/train_diffusion_wan.py \
    --csv ${DATASET}/data/train/SpatialVID_HQ_metadata.csv \
    --video_root ${DATASET}/videos/SpatialVID/videos \
    --encoder_ckpt /home/yexiaoyu/work/4DLangVGGT/ckpt/streamvggt/checkpoints.pth \
    --token_stats ${PROJECT}/ckpts/token_stats.pt \
    --wan_ckpt_dir /home/yexiaoyu/work/VggAE-Diffusion/Wan2.1/checkpoints/Wan2.1-T2V-1.3B \
    --output_dir ${PROJECT}/ckpts/diffusion_wan/exp-3-text \
    --select_levels 11 --seq_len 8 \
    --lora_rank 64 --lora_alpha 128 \
    --text_cond --cfg_dropout 0.1 \
    --decoder_ckpt ${PROJECT}/ckpts/decoder_dpt/exp-5-dpt/decoder_final.pt \
    --recon_weight 0.05 --recon_every 1 \
    --recon_num_frames 4 --recon_t_min 0.25 --recon_grad_weight 0.05 \
    --input_noise 0.005 \
    --batch_size 1 --accum_steps 4 \
    --epochs 50 --lr 1e-4 --wd 1e-2 \
    --warmup_steps 1000 --ema_decay 0.9999 \
    --eval_every 10 --save_every 1 \
    --num_workers 4 --use_checkpoint \
    --resume ${PROJECT}/ckpts/diffusion_wan/exp-3-text/checkpoint_epoch0019.pt
