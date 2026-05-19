#!/bin/bash
set -e

PROJECT=/home/yexiaoyu/work/VggAE-Diffusion

CUDA_VISIBLE_DEVICES=3 python ${PROJECT}/sample.py \
    --flow_ckpt ${PROJECT}/ckpts/diffusion_wan/exp-3-text/checkpoint_epoch0049.pt \
    --decoder_ckpt ${PROJECT}/ckpts/decoder_dpt/exp-6-big/decoder_final.pt \
    --token_stats_path ${PROJECT}/ckpts/token_stats.pt \
    --flow_levels 11 \
    --hidden_dim 768 --num_layers 12 --num_heads 12 \
    --num_samples 2 --num_steps 50 \
    --out_dir ${PROJECT}/samples/diffusion_wan_epoch0049 \
    --dtype bfloat16
