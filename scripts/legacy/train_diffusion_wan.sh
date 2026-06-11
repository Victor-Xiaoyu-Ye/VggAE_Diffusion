#!/bin/bash
set -e

PROJECT=/cache/yexiaoyu/VggAE_Diffusion
DATASET=/cache/dataset/spatial-vid-hq-oft

mkdir -p $OUTPUT_URL/yexiaoyu/ckpts

NUM_GPUS=8
GPU_IDS=0,1,2,3,4,5,6,7

# multi-node config
MASTER_ADDR=$(echo ${VC_WORKER_HOSTS} | cut -d',' -f1)
NNODES=${VC_WORKER_NUM}
NODE_RANK=${VC_TASK_INDEX}
MASTER_PORT=29506

echo "MASTER_ADDR=${MASTER_ADDR}"
echo "NNODES=${NNODES}"
echo "NODE_RANK=${NODE_RANK}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "NUM_GPUS=${NUM_GPUS}"
echo "GPU_IDS=${GPU_IDS}"

ASCEND_RT_VISIBLE_DEVICES=${GPU_IDS} \
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  OMP_NUM_THREADS=4 \
python -m torch.distributed.run \
  --nproc_per_node=${NUM_GPUS} \
  --nnodes=${NNODES} \
  --node_rank=${NODE_RANK} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  ${PROJECT}/train_diffusion_wan.py \
    --csv ${DATASET}/data/train/SpatialVID_HQ_metadata.csv \
    --video_root ${DATASET}/videos/SpatialVid/HQ//videos \
    --encoder_ckpt /cache/yexiaoyu/vggae_ref/StreamVGGT/checkpoints.pth \
    --token_stats /cache/yexiaoyu/vggae_ref/token_stats.pt \
    --wan_ckpt_dir /cache/yexiaoyu/vggae_ref/Wan2.1-T2V-1.3B \
    --output_dir $OUTPUT_URL/yexiaoyu/ckpts/diffusion_wan/exp-2-lora-ascend \
    --select_levels 11 --seq_len 8 \
    --lora_rank 0 --lora_alpha 128 \
    --text_cond --cfg_dropout 0.1 \
    --recon_weight 0.0 --recon_every 0 \
    --recon_num_frames 1 --recon_t_min 0.25 --recon_grad_weight 0.05 \
    --input_noise 0.005 \
    --batch_size 1 --accum_steps 4 \
    --epochs 100 --lr 1e-4 --wd 1e-2 \
    --warmup_steps 1000 --ema_decay 0.9999 \
    --eval_every 10 --save_every 5 \
    --num_workers 8 --use_checkpoint
