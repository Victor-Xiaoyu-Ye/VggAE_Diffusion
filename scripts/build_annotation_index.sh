#!/bin/bash
set -e

PROJECT=/home/yexiaoyu/work/VggAE-Diffusion
DATASET=/public2/LiZhen/yexiaoyu/dataset/spatial-vid-hq-oft

mkdir -p ${PROJECT}/ckpts

cd ${PROJECT}
python data/annotation_index.py \
    --csv_path ${DATASET}/data/train/SpatialVID_HQ_metadata.csv \
    --anno_dir ${DATASET}/annotations/SpatialVID/annotations \
    --out_path ${PROJECT}/ckpts/annotation_index.json
