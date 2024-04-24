#!/bin/bash

# !!! Select right project prefix and data dir
# ArchLinux (My PC)
# PROJECT_PREFIX=/home/tianen/doc/_XiDian/___FinalDesign/FinalDesign/final-design
# DATA_DIR=/home/tianen/doc/MachineLearningData/htd/
# Ubuntu Server
PROJECT_PREFIX=/home/lutianen/final-design
DATA_DIR=/home/lutianen/data/htd/

DIST_TYPE=${1}

# RUN Command
# nohup ./scripts/train_aae_aerorit_base_line.sh > /dev/null &

python ${PROJECT_PREFIX}/train_htd.py \
    --data_path ${DATA_DIR} \
    --dataset aerorit \
    --arch aae_aerorit \
    --cfg aae \
    --train_batch_size 150000 \
    --num_epochs 200 \
    --job_dir ${PROJECT_PREFIX}/experiments/ \
    --gpus 0 \
    --dist_type base_line \
    --warmup_epochs 0
