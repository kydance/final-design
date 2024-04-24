#!/bin/bash

# !!! Select right project prefix and data dir
# ArchLinux (My PC)
# PROJECT_PREFIX=/home/tianen/doc/_XiDian/___FinalDesign/FinalDesign/final-design
# DATA_DIR=/home/tianen/doc/MachineLearningData/htd/
# Ubuntu Server
PROJECT_PREFIX=/home/lutianen/final-design
DATA_DIR=/home/lutianen/data/htd/

DIST_TYPE=${1}

# !!! Set dist type: [abs, l1, gcc, knn]
# RUN Command
# nohup ./scripts/train_aae_aerorit.sh abs > /dev/null &

# 10 30 50 70 90 100
for i in 0.1 0.033 0.02 0.014 0.011 0.01
do
    echo ${i}
    echo ${DIST_TYPE1}
    python ${PROJECT_PREFIX}/train_htd.py \
        --data_path ${DATA_DIR} \
        --dataset aerorit \
        --arch aae_aerorit \
        --cfg aae \
        --train_batch_size 150000 \
        --num_epochs 200 \
        --job_dir ${PROJECT_PREFIX}/experiments/ \
        --cr ${i} \
        --gpus 2 \
        --dist_type ${DIST_TYPE}\
        --warmup_epochs 0
done
