#!/bin/bash

# !!! Select right project prefix and data dir
# ArchLinux (My PC)
# PROJECT_PREFIX=/home/tianen/doc/_XiDian/___FinalDesign/FinalDesign/final-design
# DATA_DIR=/home/tianen/doc/MachineLearningData/
# Ubuntu Server
PROJECT_PREFIX=/home/lutianen/final-design
DATA_DIR=/home/lutianen/data/

MODEL=resnet56
PRETRAIN_MODEL_PATH=${PROJECT_PREFIX}/pretrain_models/${MODEL}_cifar10.pt

DIST_TYPE=${1}

# !!! Set dist type: [abs, l1, gcc, knn]
# RUN Command
# nohup ./scripts/train_resnet_cifar_wm5.sh abs > resnet_cifar10_wm5.out &

# 10 30 50 70 90 100
for i in 0.1 0.033 0.02 0.014 0.011 0.01
do
    echo ${i}
    echo ${DIST_TYPE}

    python ${PROJECT_PREFIX}/train_cifar.py \
        --data_path ${DATA_DIR} \
        --dataset CIFAR \
        --arch resnet_cifar \
        --cfg ${MODEL} \
        --pretrain_model ${PRETRAIN_MODEL_PATH} \
        --num_batches_per_step 3 \
        --train_batch_size 128 \
        --eval_batch_size 100 \
        --num_epochs 200 \
        --job_dir ${PROJECT_PREFIX}/experiments/ \
        --momentum 0.9 \
        --lr 0.01 \
        --lr_type step \
        --lr_decay_step 50 100 150 \
        --weight_decay 5e-3 \
        --cr ${i} \
        --gpus 2 \
        --dist_type ${DIST_TYPE} \
        --warmup_epochs 5 \
        --warmup_coeff
done
