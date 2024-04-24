#!/bin/bash

# !!! Select right project prefix and data dir
# ArchLinux (My PC)
# PROJECT_PREFIX=/home/tianen/doc/_XiDian/___FinalDesign/code/final-design
# DATA_DIR=/home/tianen/doc/MachineLearningData/
# Ubuntu Server
PROJECT_PREFIX=/home/lutianen/final-design
DATA_DIR=/home/lutianen/data/

# Run command
# ./scripts/test_resnet_cifar.sh PATH

PRETRAIN_MODEL_PATH=${1}

python ${PROJECT_PREFIX}/train_cifar.py \
    --data_path ${DATA_DIR} \
    --dataset CIFAR \
    --arch resnet_cifar \
    --cfg resnet56 \
    --num_batches_per_step 2 \
    --train_batch_size 256 \
    --eval_batch_size 100 \
    --gpus 2 \
    --dist_type gcc \
    --eval 1 \
    --pretrain_model ${PRETRAIN_MODEL_PATH}
