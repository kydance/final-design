#!/bin/bash

# !!! Select right project prefix and data dir
# ArchLinux (My PC)
# PROJECT_PREFIX=/home/tianen/doc/_XiDian/___FinalDesign/FinalDesign/final_design
# DATA_DIR=/home/tianen/doc/MachineLearningData/
# Ubuntu Server
PROJECT_PREFIX=/home/lutianen/final_design
DATA_DIR=/home/lutianen/data/

# nohup ./script/train_vgg_cifar.sh > vgg_cifar10.out &

# 10
python ${PROJECT_PREFIX}/train_cifar.py \
        --data_path ${DATA_DIR} \
        --dataset CIFAR \
        --arch vgg_cifar \
        --cfg vgg16 \
        --pretrain_model ${PROJECT_PREFIX}/pretrain_model/vgg16_cifar10.pt \
        --num_batches_per_step 3 \
        --train_batch_size 128 \
        --eval_batch_size 100 \
        --num_epochs 200 \
        --job_dir ${PROJECT_PREFIX}/experiment/ \
        --momentum 0.9 \
        --lr 0.01 \
        --lr_type step \
        --lr_decay_step 50 100 \
        --weight_decay 0.0005 \
        --cr 0.1 \
        --gpus 0

