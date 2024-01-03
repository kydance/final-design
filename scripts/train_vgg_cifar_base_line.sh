#!/bin/bash

# !!! Select right project prefix and data dir
# ArchLinux (My PC)
# PROJECT_PREFIX=/home/tianen/doc/_XiDian/___FinalDesign/FinalDesign/final_design
# DATA_DIR=/home/tianen/doc/MachineLearningData/
# Ubuntu Server
PROJECT_PREFIX=/home/lutianen/final-design
DATA_DIR=/home/lutianen/data/

# RUN Command
# nohup ./scripts/train_vgg_cifar_base_line.sh > vgg_cifar10_base_line.out &

python ${PROJECT_PREFIX}/train_cifar.py \
    --data_path ${DATA_DIR} \
    --dataset CIFAR \
    --arch vgg_cifar \
    --cfg vgg16 \
    --num_batches_per_step 3 \
    --train_batch_size 128 \
    --eval_batch_size 100 \
    --num_epochs 200 \
    --job_dir ${PROJECT_PREFIX}/experiments/ \
    --momentum 0.9 \
    --lr 0.01 \
    --lr_type step \
    --lr_decay_step 50 100 150\
    --weight_decay 5e-3 \
    --gpus 2 \
    --dist_type base_line
