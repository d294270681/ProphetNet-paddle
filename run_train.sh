#!/bin/bash
DATASET=$1

if [ "$DATASET" == cnndm ]
then
python train_prophetnet.py \
    --dataset=cnndm \
    --pretrained_model_path=./model_state.pdparams \
    --batch_size=4 \
    --epochs=4 \
    --lr=0.0001 \
    --warmup_init_lr=1e-07 \
    --warmup_updates=1000 \
    --clip_norm=0.1 \
    --num_workers=4 \
    --output_dir=./ckpt/cnndm
else
python train_prophetnet.py \
    --dataset=gigaword \
    --pretrained_model_path=./model_state.pdparams \
    --batch_size=16 \
    --epochs=6 \
    --lr=0.0001 \
    --warmup_init_lr=1e-07 \
    --warmup_updates=1000 \
    --clip_norm=0.1 \
    --num_workers=8 \
    --output_dir=./ckpt/gigaword
fi