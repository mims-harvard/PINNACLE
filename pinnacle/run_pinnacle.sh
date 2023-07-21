#!/bin/bash

conda activate pinnacle

python train.py \
        --G_f ../data/networks/global_ppi_edgelist.txt \
        --ppi_dir ../data/networks/ppi_edgelists/ \
        --mg_f ../data/networks/mg_edgelist.txt \
        --batch_size=8 \
        --dropout=0.6 \
        --feat_mat=1024 \
        --hidden=64 \
        --lmbda=0.1 \
        --loader=graphsaint \
        --lr=0.01 \
        --lr_cent=0.1 \
        --n_heads=8 \
        --output=16 \
        --pc_att_channels=16 \
        --theta=0.3 \
        --wd=1e-05 \
        --epochs=250