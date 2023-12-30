#!/bin/bash

conda activate pinnacle

# Rheumatoid Arthritis (EFO_0000685)
python train.py \
    --task_name=EFO_0000685 \
    --embeddings_dir=../data/pinnacle_embeds/ \
    --positive_proteins_prefix ../data/therapeutic_target_task/positive_proteins_EFO_0000685 \
    --negative_proteins_prefix ../data/therapeutic_target_task/negative_proteins_EFO_0000685 \
    --data_split_path ../data/therapeutic_target_task/data_split_EFO_0000685 \
    --actn=relu \
    --dropout=0.2 \
    --hidden_dim_1=32 \
    --hidden_dim_2=8 \
    --lr=0.01 \
    --norm=bn \
    --order=dn \
    --wd=0.001 \
    --random_state 1 \
    --num_epoch=2000

# Inflammatory bowel disease (EFO_0003767)
python train.py \
    --task_name=EFO_0003767 \
    --embeddings_dir=../data/pinnacle_embeds/ \
    --positive_proteins_prefix ../data/therapeutic_target_task/positive_proteins_EFO_0003767 \
    --negative_proteins_prefix ../data/therapeutic_target_task/negative_proteins_EFO_0003767 \
    --data_split_path ../data/therapeutic_target_task/data_split_EFO_0003767 \
    --actn=relu \
    --dropout=0.4 \
    --hidden_dim_1=32 \
    --hidden_dim_2=8 \
    --lr=0.001 \
    --norm=ln \
    --order=nd \
    --wd=0.0001 \
    --random_state 1 \
    --num_epoch=2000
    