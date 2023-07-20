#!/bin/bash

conda activate pinnacle

# Rheumatoid Arthritis (EFO_0000685)
python train.py \
    --disease=EFO_0000685 \
    --embeddings_dir=../data/pinnacle_embeds/ \
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
    --disease=EFO_0003767 \
    --embeddings_dir=../data/pinnacle_embeds/ \
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
