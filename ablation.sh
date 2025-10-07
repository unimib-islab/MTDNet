#!/bin/bash


for SEED in 41 42 43 44 45
do
    python src/single_fold.py \
        --ds_name "miltiadous_deriv_uV_d1.0s_o0.0s" \
        --classes "hc-ad" \
        --ds_parent_dir "./data/" \
        --device "cuda:0" \
        --num_epochs 80 \
        --batch_size 256 \
        --lr 5e-4 \
        --log_dir "./local/ADFSplit_multiscale_ablation/" \
        -w 32 \
        --seed $SEED
done





