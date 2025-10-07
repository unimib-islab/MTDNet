#!/bin/bash

# no augmentations
# for MODEL in  train_20250407_093322 train_20250407_093825 train_20250407_094327 train_20250407_094830 train_20250407_095341

# flip
# for MODEL in  train_20250407_100158 train_20250407_100700 train_20250407_101205 train_20250407_101708 train_20250407_102212

# random channels
# for MODEL in train_20250407_112308 train_20250407_112810 train_20250407_113315 train_20250407_113829 train_20250407_114339 

# random zeros
# train_20250407_115412 train_20250407_115922 train_20250407_120431 train_20250407_120939 train_20250407_121452

# flip + random channels
# train_20250407_122255 train_20250407_122804 train_20250407_123312 train_20250407_123819 train_20250407_124328

# flip + random zeros
# train_20250407_131400 train_20250407_131913 train_20250407_132422 train_20250407_132931 train_20250407_133440

# random channels + random zeros

# for MODEL in train_20250407_133959 train_20250407_134514 train_20250407_135023 train_20250407_135534 train_20250407_140043
# do
# python3 ./src/test.py \
#     --ds_name "miltiadous_deriv_uV_d1.0s_o0.0s" \
#     --classes "hc-ad" \
#     --ds_parent_dir "./data/" \
#     --device "cuda:0" \
#     -ckp '/home/zino/Projects/EEG/gnn-eeg-dementia/local/checkpoints/'$MODEL'/' \
#     --version "last" \
#     -w 32 
# done


#Ablation on features
MODEL=train_20250304_180143

python3 ./src/test.py \
    --ds_name "miltiadous_deriv_uV_d1.0s_o0.0s" \
    --classes "hc-ad" \
    --ds_parent_dir "./data/" \
    --device "cuda:0" \
    -ckp '/home/zino/Projects/EEG/gnn-eeg-dementia/local/checkpoints/'$MODEL'/' \
    --version 'last' \
    -w 32 