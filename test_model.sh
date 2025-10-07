#!/bin/bash

# for MODEL in train_20250304_173920 train_20250304_174452 train_20250304_175029 train_20250304_175606 train_20250304_180143
# do
# python3 ./src/test.py \
#     --ds_name "miltiadous_deriv_uV_d1.0s_o0.0s" \
#     --classes "hc-ad" \
#     --ds_parent_dir "./data/" \
#     --device "cuda:0" \
#     -ckp '/home/zino/Projects/EEG/gnn-eeg-dementia/local/checkpoints/'$MODEL'/' \
#     -w 32 
# done

# for MODEL in train_20250304_180722 train_20250304_181354 train_20250304_182023 train_20250304_182656 train_20250304_183326
# do
# python3 ./src/test.py \
#     --ds_name "miltiadous_deriv_uV_d1.0s_o0.0s" \
#     --classes "hc-ftd-ad" \
#     --ds_parent_dir "./data/" \
#     --device "cuda:0" \
#     -ckp '/home/zino/Projects/EEG/gnn-eeg-dementia/local/checkpoints/'$MODEL'/' \
#     -w 32 
# done

# for MODEL in train_20250313_172332 train_20250313_172548 train_20250313_172805 train_20250313_173022 train_20250313_173240
# do
# python3 ./src/test.py \
#     --ds_name "APAVA_reformatted_d1.0s_o0.5s" \
#     --classes "hc-ad" \
#     --ds_parent_dir "./data/" \
#     --device "cuda:0" \
#     -ckp '/home/zino/Projects/EEG/gnn-eeg-dementia/local/checkpoints/'$MODEL'/' \
#     -w 32 
# done

# for MODEL in train_20250312_164150 train_20250312_164323 train_20250312_164456 train_20250312_164629 train_20250312_164803
# do
# python3 ./src/test.py \
#     --ds_name "ADSZ_reformatted_d1.0s_o0.5s" \
#     --classes "hc-ad" \
#     --ds_parent_dir "./data/" \
#     --device "cuda:0" \
#     -ckp '/home/zino/Projects/EEG/gnn-eeg-dementia/local/checkpoints/'$MODEL'/' \
#     -w 32 
# done

