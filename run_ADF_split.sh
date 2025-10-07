#!/bin/bash


# for SEED in 41 42 43 44 45
# do
#     python src/single_fold.py \
#         --ds_name "miltiadous_deriv_uV_d1.0s_o0.0s" \
#         --classes "hc-ad" \
#         --ds_parent_dir "./data/" \
#         --device "cuda:0" \
#         --num_epochs 80 \
#         --batch_size 256 \
#         --lr 5e-4 \
#         --log_dir "./local/ADFSplit_multiscale_final/" \
#         -w 32 \
#         --seed $SEED
# done


# # Training + test 3-class

# for SEED in 41 42 43 44 45
# do
#     echo $SEED
#     python src/single_fold.py \
#         --ds_name "miltiadous_deriv_uV_d1.0s_o0.0s" \
#         --classes "hc-ftd-ad" \
#         --ds_parent_dir "./data/" \
#         --device "cuda:0" \
#         --num_epochs 80 \
#         --batch_size 256 \
#         --lr 5e-4 \
#         --log_dir "./local/ADFSplit_multiscale_final/" \
#         -w 32 \
#         --seed $SEED
# done



# for SEED in 41 42 43 44 45
# do
#     python src/single_fold.py \
#         --ds_name "ADSZ_reformatted_d1.0s_o0.5s" \
#         --classes "hc-ad" \
#         --ds_parent_dir "./data/" \
#         --device "cuda:0" \
#         --num_epochs 80 \
#         --batch_size 128 \
#         --lr 5e-4 \
#         --log_dir "./local/ADFSplit_multiscale_final/" \
#         -w 32 \
#         --seed $SEED
# done


# for SEED in 41 42 43 44 45
# do
#     python src/single_fold.py \
#         --ds_name "APAVA_reformatted_d1.0s_o0.5s" \
#         --classes "hc-ad" \
#         --ds_parent_dir "./data/" \
#         --device "cuda:0" \
#         --num_epochs 80 \
#         --batch_size 256 \
#         --lr 5e-4 \
#         --log_dir "./local/ADFSplit_multiscale_final/" \
#         -w 32 \
#         --seed $SEED
# done

# 
for SEED in 41 42 43 44 45
do
    python src/single_fold.py \
        --ds_name "brainlat_2cl_uV_d1.0s_o0.0s_repo_islab" \
        --classes "hc-ad" \
        --ds_parent_dir "./data/" \
        --device "cuda:0" \
        --num_epochs 25 \
        --batch_size 256 \
        --lr 1e-4 \
        --log_dir "./local/ADFSplit_brainlat_debug/" \
        -w 32 \
        --seed $SEED
done




