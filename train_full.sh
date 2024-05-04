#!/bin/bash
n_clusters=10

# Loop until n_samples reaches 4
for n_clusters in 40; do
    python unet_plusplus.py train \
        --num_epochs 140 \
        --device_id 0  \
        --mode "full_syn_train" \
        --img_dir "/root/divergent-nets/data/data_files/segmented-images" \
        --test_dir "/root/divergent-nets/data/data_files/test-images/hyperkvasir" \
        --pkl_path "/root/divergent-nets/data/data_files/cluster_$n_clusters/cc_samples_dil.pkl" \
        --n_samples 3 \
        --train_aug \
        --bs 16 \
        --n_data -1 \
        --out_dir ./outputs-full/ \
        --tensorboard_dir ./outputs-full/ 
done