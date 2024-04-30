#!/bin/bash
# Initialize n_samples to 1
n_samples=1

# Loop until n_samples reaches 4
while [ "$n_samples" -le 3 ]; do
    python unet_plusplus.py train \
        --num_epochs 150 \
        --device_id "$1"  \
        --mode "full_syn_train" \
        --img_dir "/root/divergent-nets/data/data_files/segmented-images" \
        --test_dir "/root/divergent-nets/data/data_files/test-images/polyp-gen" \
        --pkl_path "/root/divergent-nets/data/data_files/GAN_dict.pkl" \
        --n_samples "$n_samples" \
        --train_aug \
        --n_data -1 \
        --out_dir ./outputs-full/ \
        --tensorboard_dir ./outputs-full/ 

    # Increment n_samples
    n_samples=$((n_samples + 1))
done