#!/bin/bash
# Initialize n_samples to 1
n_samples=1

# Loop until n_samples reaches 4
while [ "$n_samples" -le 3 ]; do
    python unet_plusplus.py train \
        --num_epochs 100 \
        --device_id 0  \
        --mode "aug_syn_train" \
        --img_dir "/root/divergent-nets/data/data_files/segmented-images" \
        --test_dir "/root/divergent-nets/data/data_files/test-images/polyp-gen" \
        --pkl_path "/root/divergent-nets/data/data_files/GAN_dict.pkl" \
        --n_samples "$n_samples" \
        --n_data 16 \
        --out_dir "./outputs-aug/" \
        --tensorboard_dir "./outputs-aug/"

    # Increment n_samples
    n_samples=$((n_samples + 1))
done