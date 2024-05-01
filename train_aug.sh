#!/bin/bash

for n_samples in 1 2 3; do
    # Loop over n_data
    for n_data in 32 64 128; do
        python unet_plusplus.py train \
            --num_epochs 100 \
            --device_id 0  \
            --mode "aug_syn_train" \
            --img_dir "/root/divergent-nets/data/data_files/segmented-images" \
            --test_dir "/root/divergent-nets/data/data_files/test-images/polyp-gen" \
            --pkl_path "/root/divergent-nets/data/data_files/GAN_dict.pkl" \
            --n_samples "$n_samples" \
            --bs 32 \
            --n_data "$n_data" \
            --out_dir "./outputs-aug/" \
            --tensorboard_dir "./outputs-aug/"
    done
done