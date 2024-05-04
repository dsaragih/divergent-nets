#!/bin/bash

for n_data in 64 128; do
    python unet_plusplus.py train \
        --num_epochs 150 \
        --device_id 0 \
        --mode "real_train" \
        --img_dir "/root/divergent-nets/data/data_files/segmented-images" \
        --test_dir "/root/divergent-nets/data/data_files/test-images/cvc" \
        --pkl_path "/root/divergent-nets/data/data_files/cluster_10/cc_samples_dil.pkl" \
        --out_dir ./outputs/ \
        --n_data $n_data \
        --bs 32 \
        --tensorboard_dir ./outputs/ 
done