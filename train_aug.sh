#!/bin/bash

for n_samples in {1..3}; do
    python unet_plusplus.py train \
        --num_epochs 150 \
        --device_id "$1"  \
        --mode "aug_syn_train" \
        --img_dir "/root/divergent-nets/data/data_files/segmented-images" \
        --test_dir "/root/divergent-nets/data/data_files/test-images/cvc" \
        --pkl_path "/root/divergent-nets/data/data_files/cluster_$2/cc_samples_dil.pkl" \
        --n_samples "$n_samples" \
        --n_data 16 \
        --out_dir "./aug-outputs/" \
        --tensorboard_dir "./aug-outputs/"
done