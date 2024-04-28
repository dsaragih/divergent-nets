#!/bin/bash

for n_samples in {1..3}; do
    python unet_plusplus.py train \
        --num_epochs 150 \
        --device_id "$1"  \
        --mode "full_syn_train" \
        --img_dir "/root/divergent-nets/data/data_files/segmented-images" \
        --test_dir "/root/divergent-nets/data/data_files/test-images/cvc" \
        --pkl_path "/root/divergent-nets/data/data_files/cluster_$2/cc_samples_dil.pkl" \
        --n_samples "$n_samples" \
        --train_aug \
        --out_dir ./full-outputs/ \
        --tensorboard_dir ./full-outputs/ 
done