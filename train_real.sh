#!/bin/bash


python unet_plusplus.py train \
    --num_epochs 150 \
    --device_id $1  \
    --mode "real_train" \
    --img_dir "/root/divergent-nets/data/data_files/segmented-images" \
    --test_dir "/root/divergent-nets/data/data_files/test-images/cvc" \
    --pkl_path "/root/divergent-nets/data/data_files/cluster_10/cc_samples_dil.pkl" \
    --out_dir ./outputs/ \
    --n_data 32 \
    --tensorboard_dir ./outputs/ 