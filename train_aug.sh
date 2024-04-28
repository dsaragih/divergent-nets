#!/bin/bash

python unet_plusplus.py train \
    --num_epochs 150 \
    --device_id 0  \
    --mode "aug_syn_train" \
    --img_dir "~/divergent-nets/data/data_files/segmented-images" \
    --test_dir "~/divergent-nets/data/data_files/test-images/cvc" \
    --pkl_path "~/divergent-nets/data/data_files/cluster_10/cc_samples_dil.pkl" \
    --n_samples 3 \
    --n_data 16 \
    --out_dir ./outputs/ \
    --tensorboard_dir ./outputs/ 