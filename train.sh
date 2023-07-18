#!/bin/bash





python unet_plusplus.py train \
    --num_epochs 2 \
    --device_id 0  \
    --pkl_path "/home/daniel/divergent-nets/data/data_files/styled_samples_3x256x256x4.pkl" \
    --out_dir ./outputs/ \
    --tensorboard_dir ./outputs/ 