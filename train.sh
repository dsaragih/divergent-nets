#!/bin/bash





python unet_plusplus.py train \
    --num_epochs 2 \
    --device_id 0  \
    --out_dir ./outputs/ \
    --tensorboard_dir ./outputs/ 