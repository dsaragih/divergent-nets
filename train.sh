#!/bin/bash
python unet_plusplus.py train \
    --num_epochs 300 \
    --device_id 0  \
    --mode "real_train" \
    --img_dir "/home/daniel/diff-seg/core/guided_diffusion/segmented-images" \
    --pkl_path "/home/daniel/divergent-nets/data/data_files/GAN_dict.pkl" \
    --out_dir ./outputs/ \
    --tensorboard_dir ./outputs/ 

python unet_plusplus.py train \
    --num_epochs 60 \
    --device_id 0  \
    --mode "aug_syn_train" \
    --img_dir "/home/daniel/diff-seg/core/guided_diffusion/segmented-images" \
    --pkl_path "/home/daniel/divergent-nets/data/data_files/GAN_dict.pkl" \
    --out_dir ./outputs/ \
    --tensorboard_dir ./outputs/ 

python unet_plusplus.py train \
    --num_epochs 60 \
    --device_id 0  \
    --mode "full_syn_train" \
    --pkl_path "/home/daniel/divergent-nets/data/data_files/GAN_dict.pkl" \
    --out_dir ./outputs/ \
    --tensorboard_dir ./outputs/ 