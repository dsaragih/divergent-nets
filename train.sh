#!/bin/bash
python unet_plusplus.py train \
    --num_epochs 150 \
    --device_id 0  \
    --mode "real_train" \
    --img_dir "/home/daniel/diffusion-gen/guided_diffusion/segmented-images" \
    --test_dir "/home/daniel/divergent-nets/data/data_files/test-images" \
    --pkl_path "/home/daniel/divergent-nets/data/data_files/GAN_dict.pkl" \
    --out_dir ./outputs/ \
    --tensorboard_dir ./outputs/ 

python unet_plusplus.py train \
    --num_epochs 150 \
    --device_id 0  \
    --mode "aug_syn_train" \
    --img_dir "/home/daniel/diffusion-gen/guided_diffusion/segmented-images" \
    --pkl_path "/home/daniel/divergent-nets/data/data_files/cc_styled_samples_dil.pkl" \
    --test_dir "/home/daniel/divergent-nets/data/data_files/test-images" \
    --n_samples 3 \
    --out_dir ./outputs/ \
    --tensorboard_dir ./outputs/ 

python unet_plusplus.py train \
    --num_epochs 150 \
    --device_id 0  \
    --mode "full_syn_train" \
    --img_dir "/home/daniel/diffusion-gen/guided_diffusion/segmented-images" \
    --pkl_path "/home/daniel/divergent-nets/data/data_files/cc_styled_samples_dil.pkl" \
    --test_dir "/home/daniel/divergent-nets/data/data_files/test-images" \
    --n_samples 1 \
    --out_dir ./outputs/ \
    --tensorboard_dir ./outputs/ 