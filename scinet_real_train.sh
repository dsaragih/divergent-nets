#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=1:00:0

module load anaconda3/2021.05   
module load cuda/11.8.0  
module load openblas/0.3.15
module load gcc/11.3.0  
module load openmpi/4.1.4+ucx-1.11.2
source activate divnets

python unet_plusplus.py train \
    --num_epochs 150 \
    --mode "real_train" \
    --img_dir "~/divergent-nets/data/data_files/segmented-images" \
    --test_dir "~/divergent-nets/data/data_files/test-images/cvc" \
    --pkl_path "~/divergent-nets/data/data_files/cluster_10/cc_samples_dil.pkl" \
    --out_dir $SCRATCH/outputs/ \
    --n_data $1 \
    --tensorboard_dir $SCRATCH/outputs/ 