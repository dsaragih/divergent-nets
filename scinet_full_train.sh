#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=1:00:0

module load anaconda3/2021.05   
module load cuda/11.4.4 
module load openblas/0.3.15
module load gcc/11.3.0  
module load openmpi/4.1.4+ucx-1.11.2
source activate divnets
# Initialize n_samples to 1
n_samples=1

# Loop until n_samples reaches 4
while [ "$n_samples" -le 3 ]; do
    python $SCRATCH/divergent-nets/unet_plusplus.py train \
        --num_epochs 150 \
        --mode "full_syn_train" \
        --img_dir $SCRATCH/divergent-nets/data/data_files/segmented-images \
        --test_dir $SCRATCH/divergent-nets/data/data_files/test-images/$3 \
        --pkl_path $SCRATCH/divergent-nets/data/data_files/cluster_$1/cc_samples_dil.pkl \
        --n_samples "$n_samples" \
        --train_aug \
        --n_data $2 \
        --out_dir $SCRATCH/divergent-nets/outputs-full/ \
        --tensorboard_dir $SCRATCH/divergent-nets/outputs-full/ 

    # Increment n_samples
    n_samples=$((n_samples + 1))
done