#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=7:00:0

module load anaconda3/2021.05
module load cuda/11.4.4
module load openblas/0.3.15
module load gcc/11.3.0
module load openmpi/4.1.4+ucx-1.11.2
source activate divnets
# Initialize n_clusters to 10
n_clusters=10

# Loop until n_clusters reaches 4
while [ "$n_clusters" -le 40 ]; do
    python $SCRATCH/divergent-nets/unet_plusplus.py train \
        --num_epochs 150 \
        --mode "full_syn_train" \
        --img_dir $SCRATCH/divergent-nets/data/data_files/segmented-images \
        --test_dir $SCRATCH/divergent-nets/data/data_files/test-images/$3 \
        --pkl_path $SCRATCH/divergent-nets/data/data_files/cluster_$n_clusters/cc_samples_dil.pkl \
        --n_samples $1 \
        --train_aug \
        --bs 32 \
        --n_data $2 \
        --out_dir $SCRATCH/divergent-nets/outputs-full/ \
        --tensorboard_dir $SCRATCH/divergent-nets/outputs-full/

    # Increment n_clusters
    n_clusters=$((n_clusters + 10))
done