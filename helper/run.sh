#!/bin/bash
#SBATCH --job-name=train_dcase_ae        # Job name
#SBATCH --output=logs/%x_%j.out          # output (%x=job name, %j=job id)
#SBATCH --error=logs/%x_%j.err           # error
#SBATCH --partition=gpu                   # Partition/queue
#SBATCH --gres=gpu:1                      # Number of GPUs
#SBATCH --cpus-per-task=4                 # CPU cores
#SBATCH --mem=32G                         # Memory
#SBATCH --time=7:00:00                    # (HH:MM:SS)

# Load Python and activate environment
module load python/3.12.4
source faithfulenv/bin/activate

# Run the training script
python train_dcase_ae.py
