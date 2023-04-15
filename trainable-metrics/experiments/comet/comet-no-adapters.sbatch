#!/bin/bash 
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --job-name="COMET-Training-no-adapters"
#SBATCH --output=./logs/R-%x-%j.out

echo "==="
echo "Starting training"
echo "==="
WANDB_NAME="comet-no-adapters" \
    accelerate launch run.py \
    --data-config="comet" \
    --model-config="comet" \
    --no-tqdm