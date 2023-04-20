#!/bin/bash 
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --job-name="debug-COMET-Training-adapters-pfeiffer"
#SBATCH --output=./logs/debug/R-%x-%j.out

echo "==="
echo "Starting training"
echo "==="
WANDB_NAME="comet-debug-adapters-pfeiffer" \
    accelerate launch run.py \
    --data-config="debug" \
    --model-config="debug" \
    --no-tqdm \
    --use-adapters \
    --adapter-config="pfeiffer"