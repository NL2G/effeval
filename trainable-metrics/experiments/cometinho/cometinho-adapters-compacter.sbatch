#!/bin/bash 
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --job-name="cometinho-adapters-compacter"
#SBATCH --output=./logs/cometinho/R-%x-%j.out

echo "==="
echo "Starting training"
echo "==="
WANDB_NAME="cometinho-adapters-compacter" \
    accelerate launch run.py \
    --prepared-data="./data/cometinho-prepared" \
    --model-config="cometinho" \
    --no-tqdm \
    --use-adapters \
    --adapter-config="compacter"