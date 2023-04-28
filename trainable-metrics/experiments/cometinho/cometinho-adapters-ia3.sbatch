#!/bin/bash 
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4-00:00:00
#SBATCH --job-name="cometinho-adapters-ia3"
#SBATCH --output=./logs/cometinho/R-%x-%j.out

echo "==="
echo "Starting iteration-1"
echo "==="
WANDB_RUN_GROUP="cometinho-adapters-ia3" WANDB_NAME="cometinho-adapters-ia3-21291" \
    accelerate launch run.py \
    --prepared-data="./data/cometinho-prepared" \
    --model-config="cometinho" \
    --no-tqdm \
    --use-adapters \
    --seed=21291 \
    --adapter-config="ia3"

echo "==="
echo "Starting iteration-2"
echo "==="
WANDB_RUN_GROUP="cometinho-adapters-ia3" WANDB_NAME="cometinho-adapters-ia3-2169" \
    accelerate launch run.py \
    --prepared-data="./data/cometinho-prepared" \
    --model-config="cometinho" \
    --no-tqdm \
    --use-adapters \
    --seed=2169 \
    --adapter-config="ia3"


echo "==="
echo "Starting iteration-3"
echo "==="
WANDB_RUN_GROUP="cometinho-adapters-ia3" WANDB_NAME="cometinho-adapters-ia3-70800" \
    accelerate launch run.py \
    --prepared-data="./data/cometinho-prepared" \
    --model-config="cometinho" \
    --no-tqdm \
    --use-adapters \
    --seed=70800 \
    --adapter-config="ia3"