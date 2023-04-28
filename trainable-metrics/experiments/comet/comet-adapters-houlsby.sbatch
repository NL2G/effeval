#!/bin/bash 
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=3-00:00:00
#SBATCH --exclusive
#SBATCH --job-name="COMET-Training-adapters-houlsby"
#SBATCH --output=./logs/comet/R-%x-%j.out

echo "==="
echo "Starting iteration-1"
echo "==="
WANDB_RUN_GROUP="comet-adapters-houlsby" WANDB_NAME="comet-adapters-houlsby-21291" \
    accelerate launch run.py \
    --prepared-data="./data/comet-prepared" \
    --model-config="comet" \
    --no-tqdm \
    --use-adapters \
    --seed=21291 \
    --adapter-config="houlsby"

echo "==="
echo "Starting iteration-2"
echo "==="
WANDB_RUN_GROUP="comet-adapters-houlsby" WANDB_NAME="comet-adapters-houlsby-2169" \
    accelerate launch run.py \
    --prepared-data="./data/comet-prepared" \
    --model-config="comet" \
    --no-tqdm \
    --use-adapters \
    --seed=2169 \
    --adapter-config="houlsby"

echo "==="
echo "Starting iteration-3"
echo "==="
WANDB_RUN_GROUP="comet-adapters-houlsby" WANDB_NAME="comet-adapters-houlsby-7800" \
    accelerate launch run.py \
    --prepared-data="./data/comet-prepared" \
    --model-config="comet" \
    --no-tqdm \
    --use-adapters \
    --seed=70800 \
    --adapter-config="houlsby"