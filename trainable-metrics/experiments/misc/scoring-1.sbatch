#!/bin/bash 
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name="COMET-scoring-1"
#SBATCH --output=./logs/misc/R-%x-%j.out

python pseudo-label.py --input="./data/10m" --gpus=1 --shard=1 --batch_size=256 --out="./data/cometinho-1.csv.gz"
