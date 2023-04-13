#!/bin/bash 
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --job-name="COMET-Training"

echo "==="
echo "Starting training"
echo "==="
accelerate launch --config_file="accelerate.config.yaml" run.py