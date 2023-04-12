#!/bin/bash 
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --exclusive
#SBATCH --job-name="Batch-size experiments on metrics evaluation @ NLLG"


for ITER in {1..3};
do
    echo "====================";
    echo "Iteration $ITER";
    echo "====================";
    python ./evaluate_metrics.py;
done