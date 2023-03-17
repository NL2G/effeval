#!/bin/bash 
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --job-name="Quantized metrics evaluation @ NLLG"


for ITER in {1..5};
do
    echo "====================";
    echo "Iteration $ITER";
    echo "====================";
    python ./evaluate_metrics.py;
done
