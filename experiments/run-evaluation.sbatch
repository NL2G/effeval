#!/bin/bash 
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --exclusive
#SBATCH --job-name="Batch-size experiments on metrics evaluation @ NLLG"

echo "==="
echo "Starting iteration-1"
echo "==="
BATCH_SIZE=1 python evaluate_metrics.py
BATCH_SIZE=1 python evaluate_metrics.py
BATCH_SIZE=1 python evaluate_metrics.py

echo "==="
echo "Starting iteration-3"
echo "==="
BATCH_SIZE=4 python evaluate_metrics.py
BATCH_SIZE=4 python evaluate_metrics.py
BATCH_SIZE=4 python evaluate_metrics.py

echo "==="
echo "Starting iteration-5"
echo "==="
BATCH_SIZE=16 python evaluate_metrics.py
BATCH_SIZE=16 python evaluate_metrics.py
BATCH_SIZE=16 python evaluate_metrics.py

echo "==="
echo "Starting iteration-7"
echo "==="
BATCH_SIZE=64 python evaluate_metrics.py
BATCH_SIZE=64 python evaluate_metrics.py
BATCH_SIZE=64 python evaluate_metrics.py

