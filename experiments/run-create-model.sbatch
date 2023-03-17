#!/bin/bash 
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --job-name="Create Quantized Models"


python ./create_quantized_models.py
