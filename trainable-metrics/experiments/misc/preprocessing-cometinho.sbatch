#!/bin/bash 
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --time=6:00:00
#SBATCH --job-name="COMETinho-data-preprocessing"
#SBATCH --output=./logs/misc/R-%x-%j.out

python preprocess.py \
    --data-config="cometinho" \
    --encoder-model-name="microsoft/Multilingual-MiniLM-L12-H384" \
    --num-processes=64 \
    --dev-size=0.0001 \
    --seed=42 \
    --max-length=512 \
    --output-dir="./data/cometinho-prepared"

