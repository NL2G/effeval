#!/usr/bin/env bash
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --job-name="GPTScore evalaution"

python score_d2t.py \
    --dataname "BAGEL" \
    --use_demo False \
    --use_ist True \
    --llama_score True \
    --out_dir_name "optScore_based"  \
    --aspect 'quality'