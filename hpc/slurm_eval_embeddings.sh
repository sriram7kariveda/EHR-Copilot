#!/bin/bash
#SBATCH --job-name=embed-eval
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/embed-eval-%j.log

set -e
mkdir -p logs

export HF_HOME=/home/018214196/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export PATH=/home/018214196/ehr-venv/bin:$PATH
source /home/018214196/ehr-venv/bin/activate

cd /home/018214196/ehr-copilot

python scripts/eval_embeddings.py \
    --eval_path results/eval_results_10patients_merged.json \
    --base_model NeuML/pubmedbert-base-embeddings \
    --finetuned_model models/pubmedbert-ehr-finetuned \
    --top_k 15
