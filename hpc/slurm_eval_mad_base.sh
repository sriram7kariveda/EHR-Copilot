#!/bin/bash
#SBATCH --job-name=eval-mad
#SBATCH --partition=gpuqm
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval-mad-%j.log

set -e
mkdir -p logs
export HF_HOME=/home/018214196/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PATH=/home/018214196/ehr-venv/bin:$PATH
source /home/018214196/ehr-venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/018214196/ehr-copilot
export PYTHONPATH=/home/018214196/ehr-copilot/src:$PYTHONPATH

python scripts/eval_medhallu_detection.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --data_path data/medhallu_eval_2k.jsonl \
    --count 1000 \
    --configs mad_base \
    --output results/eval_mad_base_1k.json
