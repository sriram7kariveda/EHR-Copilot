#!/bin/bash
#SBATCH --job-name=eval-sc
#SBATCH --partition=gpuqm
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval-sc-%j.log

set -e
mkdir -p logs
export HF_HOME=/home/018214196/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PATH=/home/018214196/ehr-venv/bin:$PATH
source /home/018214196/ehr-venv/bin/activate
cd /home/018214196/ehr-copilot
export PYTHONPATH=/home/018214196/ehr-copilot/src:$PYTHONPATH

python scripts/eval_medhallu_detection.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --data_path data/medhallu_eval_2k.jsonl \
    --count 1000 \
    --configs single_critic \
    --output results/eval_single_critic_1k.json
