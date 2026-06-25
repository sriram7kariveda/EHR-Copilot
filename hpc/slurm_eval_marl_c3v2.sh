#!/bin/bash
#SBATCH --job-name=eval-c3v2
#SBATCH --partition=gpuqm
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=logs/eval-c3v2-%j.log

set -e
mkdir -p logs results
export HF_HOME=/home/018214196/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PATH=/home/018214196/ehr-venv/bin:$PATH
source /home/018214196/ehr-venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/018214196/ehr-copilot
export PYTHONPATH=/home/018214196/ehr-copilot/src:$PYTHONPATH

echo "1K MedHallu Eval: All configs including MARL C3 v2"
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"

python scripts/eval_marl_c3v2_1k.py \
    --base_model Qwen/Qwen2.5-3B-Instruct \
    --grpo_verifier models/verifier-grpo-v3 \
    --grpo_challenger models/challenger-grpo-v3 \
    --marl_verifier models/marl-c3-v2/best/verifier \
    --marl_challenger models/marl-c3-v2/best/challenger \
    --data_path data/medhallu_eval_2k.jsonl \
    --count 1000 \
    --configs single_critic,mad_base,mad_grpo,mad_marl_c3v2 \
    --output results/eval_marl_c3v2_1k.json
