#!/bin/bash
#SBATCH --job-name=grpo-v3
#SBATCH --partition=gpuqm
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/grpo-v3-%j.log

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

AGENT=${1:-verifier}
echo "GRPO v3 (detection-aligned): agent=$AGENT"

python scripts/train_grpo_v3.py \
    --agent $AGENT \
    --model Qwen/Qwen2.5-3B-Instruct \
    --k_samples 4 --grad_accum 4 \
    --learning_rate 1e-5 --lora_r 8 \
    --eval_every 50 --patience 5
