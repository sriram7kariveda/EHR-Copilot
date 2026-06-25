#!/bin/bash
#SBATCH --job-name=grpo-train
#SBATCH --partition=gpuqm
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/grpo-train-%j.log

set -e
mkdir -p logs

export HF_HOME=/home/018214196/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export PATH=/home/018214196/ehr-venv/bin:$PATH
source /home/018214196/ehr-venv/bin/activate

cd /home/018214196/ehr-copilot
export PYTHONPATH=/home/018214196/ehr-copilot/src:$PYTHONPATH

AGENT=${1:-verifier}
echo "Training GRPO for agent: $AGENT"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/train_grpo.py \
    --agent $AGENT \
    --trajectories data/grpo_trajectories/train.jsonl \
    --eval_trajectories data/grpo_trajectories/eval.jsonl \
    --model Qwen/Qwen2.5-3B-Instruct \
    --output_dir models/${AGENT}-grpo \
    --epochs 1 \
    --k_samples 2 \
    --batch_size 1 \
    --learning_rate 1e-5 \
    --lora_r 8
