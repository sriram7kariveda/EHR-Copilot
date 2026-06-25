#!/bin/bash
#SBATCH --job-name=grpo-v2
#SBATCH --partition=gpuqm
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/grpo-v2-%j.log

set -e
mkdir -p logs

export HF_HOME=/home/018214196/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export PATH=/home/018214196/ehr-venv/bin:$PATH
source /home/018214196/ehr-venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/018214196/ehr-copilot
export PYTHONPATH=/home/018214196/ehr-copilot/src:$PYTHONPATH

AGENT=${1:-verifier}
echo "GRPO v2 Training: agent=$AGENT, k=4, QLoRA"
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"

python scripts/train_grpo_v2.py \
    --agent $AGENT \
    --trajectories data/grpo_trajectories/train.jsonl \
    --eval_trajectories data/grpo_trajectories/eval.jsonl \
    --model Qwen/Qwen2.5-3B-Instruct \
    --epochs 1 \
    --k_samples 4 \
    --batch_size 1 \
    --grad_accum 4 \
    --learning_rate 1e-5 \
    --lora_r 8 \
    --eval_every 50 \
    --patience 3
