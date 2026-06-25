#!/bin/bash
#SBATCH --job-name=marl-c3v2
#SBATCH --partition=gpuqm
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --output=logs/marl-c3v2-%j.log

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

echo "MARL C3 v2: Separate LoRA + IBR + Warm-start from GRPO v3"
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"

# Need 80GB mem for TWO 3B models with separate LoRA adapters (~12GB total)
python scripts/train_marl_c3_v2.py \
    --base_model Qwen/Qwen2.5-3B-Instruct \
    --verifier_adapter models/verifier-grpo-v3 \
    --challenger_adapter models/challenger-grpo-v3 \
    --count 1800 \
    --iterations 3 \
    --k_samples 4 \
    --lr 3e-6 \
    --output_dir models/marl-c3-v2
