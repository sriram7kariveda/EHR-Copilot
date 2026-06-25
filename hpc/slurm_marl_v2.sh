#!/bin/bash
#SBATCH --job-name=marl-v2
#SBATCH --partition=gpuqm
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/marl-v2-%j.log

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

echo "MARL v2 (detection-aligned shared reward)"

python scripts/train_marl_v2.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --iterations 3 \
    --k_samples 4 \
    --learning_rate 5e-6 \
    --lora_r 8 \
    --output_dir models/marl-v2
