#!/bin/bash
#SBATCH --job-name=marl-full
#SBATCH --partition=gpuqm
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/marl-full-%j.log

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

echo "MARL Full Pipeline — Full Run"
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"

# Trial: 50 trajectories, 1 iteration, conservative hyperparameters
python scripts/train_marl_full_pipeline.py \
    --trajectories data/grpo_trajectories/train.jsonl \
    --pipeline_model Qwen/Qwen3-8B \
    --debate_model Qwen/Qwen2.5-3B-Instruct \
    --count 500 \
    --iterations 2 \
    --k_samples 4 \
    --pipeline_lr 1e-6 \
    --debate_lr 5e-6 \
    --pipeline_lora_r 4 \
    --debate_lora_r 8 \
    --output_dir models/marl-full-pipeline-v2
