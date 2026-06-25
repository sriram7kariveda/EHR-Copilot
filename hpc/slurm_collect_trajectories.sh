#!/bin/bash
#SBATCH --job-name=collect-traj
#SBATCH --partition=gpuqm
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/collect-traj-%j.log

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

echo "Collecting pipeline trajectories for MARL"
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"

# Both models on one H100 80GB: Qwen 3 8B (~16GB) + Qwen 2.5 3B (~6GB) = ~22GB
python scripts/collect_pipeline_trajectories.py \
    --data_path data/medhallu_eval_2k.jsonl \
    --output_path data/marl_trajectories.jsonl \
    --pipeline_model Qwen/Qwen3-8B \
    --debate_model Qwen/Qwen2.5-3B-Instruct \
    --count 1000
