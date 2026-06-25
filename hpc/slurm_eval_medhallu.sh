#!/bin/bash
#SBATCH --job-name=medhallu-eval
#SBATCH --partition=gpuqm
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/medhallu-eval-%j.log

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

echo "MedHallu Detection Eval"
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"

# Run Track 2: Hallucination detection on 2K pairs (start smaller, scale up if working)
python scripts/eval_medhallu_detection.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --data_path data/medhallu_eval_2k.jsonl \
    --count 1000 \
    --configs single_critic,mad_base,mad_grpo \
    --grpo_verifier models/verifier-grpo-v3 \
    --grpo_challenger models/challenger-grpo-v3 \
    --output results/medhallu_detection_results.json \
    --seed 142
