#!/bin/bash
#SBATCH --job-name=vllm-serve
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/vllm-serve-%j.log

set -e
mkdir -p logs

export HF_HOME=/home/018214196/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export PATH=/home/018214196/ehr-venv/bin:$PATH
source /home/018214196/ehr-venv/bin/activate

echo "Starting vLLM server for Qwen 3.5 4B..."
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-4B \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --disable-log-requests
