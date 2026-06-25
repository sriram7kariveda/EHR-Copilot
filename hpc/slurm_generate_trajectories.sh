#!/bin/bash
#SBATCH --job-name=gen-traj
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/gen-traj-%j.log

set -e
mkdir -p logs

export HF_HOME=/home/018214196/.cache/huggingface
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PATH=/home/018214196/ehr-venv/bin:$PATH
source /home/018214196/ehr-venv/bin/activate

cd /home/018214196/ehr-copilot
export PYTHONPATH=/home/018214196/ehr-copilot/src:$PYTHONPATH

python scripts/generate_grpo_trajectories.py \
    --output_dir data/grpo_trajectories \
    --medhallu_count 1000 \
    --mednli_count 5000
