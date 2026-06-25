#!/bin/bash
#SBATCH --job-name=ehr-setup
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/setup-%j.log

# Setup environment (needs GPU node for torch CUDA verification)
mkdir -p logs
cd /home/018214196/ehr-copilot
bash hpc/setup_env.sh
