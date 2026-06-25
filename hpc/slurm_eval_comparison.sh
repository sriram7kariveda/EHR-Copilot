#!/bin/bash
#SBATCH --job-name=eval-compare
#SBATCH --partition=gpuqm
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval-compare-%j.log

# Phase 4: Full evaluation comparison (4 configurations)
#
# Runs on A100 GPU to support:
#   - PubMedBERT embedding inference (base + fine-tuned)
#   - Qwen 3.5 4B for MAD debate evaluation
#
# Usage:
#   sbatch hpc/slurm_eval_comparison.sh              # full MAD eval
#   sbatch hpc/slurm_eval_comparison.sh --no-llm     # embedding metrics only

set -e
mkdir -p logs

export HF_HOME=/home/018214196/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export PATH=/home/018214196/ehr-venv/bin:$PATH
source /home/018214196/ehr-venv/bin/activate

cd /home/018214196/ehr-copilot
export PYTHONPATH=/home/018214196/ehr-copilot/src:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================="
echo "Phase 4: Evaluation Comparison"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date:   $(date)"
echo "============================================="

# Check if --no-llm flag was passed
USE_LLM_FLAG="--use_local_llm"
if [ "$1" = "--no-llm" ]; then
    USE_LLM_FLAG=""
    echo "Mode: Embedding metrics only (no LLM)"
else
    echo "Mode: Full evaluation with MAD debate (LLM enabled)"
fi

python scripts/eval_full_comparison.py \
    --eval_path results/eval_results_10patients_merged.json \
    --gt_path results/ground_truth_eval.json \
    --base_model NeuML/pubmedbert-base-embeddings \
    --finetuned_model models/pubmedbert-ehr-finetuned \
    --top_k 15 \
    --llm_model Qwen/Qwen2.5-3B-Instruct \
    --grpo_verifier_model models/verifier-grpo \
    --grpo_challenger_model models/challenger-grpo \
    --output_path results/comparison_results.json \
    $USE_LLM_FLAG

echo ""
echo "============================================="
echo "Evaluation complete. Results saved to results/comparison_results.json"
echo "============================================="
