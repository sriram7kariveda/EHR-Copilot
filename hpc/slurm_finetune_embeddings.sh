#!/bin/bash
#SBATCH --job-name=embed-ft
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/embed-ft-%j.log

set -e
mkdir -p logs

# Activate environment and set HF cache (no internet on compute nodes)
export HF_HOME=/home/018214196/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PATH=/home/018214196/ehr-venv/bin:$PATH
source /home/018214196/ehr-venv/bin/activate
echo "Python: $(python3 --version)"
python3 -c 'import torch; print("torch:", torch.__version__, "CUDA:", torch.cuda.is_available())'
python3 -c 'import sentence_transformers; print("ST:", sentence_transformers.__version__)'
cd /home/018214196/ehr-copilot

echo "=== Step 1: Generate triplets ==="
python scripts/generate_embedding_triplets.py \
    --eval_path results/eval_results_10patients_merged.json \
    --output_path data/embedding_triplets.jsonl \
    --negatives_per_positive 5

echo "=== Step 2: Fine-tune PubMedBERT ==="
python scripts/finetune_embeddings.py \
    --triplets_path data/embedding_triplets.jsonl \
    --base_model NeuML/pubmedbert-base-embeddings \
    --output_dir models/pubmedbert-ehr-finetuned \
    --epochs 5 \
    --batch_size 16 \
    --learning_rate 2e-5

echo "=== Done ==="
nvidia-smi
