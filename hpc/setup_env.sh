#!/bin/bash
# HPC Environment Setup for EHR-Copilot
# Run this ONCE on HPC3 to set up the Python environment.
# IMPORTANT: Run via Slurm (sbatch) or on a compute node, NOT on head node.

set -e

echo "=== Setting up EHR-Copilot environment on HPC3 ==="

# Load modules
module load python3/3.12.12

# Create virtual environment
VENV_DIR="/home/018214196/ehr-venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists at $VENV_DIR"
fi

# Activate
source "$VENV_DIR/bin/activate"

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA 12
echo "Installing PyTorch with CUDA 12..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install training dependencies
echo "Installing training dependencies..."
pip install sentence-transformers
pip install transformers datasets accelerate
pip install trl peft bitsandbytes
pip install sentencepiece protobuf

# Install evaluation dependencies
pip install rouge-score rapidfuzz numpy scipy scikit-learn

echo "=== Setup complete ==="
echo "Activate with: source $VENV_DIR/bin/activate"
python3 --version
pip list | grep -E "torch|sentence|transformers|trl|peft"
