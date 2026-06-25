# How to Reproduce All Results

## Prerequisites

### 1. SJSU HPC Access
```bash
# Connect VPN
# Open Cisco AnyConnect → connect to vpn.sjsu.edu

# SSH to HPC1 (has internet)
ssh 018214196@coe-hpc1.sjsu.edu

# From HPC1, SSH to HPC3 (compute cluster)
ssh coe-hpc3
```

### 2. Python Environment (already set up)
```bash
# Venv location on HPC3:
source /home/018214196/ehr-venv/bin/activate

# Key packages: torch 2.6.0, transformers 5.4.0, sentence-transformers 5.3.0, peft, accelerate
```

### 3. Model Cache (already downloaded)
All models are at `/home/018214196/.cache/huggingface/hub/`:
- `Qwen/Qwen3-8B` (pipeline LLM)
- `Qwen/Qwen2.5-3B-Instruct` (debate agents)
- `NeuML/pubmedbert-base-embeddings` (retrieval)
- `UTAustin-AIHealth/MedHallu` (dataset)

### 4. Standard Slurm Environment Variables
Every Slurm script needs:
```bash
export HF_HOME=/home/018214196/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PATH=/home/018214196/ehr-venv/bin:$PATH
source /home/018214196/ehr-venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/018214196/ehr-copilot
export PYTHONPATH=/home/018214196/ehr-copilot/src:$PYTHONPATH
```

---

## Step-by-Step Reproduction

### Step 1: Sync Code to HPC
```bash
# From your local machine:
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='.venv' \
  /Users/shas232/ehr-copilot/ \
  018214196@coe-hpc1.sjsu.edu:/home/018214196/ehr-copilot/
```

### Step 2: Embedding Fine-Tuning
```bash
# On HPC3:
cd /home/018214196/ehr-copilot

# Generate triplets (already done, file exists)
# python scripts/generate_embedding_triplets.py --eval_path results/eval_results_10patients_merged.json

# Train (3 min on A100)
sbatch hpc/slurm_finetune_embeddings.sh

# Evaluate
sbatch hpc/slurm_eval_embeddings.sh

# Expected output:
# MRR: 0.623 → 0.914
# NDCG: 0.676 → 0.934
```

### Step 3: Generate Training Data
```bash
# Generate GRPO trajectories from MedHallu (run on HPC1 — needs internet)
ssh coe-hpc1
module load python3/3.11.5
source /home/018214196/tmp-venv/bin/activate
cd /home/018214196/ehr-copilot
python3 scripts/generate_grpo_trajectories.py \
    --output_dir data/grpo_trajectories \
    --medhallu_count 1000 --mednli_count 0

# Generate eval data (also on HPC1)
python3 -c "
import json, random
from datasets import load_dataset
random.seed(142)
labeled = load_dataset('UTAustin-AIHealth/MedHallu', name='pqa_labeled', split='train')
artificial = load_dataset('UTAustin-AIHealth/MedHallu', name='pqa_artificial', split='train')
pairs = []
for ds in [labeled, artificial]:
    indices = list(range(len(ds)))
    random.shuffle(indices)
    for idx in indices:
        if len(pairs) >= 2000: break
        row = ds[idx]
        pairs.append({'question': row['Question'], 'knowledge': row['Knowledge'],
                      'ground_truth': row['Ground Truth'], 'hallucinated': row['Hallucinated Answer'],
                      'difficulty': row.get('Difficulty_Level', 'unknown'),
                      'category': row.get('Category of Hallucination', '')})
with open('data/medhallu_eval_2k.jsonl', 'w') as f:
    for p in pairs:
        f.write(json.dumps(p) + '\n')
print(f'Saved {len(pairs)} pairs')
"
```

### Step 4: GRPO v3 Training (Detection-Aligned Reward)
```bash
# On HPC3 — train both agents in parallel on separate H100s:
sbatch hpc/slurm_grpo_v3.sh verifier    # Job on one H100
sbatch hpc/slurm_grpo_v3.sh challenger  # Job on another H100

# Monitor:
tail -f logs/grpo-v3-*.log

# Expected: Verifier eval reward 0.221 → ~0.705
# Expected: Challenger eval reward 0.132 → ~0.218
# Time: ~4-8 hours each

# Models saved to:
# models/verifier-grpo-v3/
# models/challenger-grpo-v3/
```

### Step 5: MARL Full Pipeline Training
```bash
# Trial first (50 examples, ~2 hours):
# Edit hpc/slurm_marl_full_trial.sh: --count 50 --iterations 1
sbatch hpc/slurm_marl_full_trial.sh

# Check sanity: look for "SANITY CHECK" lines in log
# If 8B outputs look normal → proceed to full run

# Full run (500 examples, 2 iterations, ~12 hours):
# Edit hpc/slurm_marl_full_trial.sh: --count 500 --iterations 2
sbatch hpc/slurm_marl_full_trial.sh

# Models saved to:
# models/marl-full-pipeline-v2/8b/iter_1, iter_2
# models/marl-full-pipeline-v2/3b/iter_1, iter_2
```

### Step 6: Final Evaluation (1K MedHallu pairs, parallel)
```bash
# Run all 3 configs in parallel on 3 H100s:
sbatch hpc/slurm_eval_single_critic.sh  # ~15 min
sbatch hpc/slurm_eval_mad_base.sh       # ~8 hours
sbatch hpc/slurm_eval_mad_grpo.sh       # ~8 hours

# Results saved to:
# results/eval_single_critic_1k.json
# results/eval_mad_base_1k.json
# results/eval_mad_grpo_1k.json

# Expected results:
# Single Critic: F1=0.551 [0.522, 0.579]
# MAD base:      F1=0.642 [0.619, 0.663]
# MAD + GRPO:    F1=0.657 [0.636, 0.678]
```

### Step 7: MARL Full Pipeline Evaluation (TODO)
```bash
# Needs new eval script that loads MARL-trained 8B + 3B adapters
# Use models/marl-full-pipeline-v2/8b/iter_2 and 3b/iter_2
# Run on 1K MedHallu pairs
# Compare with mad_grpo results
```

---

## Quick Commands

### Check running jobs
```bash
ssh 018214196@coe-hpc1.sjsu.edu "ssh coe-hpc3 'squeue -u 018214196'"
```

### Check job output
```bash
ssh 018214196@coe-hpc1.sjsu.edu "ssh coe-hpc3 'tail -20 /home/018214196/ehr-copilot/logs/<log-file>.log'"
```

### Check available GPUs
```bash
ssh 018214196@coe-hpc1.sjsu.edu "ssh coe-hpc3 'sinfo -p gpuqm -o \"%N %G %T\" | grep -E \"idle|mix\"'"
```

### Cancel a job
```bash
ssh 018214196@coe-hpc1.sjsu.edu "ssh coe-hpc3 'scancel <JOB_ID>'"
```

### Sync local changes to HPC
```bash
rsync -avz --exclude='.git' --exclude='__pycache__' \
  /Users/shas232/ehr-copilot/ \
  018214196@coe-hpc1.sjsu.edu:/home/018214196/ehr-copilot/
```

### Download model (run on HPC1 only — has internet)
```bash
ssh 018214196@coe-hpc1.sjsu.edu
source /home/018214196/tmp-venv/bin/activate
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('MODEL_NAME', cache_dir='/home/018214196/.cache/huggingface/hub')
"
```

### Install new pip package
```bash
# Download on HPC1:
ssh 018214196@coe-hpc1.sjsu.edu
source /home/018214196/tmp-venv/bin/activate
pip download -d /home/018214196/pip-cache PACKAGE_NAME

# Install on HPC3 (offline):
ssh coe-hpc3
/home/018214196/ehr-venv/bin/pip install --no-index --find-links=/home/018214196/pip-cache PACKAGE_NAME
```

---

## Trained Models Summary

| Model | Path on HPC | What it is |
|-------|-------------|------------|
| Fine-tuned PubMedBERT | `models/pubmedbert-ehr-finetuned/` | Embedding model (768d) |
| GRPO v3 Verifier | `models/verifier-grpo-v3/` | Best Verifier LoRA adapter |
| GRPO v3 Challenger | `models/challenger-grpo-v3/` | Best Challenger LoRA adapter |
| MARL Full 8B | `models/marl-full-pipeline-v2/8b/iter_2/` | Pipeline LoRA (Triage+CRAG) |
| MARL Full 3B | `models/marl-full-pipeline-v2/3b/iter_2/` | Debate LoRA (Verifier+Challenger) |
| MARL v1 (didn't improve) | `models/marl/final/` | Binary reward, for reference only |

## Data Files Summary

| File | Location | Size | What it is |
|------|----------|------|------------|
| Embedding triplets | `data/embedding_triplets.jsonl` | 760 entries | Query-chunk training pairs |
| GRPO trajectories (train) | `data/grpo_trajectories/train.jsonl` | 1800 entries | MedHallu for GRPO training |
| GRPO trajectories (eval) | `data/grpo_trajectories/eval.jsonl` | 200 entries | MedHallu for GRPO eval |
| MedHallu eval | `data/medhallu_eval_2k.jsonl` | 2000 entries | Pre-generated eval pairs |
| MARL trajectories | `data/marl_trajectories.jsonl` | 2000 entries | Full pipeline trajectories |
| Pipeline eval results | `results/eval_results_10patients_merged.json` | 80 queries | Source for embedding triplets |
| Ground truth eval | `results/ground_truth_eval.json` | 78 queries | Entity F1, halluc rate baselines |

---

## Troubleshooting

### "TRANSFORMERS_OFFLINE" error on compute nodes
Compute nodes have no internet. Always set `TRANSFORMERS_OFFLINE=1` and `HF_HUB_OFFLINE=1`. Pre-download models on HPC1.

### "bitsandbytes" CUDA error
bitsandbytes doesn't work on this HPC. Use full bf16 on H100 instead. If you see this error, run: `/home/018214196/ehr-venv/bin/pip uninstall -y bitsandbytes`

### OOM on A100 (40GB)
- Use H100 (80GB) instead: `--gres=gpu:h100:1`
- Or reduce k_samples from 4 to 2
- Or reduce max_new_tokens
- Or enable gradient checkpointing: `model.gradient_checkpointing_enable()`

### SSH connection drops
VPN (Cisco AnyConnect) disconnects frequently. Jobs keep running on HPC regardless. Just reconnect and check logs.

### Model not found on compute node
Model was cached with different Python version. Re-download on HPC1: `source tmp-venv/bin/activate && python3 -c "from huggingface_hub import snapshot_download; snapshot_download('MODEL_NAME', cache_dir='...')"`
Also cache tokenizer: `from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('MODEL_NAME', cache_dir='...')`
