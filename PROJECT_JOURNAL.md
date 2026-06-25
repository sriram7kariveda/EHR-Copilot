# EHR-Copilot: Complete Project Journal

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Architecture Evolution](#2-architecture-evolution)
3. [Phase 1: Embedding Fine-Tuning](#3-phase-1-embedding-fine-tuning)
4. [Phase 2: Multi-Agent Debate (MAD)](#4-phase-2-multi-agent-debate-mad)
5. [Phase 3: GRPO Training](#5-phase-3-grpo-training)
6. [Phase 4: MARL (Multi-Agent RL)](#6-phase-4-marl-multi-agent-rl)
7. [Phase 5: Final Evaluation](#7-phase-5-final-evaluation)
8. [Infrastructure & HPC Setup](#8-infrastructure--hpc-setup)
9. [What Worked & What Didn't](#9-what-worked--what-didnt)
10. [File Manifest](#10-file-manifest)
11. [How to Reproduce](#11-how-to-reproduce)
12. [Remaining TODO](#12-remaining-todo)

---

## 1. Project Overview

**Goal:** Build a multi-agent RAG pipeline for clinical EHR question answering, with MARL (Multi-Agent Reinforcement Learning) to train agents to cooperate.

**Professor's request:** Implement RL/MARL in the critic agent of the EHR-Copilot pipeline. Evolved into full-pipeline MARL across 11 agents.

**Final architecture:** 11-agent tree pipeline with:
- Query-type-aware routing (7 types, 5 branches)
- Fine-tuned PubMedBERT embeddings for retrieval
- Multi-Agent Debate (MAD) replacing single Critic (3 debate agents)
- GRPO + MARL training for debate agents and pipeline agents
- Deterministic entity verification

**Key results (1K MedHallu pairs):**
| Config | Detection F1 | Precision | Recall |
|--------|-------------|-----------|--------|
| Single Critic (baseline) | 0.551 | 0.679 | 0.463 |
| MAD Debate | 0.642 | 0.543 | 0.784 |
| MAD + GRPO v3 | 0.657 | 0.533 | 0.855 |
| MAD + MARL Full Pipeline | pending eval | — | — |

**Embedding fine-tuning:** MRR 0.623→0.914 (+46.7%), NDCG 0.676→0.934 (+38.2%)

**Baseline pipeline:** Entity F1=0.616, Hallucination Rate=8.2%, beats GPT-5 (0.329), Sonnet 4.5 (0.529), Gemini 3 Pro (0.234)

---

## 2. Architecture Evolution

### Original Pipeline (before this work)
```
Query → Router → Retrieval → Reasoning → Temporal/Numeric Validators → Single Critic → Entity Verifier → Answer
```
- 7 agents, sequential
- Single Critic = one LLM call that says APPROVE/REVISE/ABSTAIN
- No RL training, no query-type-aware processing

### Intermediate Pipeline (tree routing added)
```
Query → Triage → Decomposer → Retrieval (branch-specific) → CRAG Evaluator →
  [5 branches with different prompts] → Reasoning → Validators → Critic → Entity Verifier
```
- Added: Query Decomposer, CRAG Evaluator, Chunk Filter, 5 specialized reasoning prompts
- Why: Professor wanted tree architecture. Router classifies query type, each type gets different retrieval strategy, different CoT prompt, different validators.
- Key files: `chunk_filter.py`, `query_decomposer.py`, `retrieval_evaluator.py`, 5 `reasoning_*.txt` prompts

### Final Pipeline (MAD + MARL)
```
Query → Triage → Decomposer → Retrieval (fine-tuned embeddings) → CRAG →
  [5 branches] → Reasoning → Validators →
  MAD Debate (Verifier ↔ Challenger → Judge) → Entity Verifier → Answer
```
- Replaced single Critic with 3-agent debate
- Added embedding fine-tuning
- Added GRPO/MARL training for debate agents + pipeline agents
- 11 agents total

### Why each agent exists

| Agent | Why added | What it does |
|-------|----------|-------------|
| Triage | Professor wanted tree routing | Classifies into 7 query types, controls 4 downstream dimensions |
| Decomposer | Complex queries need multiple retrievals | Breaks COMPARISON/SUMMARY into 2-4 sub-queries |
| Retrieval + Chunk Filter | Different query types need different evidence | Branch-specific top_k, section boosting, date sorting |
| CRAG Evaluator | Prevent reasoning on bad evidence | Checks if retrieval is sufficient before reasoning |
| Reasoning (5 prompts) | One-size-fits-all prompt is suboptimal | Timeline CoT for temporal, calculation CoT for numeric, etc. |
| Temporal Validator | Date errors in clinical data are dangerous | Regex + LLM date consistency checking |
| Numeric Validator | Wrong lab values are dangerous | UCUM unit checking, calculation verification |
| Verifier (MAD) | Single Critic misses hallucinations | Checks each claim against evidence with confidence scores |
| Challenger (MAD) | Verifier alone is too generous | Adversarially challenges SUPPORTED claims with medical queries |
| Judge (MAD) | Need unbiased final decision | Blind scoring (doesn't see confidence) prevents anchoring |
| Entity Verifier | Final safety net | Deterministic, zero LLM cost, removes ungrounded entities |

---

## 3. Phase 1: Embedding Fine-Tuning

### Problem
PubMedBERT (NeuML/pubmedbert-base-embeddings, 768d) was trained on PubMed research abstracts, not clinical notes (discharge summaries, progress notes, lab reports). Relevant chunks were ranking 3rd-5th instead of 1st.

### Data
- Source: `results/eval_results_10patients_merged.json` (78 pipeline evaluations)
- Positive pairs: query → chunks that were CITED in the answer (actually useful)
- Negative pairs: query → chunks that were RETRIEVED but NOT CITED (noise)
- 152 positive pairs, 922 hard negatives
- Sampled 5 negatives per positive → **760 triplets**
- Script: `scripts/generate_embedding_triplets.py`

### Training
- Model: PubMedBERT (110M params, all fine-tuned — not LoRA, model is small enough)
- Loss: **MultipleNegativesRankingLoss** (InfoNCE)
  - First tried TripletLoss → accuracy DROPPED to 6.6% (wrong loss for this task)
  - MNRL uses in-batch negatives → much better gradient signal
- Config: batch_size=16, epochs=5, lr=2e-5, warmup=10%
- Infrastructure: SJSU HPC3, A100 GPU, ~3 minutes training
- Script: `scripts/finetune_embeddings.py`
- Slurm: `hpc/slurm_finetune_embeddings.sh`

### Results
| Metric | Base PubMedBERT | Fine-tuned | Improvement |
|--------|----------------|------------|-------------|
| Recall@15 | 1.000 | 1.000 | same |
| MRR | 0.623 | **0.914** | **+46.7%** |
| NDCG@15 | 0.676 | **0.934** | **+38.2%** |

- Eval script: `scripts/eval_embeddings.py`
- Model saved: `models/pubmedbert-ehr-finetuned` on HPC

### Key decisions
- Used MNRL instead of TripletLoss because TripletLoss needs well-calibrated margins; our positives/negatives were too semantically similar
- 760 triplets is small but sufficient for a 110M model (embedding models need less data than LLMs)

---

## 4. Phase 2: Multi-Agent Debate (MAD)

### Problem
Single Critic agent (one LLM call) had limited hallucination detection. It sees the answer once and decides — no adversarial checking.

### Inspiration
- `guardrails-enterprise` repo (github.com/vineeth917/guardrails-enterprise)
- Their MAD architecture: Agent A (verifier) + Agent B (challenger) + Judge (blind)
- Key ideas borrowed: True→Skeptic rule, Brier score reward, partially blind judge, material vs contextual claims

### Architecture
```
Answer → Claim Extractor → list of claims (material vs contextual)
  → Verifier: checks each claim vs evidence → SUPPORTED/NOT_SUPPORTED/PARTIAL + confidence
  → Challenger: challenges SUPPORTED claims with 5 medical query types
      - CONTRAINDICATION, DOSAGE_CHECK, INTERACTION, GUIDELINE_CURRENCY, GAP_FINDING
  → Verifier revises confidence (NOT claim text)
  → [Repeat for 2 cycles]
  → Judge (blind — doesn't see confidence): scores each claim 1.0/0.5/0.0
  → Routing: material_min ≥ 0.9 → APPROVED, material_min = 0.0 → ABSTAINED, else REVISED
```

### Files
- `src/ehr_copilot/agents/mad/__init__.py`
- `src/ehr_copilot/agents/mad/claim_extractor.py` — decomposes answer into atomic claims
- `src/ehr_copilot/agents/mad/verifier.py` — Agent A
- `src/ehr_copilot/agents/mad/challenger.py` — Agent B
- `src/ehr_copilot/agents/mad/judge.py` — blind judge
- `src/ehr_copilot/agents/mad/debate_engine.py` — orchestrator
- `src/ehr_copilot/agents/mad/storage.py` — SQLite trajectory logging for GRPO

### LLM
- Model: Qwen 2.5 3B Instruct (loaded locally on HPC A100/H100)
- Why not Qwen 3 8B: debate needs multiple LLM calls per answer (5-6), 3B is faster and fits alongside other models
- Client: `src/ehr_copilot/llm/local_client.py` (loads model directly with transformers, no vLLM — vLLM wouldn't install on HPC)

### Claim Extractor Issues
- Initial version extracted only 1 claim from long answers (12 diagnoses → 1 claim)
- Fixed with: (a) chunking long answers at list boundaries, (b) improved prompt requiring each diagnosis as separate claim, (c) regex fallback that splits numbered/bullet lists
- For MedHallu (short answers, 1-3 sentences): entire answer becomes one claim

### Routing Decision Evolution
1. **v1 (original):** aggregate ≥ 0.8 → APPROVED, any material 0.0 → ABSTAINED, else REVISED
   - Result: F1=0.628, Precision=0.521, Recall=0.790 ← **BEST VERSION**
2. **v2 (three-zone):** score < 0.3 → flag, 0.3-0.7 → only flag if Challenger found issues, > 0.7 → approve
   - Result: F1=0.442 — WORSE. Zone 2 approved everything because Challenger rarely fires
3. **v3 (zone + material check):** zone 2 flags if Challenger OR material claims < 0.5
   - Result: F1=0.473 — still worse than v1
4. **Reverted to v1** for final eval — strict prompts with simple threshold worked best

### Prompt Evolution
- Original Verifier prompt: "determine if evidence SUPPORTS it" → too generous, approved paraphrased hallucinations
- Strict Verifier prompt: "STRICT... only SUPPORTED if evidence EXPLICITLY states the same fact" → much better recall
- Original Judge prompt: "Be strict with MATERIAL, lenient with CONTEXTUAL" → too lenient
- Strict Judge prompt: "STRICT... score 0.0 if evidence does NOT mention this fact" → better detection

### Results (200 pairs, initial eval)
| Config | F1 | Precision | Recall |
|--------|-----|-----------|--------|
| Single Critic | 0.531 | 0.647 | 0.450 |
| MAD Debate | **0.628** | 0.521 | **0.790** |

---

## 5. Phase 3: GRPO Training

### What is GRPO
Group Relative Policy Optimization (from DeepSeek-R1). For each input:
1. Generate K completions from the model
2. Score each with reward function
3. Compute group-relative advantage: A_i = (r_i - mean(r)) / std(r)
4. Policy gradient update weighted by advantage

No reward model needed (unlike PPO). No preference pairs needed (unlike DPO).

### Training Data
- Source: MedHallu dataset (UTAustin-AIHealth/MedHallu on HuggingFace)
- 1000 labeled pairs (pqa_labeled) + 9000 artificial pairs (pqa_artificial)
- Generated 2000 trajectories: 1000 hallucinated + 1000 ground truth answers
- Script: `scripts/generate_grpo_trajectories.py`
- Pre-generated on HPC1 (has internet): `data/grpo_trajectories/train.jsonl` (1800), `eval.jsonl` (200)

### GRPO v1 (k=2, Brier reward)
- **k=2 samples** (OOM with k=4 on A100 40GB with Qwen 3.5 4B)
- **Brier reward** for Verifier: `2 * confidence * judge_score - confidence²` (rewards calibrated confidence)
- **Challenge precision** for Challenger: +1 if challenged wrong claim, -1 if gaslighting
- Model: Qwen 3.5 4B → OOM. Switched to **Qwen 2.5 3B Instruct**
- Training: 1800 trajectories, batch=1, grad_accum=8
- Result: Verifier reward 0.038→0.220 (5.8x), Challenger 0.090→0.180 (2x)
- **Problem:** Loss=0.0 on most steps because k=2 means both samples often get same reward → zero gradient

### GRPO v2 (k=4, Brier reward, H100)
- Moved to **H100 (80GB)** to fit k=4
- k=4 gives more diverse rewards → better gradient signal
- Added: gradient checkpointing, eval every 50 steps, early stopping (patience=3), CSV logging
- Fixed: separate generation phase (model.eval()) from gradient phase (model.train()) — was slow because gradient checkpointing ran during generation
- max_new_tokens reduced from 256 to 128
- Result: Verifier 0.038→0.220, Challenger 0.090→0.180 (similar to v1)
- **Unique_k metric:** tracked how many of k=4 samples got different rewards. k=4 showed Unique_k=3-4 much more often than k=2

### GRPO v3 (k=4, Detection-Aligned Reward) ← **KEY CHANGE**
- **Problem:** v1/v2 trained with Brier reward (confidence calibration) but evaluated on detection F1. Training metric ≠ eval metric.
- **Research insight:** MMOA-RAG, DeepSeek-R1, HuatuoGPT-o1 all train on the SAME metric they evaluate on.
- **Fix:** Changed reward to directly measure detection accuracy:
  - Hallucinated answer correctly flagged = +1.0
  - Hallucinated answer missed = -1.0
  - Good answer correctly approved = +1.0
  - Good answer wrongly flagged = -0.5 (asymmetric — missing hallucination worse than false alarm)
  - Confidence bonus: high confidence when correct = +0.3
  - Format bonus: valid JSON = +0.1

- **Results:**
  - Verifier: 0.221 → **0.705** (+219% improvement!)
  - Challenger: 0.132 → **0.218** (+65%)

### Training Files
| Version | Script | Slurm | Key change |
|---------|--------|-------|------------|
| v1 | `scripts/train_grpo.py` | `hpc/slurm_train_grpo.sh` | k=2, Brier reward, A100 |
| v2 | `scripts/train_grpo_v2.py` | `hpc/slurm_train_grpo_v2.sh` | k=4, early stopping, H100 |
| v3 | `scripts/train_grpo_v3.py` | `hpc/slurm_grpo_v3.sh` | Detection-aligned reward |

### DPO Attempt (before GRPO)
- Initially tried DPO on Colab Pro (T4 16GB) with Qwen 3.5 4B
- v2 training completed (195 steps) but model COLLAPSED — output was `{{{{{...` gibberish
- Cause: 3 epochs, lr=5e-5, beta=0.1 on only 570 pairs → extreme overfit
- Created v5 notebook with anti-overfit settings but pivoted to GRPO instead
- Files: `/Desktop/RL/critic_dpo_v1-v5.ipynb`, `scripts/train_critic_dpo.py`, `scripts/generate_dpo_pairs.py`

---

## 6. Phase 4: MARL (Multi-Agent RL)

### What is MARL vs Independent GRPO
- **Independent GRPO:** Each agent trains alone with its own reward. Verifier doesn't know about Challenger.
- **MARL:** Agents share a reward signal and learn to cooperate. Both get rewarded/penalized based on their JOINT performance.

### MARL v1 (Binary Shared Reward)
- Reward: binary correct/incorrect (both agents correct = +1, one correct = 0, both wrong = -1)
- 3 iterations, round-robin (train verifier → train challenger)
- 1800 trajectories from MedHallu
- **Result: 49% accuracy, stuck across all 3 iterations — no improvement**
- **Why it failed:** Binary reward gives no gradient signal. With k=2, both samples often get same binary reward → zero advantage.
- Script: `scripts/train_marl.py`, Slurm: `hpc/slurm_train_marl.sh`

### MARL v2 (Detection-Aligned Shared Reward)
- Same continuous reward as GRPO v3, but applied to BOTH agents jointly
- Agreement bonus: if both agents agree, extra +0.3 reward
- Round-robin training
- **Result: Cancelled — too slow.** Each trajectory needs both agents' outputs → 2x compute. Step 10 reward was 0.58 (trending up from 0.43 baseline) but speed was ~40 min per 10 steps.
- Script: `scripts/train_marl_v2.py`, Slurm: `hpc/slurm_marl_v2.sh`

### MARL Full Pipeline (Option 2 — both 8B and 3B)
- **The professor's request:** MARL across the WHOLE pipeline, not just MAD
- Put LoRA on Qwen 3 8B (affects Triage, CRAG, Reasoning) + LoRA on Qwen 2.5 3B (affects Verifier, Challenger)
- Shared reward = detection accuracy (same for all agents)
- Round-robin: train 8B (3B frozen) → sanity check → train 3B (8B frozen) → sanity check

**De-risking:**
- Trial run first: 50 trajectories, 1 iteration → **passed, 8B didn't degrade**
- Conservative hyperparameters for 8B: LoRA r=4 (tiny), lr=1e-6 (10x lower than 3B)
- Sanity check: compare model outputs before/after training, auto-stop if gibberish detected

**Full run:** 500 trajectories, 2 iterations on H100
- 8B iter 1: 100 steps, reward -0.204
- 8B iter 2: 100 steps, reward -0.200
- 3B iter 1: 100 steps, reward 0.034
- 3B iter 2: 100 steps, reward ~pending

**Results:** Training complete. Models saved at `models/marl-full-pipeline-v2/`. Eval pending.

**Script:** `scripts/train_marl_full_pipeline.py`, Slurm: `hpc/slurm_marl_full_trial.sh`

---

## 7. Phase 5: Final Evaluation

### Evaluation Design (following 2025-26 paper standards)

**Track 1: Retrieval Quality**
- 72 queries (10 MIMIC patients)
- Metrics: MRR, NDCG@15, Recall@15
- Script: `scripts/eval_embeddings.py`
- Slurm: `hpc/slurm_eval_embeddings.sh`

**Track 2: Hallucination Detection (primary)**
- 1000 MedHallu pairs (500 hallucinated + 500 ground truth)
- Pre-generated data: `data/medhallu_eval_2k.jsonl`
- Metrics: Detection F1, Precision, Recall, Accuracy, 95% Bootstrap CI
- 4 configs run in parallel on 3 H100s
- Script: `scripts/eval_medhallu_detection.py`
- Slurm: `hpc/slurm_eval_single_critic.sh`, `slurm_eval_mad_base.sh`, `slurm_eval_mad_grpo.sh`

### Final Results (1K MedHallu pairs)

| Config | F1 | 95% CI | Precision | Recall | Accuracy |
|--------|-----|--------|-----------|--------|----------|
| Single Critic | 0.551 | [0.522, 0.579] | 0.679 | 0.463 | 0.622 |
| MAD Debate (base) | 0.642 | [0.619, 0.663] | 0.543 | 0.784 | 0.562 |
| MAD + GRPO v3 | **0.657** | [0.636, 0.678] | 0.533 | **0.855** | 0.553 |
| MAD + MARL Full | pending | — | — | — | — |

**Retrieval Results (72 queries):**

| Model | MRR | NDCG@15 |
|-------|-----|---------|
| Base PubMedBERT | 0.623 | 0.676 |
| Fine-tuned | **0.914** | **0.934** |

**Baseline Pipeline (78 queries, vs other models):**

| Model | Entity F1 | Precision | Halluc Rate |
|-------|-----------|-----------|-------------|
| EHR-Copilot (Ours) | **0.616** | **0.904** | **8.2%** |
| Claude Sonnet 4.5 | 0.529 | 0.761 | 21.0% |
| Claude Haiku 3.5 | 0.480 | 0.794 | 13.4% |
| GPT-5 | 0.329 | 0.555 | 13.1% |
| Gemini 3 Pro | 0.234 | 0.626 | 23.1% |

### Statistical Significance
- All confidence intervals computed with 10,000 bootstrap resamples
- Single Critic CI [0.522, 0.579] vs MAD CI [0.619, 0.663] — **non-overlapping = significant**
- MAD CI [0.619, 0.663] vs GRPO CI [0.636, 0.678] — overlapping but F1 higher

---

## 8. Infrastructure & HPC Setup

### SJSU COE HPC3
- **Access:** SSH hop: local → coe-hpc1.sjsu.edu → coe-hpc3 (via SSH key)
- **VPN:** Cisco AnyConnect to vpn.sjsu.edu required
- **Username:** 018214196
- **Guides:** https://coe-hpc2.sjsu.edu/hpc/coe/slurm_user_guide.html, https://www.sjsu.edu/cmpe/resources/hpc.php
- **NEVER run jobs on head node — always use Slurm (sbatch)**

### GPU Nodes
| Node | GPU | VRAM | Used for |
|------|-----|------|----------|
| cs001-004 | A100 × 4 | 40GB each | GRPO v1, embedding fine-tuning |
| g16 | H100 | 80GB | MAD eval, MARL |
| g18-19 | H100 | 80GB | GRPO v2/v3, MARL full pipeline |

### Python Environment
- **HPC1 (has internet):** `/home/018214196/tmp-venv/` — for downloading packages and models
- **HPC3 (no internet):** `/home/018214196/ehr-venv/` — for training/eval. Python 3.11.7
- Packages installed via: download on HPC1 → `pip download -d pip-cache` → install on HPC3 with `--no-index --find-links`
- Key packages: torch 2.6.0+cu124, transformers 5.4.0, sentence-transformers 5.3.0, peft, accelerate, datasets
- **bitsandbytes:** Wouldn't work on HPC3 (CUDA library issues). Removed. QLoRA not used — full bf16 on H100 instead.
- **vLLM:** Wouldn't install (build dependencies). Used `local_client.py` (direct transformers inference) instead.

### Model Cache
All models pre-downloaded on HPC1 to `/home/018214196/.cache/huggingface/hub/` (shared /home):
- Qwen/Qwen3-8B (~16GB)
- Qwen/Qwen3.5-4B (~8GB)
- Qwen/Qwen2.5-3B-Instruct (~6GB)
- NeuML/pubmedbert-base-embeddings (~440MB)
- MedHallu dataset
- Compute nodes use `TRANSFORMERS_OFFLINE=1` and `HF_HUB_OFFLINE=1`

### Slurm Job Patterns
- Always set: `export HF_HOME=..., TRANSFORMERS_OFFLINE=1, HF_HUB_OFFLINE=1, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- Always: `export PYTHONPATH=.../src:$PYTHONPATH`
- Partitions: `gpuqs` (2-day limit), `gpuqm` (7-day limit)
- Request H100: `--gres=gpu:h100:1`

---

## 9. What Worked & What Didn't

### What Worked
1. **Embedding fine-tuning with MNRL** — huge improvement (MRR +47%), small data (760 triplets), fast training (3 min)
2. **Strict verification prompts** — changing "determine if supported" to "STRICT, only if EXPLICITLY stated" dramatically improved recall
3. **MAD debate architecture** — F1 0.551→0.642 (+16.5%) just from architecture, no training
4. **Detection-aligned reward (GRPO v3)** — training on same metric as eval. Verifier reward jumped 0.221→0.705
5. **k=4 vs k=2 for GRPO** — much better gradient signal. k=2 had Loss=0.0 most steps
6. **Early stopping** — saved compute, prevented overfit
7. **Sanity checks for 8B training** — auto-detected if model degraded

### What Didn't Work
1. **DPO on Colab (v2)** — model collapsed (loss 0.0096→0.00004, output: `{{{{{`). Too aggressive (3 epochs, lr=5e-5, beta=0.1 on 570 pairs)
2. **TripletLoss for embeddings** — accuracy dropped to 6.6%. MNRL fixed it
3. **Three-zone routing** — F1 dropped from 0.628 to 0.442. Uncertain zone approved too many hallucinated answers
4. **MARL v1 (binary reward)** — stuck at 49% for 3 iterations. Binary +1/-1 gives zero gradient when k=2 samples match
5. **MARL v2 (detection-aligned but sequential)** — too slow. Each trajectory needs both agents → 2x compute
6. **bitsandbytes on HPC** — CUDA setup failed. Removed it, used full bf16 on H100 instead
7. **vLLM on HPC** — build dependencies failed. Used direct transformers loading instead
8. **QLoRA for GRPO** — transformers 5.4 requires bitsandbytes ≥0.46 which doesn't exist for the HPC platform
9. **Brier reward for GRPO** — trained confidence calibration but didn't improve detection F1 (different metric)

### Key Lessons
1. **Train on what you measure.** GRPO v1/v2 (Brier reward) improved confidence but not detection. GRPO v3 (detection reward) improved detection. Always align training reward with eval metric.
2. **k=4 >> k=2 for GRPO.** With k=2, both samples often get identical rewards → zero gradient. k=4 gives diverse rewards.
3. **Conservative 8B training.** LoRA r=4 + lr=1e-6 for 8B is safe. r=8 + lr=1e-5 risks degradation on small data.
4. **HPC offline setup is painful.** Pre-download everything on the internet-connected node. Always test with a small job before scaling.
5. **Strict prompts > clever routing.** Simple "be STRICT" in the prompt beat sophisticated three-zone routing logic.

---

## 10. File Manifest

### New Agent Code
| File | Description |
|------|-------------|
| `src/ehr_copilot/agents/mad/__init__.py` | MAD module exports |
| `src/ehr_copilot/agents/mad/claim_extractor.py` | Decomposes answers into atomic claims |
| `src/ehr_copilot/agents/mad/verifier.py` | Agent A — verifies claims vs evidence |
| `src/ehr_copilot/agents/mad/challenger.py` | Agent B — adversarial medical challenges |
| `src/ehr_copilot/agents/mad/judge.py` | Blind judge — scores without seeing confidence |
| `src/ehr_copilot/agents/mad/debate_engine.py` | Orchestrates 2-cycle debate |
| `src/ehr_copilot/agents/mad/storage.py` | SQLite trajectory logging |
| `src/ehr_copilot/agents/chunk_filter.py` | Branch-specific post-retrieval filtering |
| `src/ehr_copilot/agents/query_decomposer.py` | Complex query → sub-queries |
| `src/ehr_copilot/agents/retrieval_evaluator.py` | CRAG-style sufficiency check |
| `src/ehr_copilot/llm/local_client.py` | Local LLM inference (transformers) |
| `src/ehr_copilot/llm/vllm_client.py` | vLLM client (not used — wouldn't install) |

### Prompt Templates
| File | Description |
|------|-------------|
| `src/ehr_copilot/agents/prompts/reasoning_temporal.txt` | Timeline-focused CoT |
| `src/ehr_copilot/agents/prompts/reasoning_numeric.txt` | Calculation-focused CoT |
| `src/ehr_copilot/agents/prompts/reasoning_medication.txt` | Drug-aware CoT |
| `src/ehr_copilot/agents/prompts/reasoning_summary.txt` | Category-organized CoT |
| `src/ehr_copilot/agents/prompts/reasoning_comparison.txt` | Side-by-side comparison CoT |

### Training Scripts
| File | Description |
|------|-------------|
| `scripts/generate_embedding_triplets.py` | Extract query-chunk triplets from eval data |
| `scripts/finetune_embeddings.py` | PubMedBERT fine-tuning with MNRL |
| `scripts/generate_grpo_trajectories.py` | Generate MedHallu trajectories for GRPO |
| `scripts/generate_dpo_pairs.py` | Generate DPO preference pairs (original, pre-GRPO) |
| `scripts/train_critic_dpo.py` | DPO training (original, pre-GRPO) |
| `scripts/train_grpo.py` | GRPO v1 (k=2, Brier reward) |
| `scripts/train_grpo_v2.py` | GRPO v2 (k=4, early stopping) |
| `scripts/train_grpo_v3.py` | GRPO v3 (detection-aligned reward) |
| `scripts/train_marl.py` | MARL v1 (binary shared reward) |
| `scripts/train_marl_v2.py` | MARL v2 (detection-aligned shared reward) |
| `scripts/train_marl_full_pipeline.py` | MARL full pipeline (8B + 3B) |
| `scripts/collect_pipeline_trajectories.py` | Collect full pipeline trajectories |

### Evaluation Scripts
| File | Description |
|------|-------------|
| `scripts/eval_embeddings.py` | Compare base vs fine-tuned PubMedBERT |
| `scripts/eval_medhallu_detection.py` | Hallucination detection eval on MedHallu |
| `scripts/eval_full_comparison.py` | Full 4-config comparison |
| `scripts/test_mad_debate.py` | Quick MAD test on single query |

### HPC Slurm Scripts
| File | Description |
|------|-------------|
| `hpc/setup_env.sh` | Python venv setup |
| `hpc/slurm_setup_env.sh` | Slurm job for env setup |
| `hpc/slurm_finetune_embeddings.sh` | Embedding fine-tuning |
| `hpc/slurm_eval_embeddings.sh` | Embedding eval |
| `hpc/slurm_test_mad.sh` | MAD test |
| `hpc/slurm_serve_vllm.sh` | vLLM server (not used) |
| `hpc/slurm_generate_trajectories.sh` | Trajectory generation |
| `hpc/slurm_train_grpo.sh` | GRPO v1 |
| `hpc/slurm_train_grpo_v2.sh` | GRPO v2 |
| `hpc/slurm_grpo_v3.sh` | GRPO v3 |
| `hpc/slurm_train_marl.sh` | MARL v1 |
| `hpc/slurm_marl_v2.sh` | MARL v2 |
| `hpc/slurm_marl_full_trial.sh` | MARL full pipeline |
| `hpc/slurm_collect_trajectories.sh` | Pipeline trajectory collection |
| `hpc/slurm_eval_medhallu.sh` | MedHallu eval (all configs) |
| `hpc/slurm_eval_single_critic.sh` | Single critic eval (parallel) |
| `hpc/slurm_eval_mad_base.sh` | MAD base eval (parallel) |
| `hpc/slurm_eval_mad_grpo.sh` | MAD GRPO eval (parallel) |
| `hpc/slurm_eval_comparison.sh` | Full comparison eval |

### Data Files (on HPC)
| File | Description |
|------|-------------|
| `data/embedding_triplets.jsonl` | 760 query-chunk triplets |
| `data/grpo_trajectories/train.jsonl` | 1800 MedHallu trajectories (train) |
| `data/grpo_trajectories/eval.jsonl` | 200 MedHallu trajectories (eval) |
| `data/medhallu_eval_2k.jsonl` | 2000 MedHallu pairs for evaluation |
| `data/marl_trajectories.jsonl` | 2000 full pipeline trajectories |
| `data/dpo_pairs.jsonl` | 54 DPO pairs (local eval) |
| `data/dpo_pairs_hf.jsonl` | 54 DPO pairs (HF format) |

### Models (on HPC at /home/018214196/ehr-copilot/models/)
| Directory | Description |
|-----------|-------------|
| `pubmedbert-ehr-finetuned/` | Fine-tuned embedding model |
| `verifier-grpo-v2/` | GRPO v2 Verifier (Brier reward) |
| `challenger-grpo-v2/` | GRPO v2 Challenger (Brier reward) |
| `verifier-grpo-v3/` | GRPO v3 Verifier (detection reward) — best |
| `challenger-grpo-v3/` | GRPO v3 Challenger (detection reward) — best |
| `marl/iter_1,2,3,final/` | MARL v1 (binary, didn't improve) |
| `marl-full-pipeline/8b,3b/` | MARL full pipeline trial (50 examples) |
| `marl-full-pipeline-v2/8b,3b/` | MARL full pipeline full run (500 examples) |

### Other Files
| File | Description |
|------|-------------|
| `MARL_PLAN.md` | Original MARL implementation plan |
| `Desktop/RL/critic_dpo_v1-v5.ipynb` | DPO notebooks for Colab |
| `Desktop/RL/architecture_diagram.html` | Interactive draggable architecture diagram |
| `Desktop/flow_status.html` | Architecture flow diagram |
| `Desktop/final_results.html` | Results summary page |

---

## 11. How to Reproduce

### Prerequisites
1. SSH access to SJSU COE HPC (VPN + SSH key setup)
2. Models cached at `/home/018214196/.cache/huggingface/hub/`
3. Python venv at `/home/018214196/ehr-venv/`

### Embedding Fine-Tuning
```bash
# On HPC3:
cd /home/018214196/ehr-copilot
sbatch hpc/slurm_finetune_embeddings.sh
# Eval:
sbatch hpc/slurm_eval_embeddings.sh
```

### GRPO v3 Training (Detection-Aligned)
```bash
# Train verifier and challenger in parallel:
sbatch hpc/slurm_grpo_v3.sh verifier
sbatch hpc/slurm_grpo_v3.sh challenger
```

### MARL Full Pipeline
```bash
# Trial first (50 examples):
# Edit slurm_marl_full_trial.sh: --count 50 --iterations 1
sbatch hpc/slurm_marl_full_trial.sh

# Full run:
# Edit: --count 500 --iterations 2
sbatch hpc/slurm_marl_full_trial.sh
```

### Final Evaluation (1K MedHallu, parallel)
```bash
sbatch hpc/slurm_eval_single_critic.sh
sbatch hpc/slurm_eval_mad_base.sh
sbatch hpc/slurm_eval_mad_grpo.sh
```

---

## 12. Remaining TODO

1. **Run MARL full pipeline eval** — submit eval with `models/marl-full-pipeline-v2/` models on 1K MedHallu pairs (~8 hours)
2. **100-patient MIMIC eval** — run full pipeline on all 100 MIMIC patients (800 queries) for end-to-end Entity F1. Need Qwen 3 8B on HPC.
3. **NLI-based verification** — replace LLM Verifier with DeBERTa-v3-large-mnli for better precision. Research shows NLI models are more reliable than LLM prompting for entailment.
4. **Commit remaining code** — `train_grpo_v3.py`, `train_marl_v2.py`, `train_marl_full_pipeline.py`, updated eval scripts
5. **Paper writing** — architecture section, results tables, related work, methodology
6. **Update PPT** — current slides are outdated (pre-GRPO v3, pre-MARL full pipeline)

### Paper Citations to Include
| Paper | Venue | What we adopted |
|-------|-------|-----------------|
| Adaptive-RAG | NAACL 2024 | Query-type routing (Triage Agent) |
| Self-RAG | ICLR 2024 | Reflection tokens → our Critic verdicts |
| CRAG | 2024 | Corrective retrieval → our CRAG Evaluator |
| RAG-HAT | EMNLP 2024 | DPO for hallucination tuning |
| MedHallu | EMNLP 2025 | Training + eval dataset (10K pairs) |
| MA-RAG | 2025 | Multi-agent decomposition → our Query Decomposer |
| MEGA-RAG | 2025 | Dense+Sparse+Reranking → our Hybrid Retrieval |
| MMOA-RAG | 2025 | MAPPO for RAG → our MARL shared reward |
| RAGAS | EACL 2024 | Evaluation framework |
| DeepSeek-R1 | 2025 | GRPO algorithm |
| guardrails-enterprise | — | MAD debate architecture reference |
