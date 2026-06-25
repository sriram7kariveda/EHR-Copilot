# EHR-Copilot: Complete Experiment Results & Analysis

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Track 1: Embedding Fine-Tuning](#2-track-1-embedding-fine-tuning)
3. [Track 2: Baseline Pipeline vs Frontier LLMs](#3-track-2-baseline-pipeline-vs-frontier-llms)
4. [Track 3: Hallucination Detection — Architecture](#4-track-3-hallucination-detection--architecture)
5. [Track 4: RL Training Progression](#5-track-4-rl-training-progression)
6. [Track 5: MARL Experiments](#6-track-5-marl-experiments)
7. [Track 6: Final 1K MedHallu Evaluation](#7-track-6-final-1k-medhallu-evaluation)
8. [The Prompt Distribution Mismatch Problem](#8-the-prompt-distribution-mismatch-problem)
9. [Key Decisions & Why](#9-key-decisions--why)
10. [What Worked & What Didn't](#10-what-worked--what-didnt)
11. [Conclusions for Thesis](#11-conclusions-for-thesis)

---

## 1. Executive Summary

**Project:** Multi-agent RAG pipeline for clinical EHR question answering with MARL-trained hallucination detection.

**Best results by track:**
- Embedding retrieval: MRR 0.623 → **0.914** (+46.7%)
- Pipeline vs frontier LLMs: Entity F1 **0.616** (beats GPT-5 0.329, Sonnet 4.5 0.529, Gemini 3 Pro 0.234)
- Hallucination detection (full pipeline): Single Critic F1 0.551 → MAD Debate **0.645** (+17%)
- Hallucination detection (raw prompts): Base 0.357 → MARL C3 v2 **0.752** (+111%)
- MARL training: F1 0.384 → **0.847** over 3 IBR iterations (200-sample eval)

**Critical finding:** MARL C3 v2 dramatically improved hallucination detection with raw prompts (F1=0.752), but the improvement did NOT transfer to the full debate engine pipeline (F1=0.643 ≈ base MAD 0.645). This is due to prompt distribution mismatch between training and the debate engine. See [Section 8](#8-the-prompt-distribution-mismatch-problem).

---

## 2. Track 1: Embedding Fine-Tuning

**Goal:** Improve retrieval quality for clinical EHR queries.

**Model:** PubMedBERT (NeuML/pubmedbert-base-embeddings, 110M params, 768d)

**Problem:** PubMedBERT was trained on PubMed abstracts, not clinical notes (discharge summaries, lab reports). Relevant chunks ranked 3rd-5th instead of 1st.

**Data:** 760 triplets (152 positives, 5 hard negatives each) extracted from 78 pipeline evaluations.

**Method:** MultipleNegativesRankingLoss (InfoNCE). First tried TripletLoss → accuracy dropped to 6.6%. MNRL uses in-batch negatives for better gradient signal.

**Training:** batch_size=16, epochs=5, lr=2e-5, warmup=10%, ~3 minutes on A100.

| Metric | Base PubMedBERT | Fine-tuned | Improvement |
|--------|----------------|------------|-------------|
| Recall@15 | 1.000 | 1.000 | — |
| MRR | 0.623 | **0.914** | **+46.7%** |
| NDCG@15 | 0.676 | **0.934** | **+38.2%** |

**Scripts:** `scripts/finetune_embeddings.py`, `scripts/eval_embeddings.py`

---

## 3. Track 2: Baseline Pipeline vs Frontier LLMs

**Eval:** 78 queries across 10 MIMIC-IV patients. Our pipeline (MiniMax M2.5 backbone) vs frontier LLMs in one-shot (full patient context in prompt).

**Method:** Zero-cost ground truth from FHIR structured data. Entity-level F1, precision, hallucination rate.

| Model | Entity F1 | Precision | Halluc Rate |
|-------|-----------|-----------|-------------|
| **EHR-Copilot (Ours)** | **0.616** | **0.904** | **8.2%** |
| Claude Sonnet 4.5 | 0.529 | 0.761 | 21.0% |
| Claude Haiku 3.5 | 0.480 | 0.794 | 13.4% |
| GPT-5 | 0.329 | 0.555 | 13.1% |
| Gemini 3 Pro | 0.234 | 0.626 | 23.1% |

**Key insight:** Multi-agent pipeline with specialized agents beats all frontier LLMs, even though backbone model is much smaller. Architecture > model size for clinical QA.

---

## 4. Track 3: Hallucination Detection — Architecture

**Goal:** Replace single Critic agent with Multi-Agent Debate (MAD) for better hallucination detection.

### Single Critic (Baseline)
One LLM call: answer + evidence → APPROVED/REVISED/ABSTAINED. No adversarial checking.

### MAD Debate Architecture
```
Answer → Claim Extractor → list of claims
  → Verifier (Agent A): checks each claim vs evidence → verdicts + confidence
  → Challenger (Agent B): challenges SUPPORTED claims with medical queries
  → Verifier revises confidence (NOT claim text)
  → [Repeat × 2 cycles]
  → Judge (blind — no confidence visible): scores each claim 1.0/0.5/0.0
  → Routing: material_min ≥ 0.9 → APPROVED, material_min = 0.0 → ABSTAINED, else REVISED
```

**Agents:**
- **Claim Extractor:** Decomposes answer into atomic claims (material vs contextual)
- **Verifier (Agent A):** Strict verification — only SUPPORTED if evidence EXPLICITLY states the fact
- **Challenger (Agent B):** 5 medical challenge types: CONTRAINDICATION, DOSAGE_CHECK, INTERACTION, GUIDELINE_CURRENCY, GAP_FINDING
- **Judge (Blind):** Independent scoring without seeing confidence to prevent anchoring

### Routing Decision Evolution
1. **v1 (kept):** aggregate ≥ 0.9 → APPROVED, any material 0.0 → ABSTAINED, else REVISED → F1=0.628
2. **v2 (tried, worse):** Three-zone routing → F1=0.442 (uncertain zone approved too much)
3. **v3 (tried, worse):** Zone + material check → F1=0.473
4. **Reverted to v1** — strict prompts + simple threshold beat clever routing

### Prompt Evolution
- Original Verifier: "determine if evidence SUPPORTS it" → too generous
- **Strict Verifier:** "only SUPPORTED if evidence EXPLICITLY states the same fact" → much better recall
- Original Judge: "Be strict with MATERIAL, lenient with CONTEXTUAL" → too lenient
- **Strict Judge:** "score 0.0 if evidence does NOT mention this fact" → better detection

### Initial Results (200 pairs)
| Config | F1 | Precision | Recall |
|--------|-----|-----------|--------|
| Single Critic | 0.531 | 0.647 | 0.450 |
| MAD Debate | **0.628** | 0.521 | **0.790** |

---

## 5. Track 4: RL Training Progression

### DPO Attempt (Colab, pre-GRPO)
- Qwen 3.5 4B on Colab Pro T4 16GB
- 195 steps, model COLLAPSED — output: `{{{{{...` gibberish
- Cause: 3 epochs, lr=5e-5, beta=0.1 on only 570 pairs → extreme overfit
- **Abandoned. Pivoted to GRPO.**

### GRPO v1 (k=2, Brier Reward, A100)
- **k=2 samples** (OOM with k=4 on A100 40GB)
- **Brier reward:** `2 * confidence * judge_score - confidence²`
- Verifier: 0.038 → 0.220 (+5.8×), Challenger: 0.090 → 0.180 (+2×)
- **Problem:** k=2 means both samples often get same reward → zero gradient (Loss=0.0 most steps)

### GRPO v2 (k=4, Brier Reward, H100)
- Moved to **H100 (80GB)** for k=4
- Added: gradient checkpointing, eval every 50 steps, early stopping, CSV logging
- Fixed: separate model.eval() for generation vs model.train() for gradient
- Result: same as v1 (Verifier 0.220, Challenger 0.180)
- **k=4 gave better unique_k** (3-4 distinct rewards per group vs 1-2 with k=2)

### GRPO v3 (k=4, Detection-Aligned Reward) — KEY BREAKTHROUGH
- **The problem:** v1/v2 trained on Brier reward (confidence calibration) but evaluated on detection F1. Training metric ≠ eval metric.
- **The fix:** Changed reward to directly measure detection:
  - Hallucinated answer correctly flagged = **+1.0**
  - Hallucinated answer missed = **-1.0** (worst case)
  - Good answer correctly approved = **+1.0**
  - Good answer wrongly flagged = **-0.5** (asymmetric — missing hallucination is worse)
  - Confidence bonus: ±0.3
  - Format bonus: valid JSON = +0.1

**Results:**
| Agent | Before | After | Improvement |
|-------|--------|-------|-------------|
| Verifier | 0.221 | **0.705** | **+219%** |
| Challenger | 0.132 | **0.218** | **+65%** |

**Key lesson: Train on what you measure.** Brier reward improved confidence calibration but not detection. Detection reward improved detection.

---

## 6. Track 5: MARL Experiments

### The Goal
Independent GRPO trains each agent alone. MARL trains agents to cooperate — they share a reward signal based on JOINT performance.

### MARL v1: Binary Shared Reward, Shared LoRA
- Reward: both correct = +1, one correct = 0, both wrong = -1
- 3 iterations, round-robin training
- **Result: 49% accuracy, stuck across all 3 iterations**
- **Why failed:** Binary reward → zero gradient. With k=2, both samples get same binary reward.
- Script: `scripts/train_marl.py`

### MARL v2: Detection Reward, Shared LoRA
- Same continuous reward as GRPO v3, applied jointly
- Agreement bonus: +0.3 if both agents agree
- **Result: 43% initial, cancelled at iteration 1** — too slow (40 min per 10 steps, each trajectory needs both agents)
- Script: `scripts/train_marl_v2.py`

### MARL Full Pipeline: 8B + 3B, Shared Reward
- Professor's request: MARL across the WHOLE pipeline
- LoRA on Qwen 3 8B (Triage, CRAG, Reasoning) + LoRA on Qwen 2.5 3B (Verifier, Challenger)
- Conservative 8B: LoRA r=4, lr=1e-6
- 500 trajectories, 2 iterations
- 8B reward: -0.200, 3B reward: 0.034
- **Result: Models saved, eval showed no improvement**
- Script: `scripts/train_marl_full_pipeline.py`

### MARL C3 v1: Counterfactual Credit Assignment, Shared LoRA
- Innovation: Each agent's reward = marginal contribution via counterfactual replay
  - Run both agents → pipeline score
  - Replace agent_i with default output → recompute score
  - marginal_i = score_with - score_without
- Hybrid reward: 0.5 × individual + 0.3 × counterfactual + 0.2 × shared
- 500 trajectories, 2 iterations
- **Result: 49.5% → 48.5% — GOT WORSE**
- **Why failed:** Still shared LoRA weights. Training one agent degrades the other. Only 8 gradient steps for verifier.
- Script: `scripts/train_marl_c3.py`, Job: 29131

### MARL C3 v2: Separate LoRA + IBR + Warm-Start — BREAKTHROUGH

**Four fixes applied simultaneously:**

| Fix | What changed | Why |
|-----|-------------|-----|
| Separate LoRA adapters | Two full models, no weight sharing | Eliminates interference |
| Warm-start from GRPO v3 | Start from F1=0.657, not scratch | Only learn coordination |
| Iterated Best Response (IBR) | Train verifier → freeze → train challenger → repeat | Classical game theory |
| More data + steps | 1800 traj, grad_accum=2 → ~900 steps/agent | Enough gradient signal |

**Iteration-by-iteration results (200-sample eval):**

| Iteration | Accuracy | F1 | Precision | Recall |
|-----------|----------|-----|-----------|--------|
| Warm-start (GRPO v3) | 47.0% | 0.384 | 0.559 | 0.292 |
| Iter 1 (mid — verifier trained) | 53.0% | 0.460 | — | — |
| Iter 1 (full) | 60.5% | 0.599 | 0.702 | 0.522 |
| Iter 2 (mid — verifier trained) | 64.5% | 0.679 | — | — |
| Iter 2 (full) | 72.0% | 0.769 | 0.721 | 0.823 |
| Iter 3 (mid — verifier trained) | 76.5% | 0.807 | — | — |
| **Iter 3 (full)** | **82.0%** | **0.847** | **0.813** | **0.885** |

- Every iteration improved (no plateau, still climbing at iter 3)
- Recall: 29.2% → 88.5% (catches nearly all hallucinations)
- Precision held at 81.3%
- New best saved every iteration — monotonic improvement
- Script: `scripts/train_marl_c3_v2.py`, Job: 29302

### Why MARL C3 v2 Worked (and v1 Didn't)

| Factor | Failed experiments (v1/v2/C3 v1) | C3 v2 |
|--------|----------------------------------|-------|
| LoRA weights | Shared — training one degrades other | Separate — no interference |
| Starting point | From scratch (random at task) | From GRPO v3 (already knows detection) |
| Training schedule | Simultaneous or sequential on same model | IBR: freeze one, train other |
| Gradient steps | 8-47 per agent | ~900 per agent |
| Credit assignment | Shared reward (v1/v2) or counterfactual with shared weights (C3 v1) | Counterfactual with separate weights |

---

## 7. Track 6: Final 1K MedHallu Evaluation

### Full Pipeline Eval (debate engine with claim extraction, multi-cycle debate, judge)
**Job 29899 — the paper-ready numbers**

| Config | F1 | 95% CI | Precision | Recall | Accuracy |
|--------|-----|--------|-----------|--------|----------|
| Single Critic | 0.551 | [0.522, 0.578] | 0.679 | 0.463 | — |
| MAD Debate (base) | **0.645** | [0.622, 0.666] | 0.545 | 0.789 | — |
| MAD + GRPO v3 | 0.642 | [0.620, 0.664] | 0.539 | 0.796 | — |
| MAD + MARL C3 v2 | 0.643 | [0.621, 0.665] | 0.537 | 0.802 | — |

**Key finding:** All three MAD configs are statistically indistinguishable (overlapping CIs). RL training did NOT improve the full debate pipeline.

### Raw Prompt Eval (simple verifier + challenger prompts, no debate engine)
**Job 29813 — shows what the RL actually learned**

| Config | F1 | 95% CI | Precision | Recall |
|--------|-----|--------|-----------|--------|
| Single Critic | 0.080 | [0.058, 0.104] | 0.894 | 0.042 |
| MAD Base | 0.357 | [0.325, 0.388] | 0.528 | 0.269 |
| MAD + GRPO v3 | 0.393 | [0.363, 0.423] | 0.521 | 0.315 |
| **MAD + MARL C3 v2** | **0.752** | **[0.732, 0.772]** | **0.711** | **0.799** |

**Key finding:** MARL C3 v2 massively outperforms everything (0.752 vs 0.393 for GRPO). Non-overlapping CIs — statistically significant.

---

## 8. The Prompt Distribution Mismatch Problem

### What happened
MARL C3 v2 showed F1=0.752 on raw prompts but F1=0.643 (≈ base MAD) on the full pipeline. The RL training had no effect through the debate engine.

### Why
**Training prompts:**
```
You are a STRICT clinical evidence verifier.
Answer: {full answer text}
Evidence: {evidence text}
Return ONLY JSON: {"verdict": "supported"|"not_supported"|"partial", "confidence": 0.0-1.0}
```

**Full pipeline Verifier prompts:**
```
You are a STRICT clinical evidence verifier. Your job is to catch hallucinated medical claims...
Claims to verify:
  Claim 1: "Patient has Type 2 DM" (type: material)
  Claim 2: "Metformin 500mg prescribed" (type: material)
Evidence:
  [1] Discharge Summary: ...
  [2] Lab Report: ...
STRICT RULES: ...
For each claim, respond with: verdict, confidence, evidence_chunks, reasoning
Respond as JSON array: [{"claim_id": 1, "verdict": "supported", ...}, ...]
```

The prompts are fundamentally different:
- Training: one answer → one verdict
- Pipeline: multiple extracted claims → structured array with claim_ids, evidence_chunks, reasoning

The LoRA weights learned patterns specific to the training prompt format. When the pipeline asks different questions in a different format, the fine-tuning doesn't activate.

### This is a known problem
This is prompt distribution shift — the same problem that plagues all RL fine-tuning when training prompts ≠ deployment prompts. Papers like InstructGPT, RLHF for code, etc. all note that RL improvements are brittle to prompt format changes.

### How to fix (not done yet)
**Option A (correct fix):** Generate training trajectories using the actual debate engine prompts. Capture the exact prompts `Verifier.verify_claims()` and `Challenger.challenge_claims()` produce, train GRPO/MARL on those.

**Option B (workaround):** Replace the debate engine at inference with the raw prompt format. Loses claim extraction, multi-cycle debate, judge scoring. Simpler system but real numbers.

---

## 9. Key Decisions & Why

### Decision 1: MNRL over TripletLoss for embeddings
- TripletLoss dropped accuracy to 6.6%
- Reason: positives and hard negatives were too semantically similar, TripletLoss needs well-calibrated margins
- MNRL uses in-batch negatives → much better gradient signal with small data

### Decision 2: Strict prompts over clever routing
- Three-zone routing (v2, v3) dropped F1 from 0.628 to 0.442
- Simple "be STRICT" in prompts + simple thresholds worked better
- Lesson: prompt engineering > routing logic for LLM-based verification

### Decision 3: Detection-aligned reward (GRPO v3)
- v1/v2 used Brier score (confidence calibration reward)
- v3 used detection accuracy as reward
- Verifier reward jumped 0.221 → 0.705
- Lesson: ALWAYS train on the metric you evaluate on (DeepSeek-R1, MMOA-RAG do this too)

### Decision 4: k=4 over k=2 for GRPO
- k=2: both samples often get same reward → zero gradient → Loss=0.0
- k=4: 3-4 distinct rewards per group → meaningful advantages → actual learning
- Lesson: GRPO needs sufficient sample diversity

### Decision 5: Separate LoRA adapters for MARL
- Shared LoRA: training verifier destroys challenger (all v1/v2/C3v1 failed)
- Separate LoRA: each agent specializes independently → C3 v2 succeeded
- Lesson: parameter sharing in MARL is a trap with limited data

### Decision 6: Warm-start MARL from GRPO v3
- Training MARL from scratch means agents must learn detection AND coordination simultaneously
- Warm-start from GRPO v3 = agents already know detection, MARL only teaches coordination
- Baseline jumped from ~49% (random) to 47% (GRPO v3 baseline), then improved to 82%

### Decision 7: Iterated Best Response over simultaneous training
- Simultaneous: both agents changing at once → non-stationary environment → unstable
- IBR: freeze one, train other against frozen partner → stable optimization
- Classical game theory approach, proven convergence properties

---

## 10. What Worked & What Didn't

### What Worked
1. **Embedding fine-tuning with MNRL** — MRR +47%, tiny data (760 triplets), 3 min training
2. **Strict verification prompts** — "EXPLICITLY states" >> "determine if supported"
3. **MAD debate architecture** — F1 0.551 → 0.645 (+17%) from architecture alone, no training needed
4. **Detection-aligned reward** — train on what you measure (GRPO v3)
5. **k=4 for GRPO** — sufficient sample diversity for gradient signal
6. **Separate LoRA + IBR + warm-start** — the combination that made MARL work (C3 v2)
7. **Conservative 8B training** — LoRA r=4 + lr=1e-6 kept 8B stable
8. **Counterfactual credit assignment** — each agent's marginal contribution as reward signal

### What Didn't Work
1. **DPO on Colab** — model collapsed (output: `{{{{{`). Too aggressive hyperparams on tiny data
2. **TripletLoss for embeddings** — accuracy dropped to 6.6%
3. **Three-zone routing** — F1 dropped 0.628 → 0.442. Uncertain zone approved everything
4. **Brier reward for GRPO** — improved calibration, not detection (wrong training metric)
5. **MARL v1 (binary shared reward)** — stuck at 49% for 3 iterations
6. **MARL v2 (detection + shared LoRA)** — cancelled, too slow
7. **MARL C3 v1 (counterfactual + shared LoRA)** — 49.5% → 48.5%, got worse
8. **MARL Full Pipeline (8B+3B)** — no improvement
9. **RL training through full debate pipeline** — prompt distribution mismatch negated all gains

### The Big Lesson
**Architecture changes (MAD) give robust improvements. RL training gives fragile improvements that don't survive prompt format changes.** The MAD debate consistently improved from 0.551 → 0.645 F1 regardless of whether agents were trained. The RL training (GRPO/MARL) only improved performance when evaluated with the exact prompt format used during training.

---

## 11. Conclusions for Thesis

### What we can confidently claim
1. **Multi-agent RAG pipeline beats frontier LLMs** for clinical QA (Entity F1 0.616 vs GPT-5 0.329)
2. **MAD debate improves hallucination detection** over single critic (F1 0.551 → 0.645, p < 0.05, non-overlapping 95% CIs)
3. **Embedding fine-tuning with MNRL** dramatically improves clinical retrieval (MRR +47%)
4. **MARL with separate adapters + IBR + warm-start** successfully trains cooperative multi-agent detection (F1 0.384 → 0.847 on training eval, 0.357 → 0.752 on 1K held-out with matched prompts)
5. **Shared LoRA for MARL is fundamentally broken** — four experiments all failed

### What we need to be careful about
1. The MARL improvement (0.752) is on **raw prompts**, not the full debate engine (0.643 ≈ base)
2. **Prompt distribution mismatch** is a real limitation — needs to be disclosed
3. The raw prompt eval uses a simpler pipeline (no claim extraction, no judge, no debate cycles)

### The honest story for the paper
- **Architecture contribution (MAD):** +17% F1, robust, statistically significant
- **RL contribution (MARL C3 v2):** +111% F1 on matched prompts, but 0% on full pipeline
- **Open problem:** Bridging prompt distribution shift between RL training and deployment pipeline

### Slurm Jobs Reference
| Job ID | Script | Config | Status |
|--------|--------|--------|--------|
| 28570 | train_grpo_v2.py | GRPO v2 verifier | Complete |
| 28571 | train_grpo_v2.py | GRPO v2 challenger | Complete |
| 28656 | train_marl.py | MARL v1 | Complete (failed) |
| 28686 | train_grpo_v3.py | GRPO v3 verifier | Complete |
| 28687 | train_grpo_v3.py | GRPO v3 challenger | Complete |
| 28688 | train_marl_v2.py | MARL v2 | Cancelled (timeout) |
| 28836 | train_marl_full_pipeline.py | MARL full 3B | Complete |
| 28842 | train_marl_full_pipeline.py | MARL full 3B v2 | Complete |
| 29131 | train_marl_c3.py | MARL C3 v1 (shared) | Complete (failed) |
| 29302 | train_marl_c3_v2.py | MARL C3 v2 (separate) | Complete (best) |
| 29813 | eval_marl_c3v2_1k.py | 1K raw prompt eval | Complete |
| 29899 | eval_marl_c3v2_full_pipeline.py | 1K full pipeline eval | Complete |

### Models on HPC (/home/018214196/ehr-copilot/models/)
| Directory | What |
|-----------|------|
| pubmedbert-ehr-finetuned/ | Fine-tuned embedding model |
| verifier-grpo-v2/ | GRPO v2 Verifier (Brier reward) |
| challenger-grpo-v2/ | GRPO v2 Challenger (Brier reward) |
| verifier-grpo-v3/ | GRPO v3 Verifier (detection reward) |
| challenger-grpo-v3/ | GRPO v3 Challenger (detection reward) |
| marl/iter_1,2,3,final/ | MARL v1 (binary, failed) |
| marl-full-pipeline/8b,3b/ | MARL full pipeline trial |
| marl-full-pipeline-v2/8b,3b/ | MARL full pipeline full run |
| marl-c3/iter_1,iter_2/ | MARL C3 v1 (shared, failed) |
| marl-c3-v2/best/verifier/ | MARL C3 v2 best verifier adapter |
| marl-c3-v2/best/challenger/ | MARL C3 v2 best challenger adapter |
| marl-c3-v2/iter_1,2,3/ | MARL C3 v2 iteration checkpoints |
