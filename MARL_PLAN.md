# EHR-Copilot: MARL Implementation Plan

## Final Architecture (11 agents, tree structure)

```
Clinical Query
    ↓
[Agent 1] Triage Agent — classifies into 7 query types
    ↓
[Agent 2] Query Decomposer — complex queries → 2-4 sub-queries
    ↓
[Agent 3] Hybrid Retrieval + Chunk Filter
    │   Dense (FAISS + fine-tuned PubMedBERT) ← embedding fine-tuning
    │   Sparse (BM25) → RRF Fusion → CrossEncoder Rerank
    │   Branch-specific chunk filtering
    ↓
[Agent 4] Retrieval Evaluator (CRAG) — SUFFICIENT / INSUFFICIENT / AMBIGUOUS
    ↓
[Agent 5] Branch-Specific Reasoning (5 CoT prompt templates)
    ↓
[Agent 6] Temporal Validator (TEMPORAL branches only)
[Agent 7] Numeric Validator (NUMERIC branches only)
    ↓
┌─── MULTI-AGENT DEBATE (replaces single Critic) ───┐
│  [Agent 8]  Verifier — verify claims vs evidence   │
│  [Agent 9]  Challenger — adversarial medical checks │
│  [Agent 10] Judge (blind) — scores claims 1.0/0.5/0│
└────────────────────────────────────────────────────┘
    ↓
[Agent 11] Entity Verifier — deterministic hallucination removal
    ↓
CopilotAnswer

```

## 4 Implementation Phases

### Phase 1: Embedding Fine-Tuning (HPC A100)
- Fine-tune PubMedBERT on EHR query-chunk triplets
- Positive = cited chunks, Negative = retrieved but not cited
- Multiple Negatives Ranking Loss
- Improves Context Precision/Recall

### Phase 2: Multi-Agent Debate (MAD) for Critic
- Verifier: extract claims, verify against evidence, confidence scores
- Challenger: medical challenges (contraindications, dosage, interactions, guidelines, gaps)
- Judge: blind scoring, routing decision (APPROVE/REVISE/ABSTAIN)
- SQLite trajectory logging

### Phase 3: GRPO Training for Debate Agents (HPC A100)
- Verifier reward: Brier score (calibrated confidence)
- Challenger reward: +1 challenged wrong claim, -1 challenged correct claim
- GRPO: generate multiple outputs, group-relative advantage
- No hand-crafted preference pairs needed

### Phase 4: Evaluation
| Configuration | Entity F1 | Halluc Rate | Precision |
|--------------|-----------|-------------|-----------|
| Baseline | 0.616 | 8.2% | 0.904 |
| + Fine-tuned embeddings | ? | ? | ? |
| + MAD Critic (no training) | ? | ? | ? |
| + MAD Critic with GRPO | ? | ? | ? |

## Key Papers
- MMOA-RAG (2025) — MAPPO for RAG pipeline
- Adaptive-RAG (NAACL 2024) — query routing
- CRAG (2024) — corrective retrieval
- RAG-HAT (EMNLP 2024) — hallucination-aware tuning
- MedHallu (EMNLP 2025) — medical hallucination benchmark
- guardrails-enterprise — MAD architecture reference
- GRPO / DeepSeek-R1 — group relative policy optimization

## HPC Access
- SSH hop: local → coe-hpc1.sjsu.edu → coe-hpc3
- GPUs: 4x A100 (80GB) on cs001-004, H100 on g16/g18-19
- Partition: gpuqs (2-day), gpuqm (7-day)
- Python 3.12, CUDA 12, pip3 (no conda)
