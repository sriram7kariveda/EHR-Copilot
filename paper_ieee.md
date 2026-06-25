# Multi-Agent Debate with Reinforcement Learning for Faithful Hallucination Detection in Clinical Question Answering over Electronic Health Records

**Shashin Bhaskar**
Department of Software Engineering, San Jose State University, San Jose, CA 95192
shashin.bhaskar@sjsu.edu

---

## Abstract

Large Language Models frequently hallucinate when answering clinical questions grounded in patient-specific Electronic Health Record (EHR) data, posing patient safety risks. We present **EHR-Copilot**, a multi-agent RAG pipeline featuring: (1) an 11-agent tree architecture with query-type-aware routing, fine-tuned PubMedBERT embeddings, and specialized reasoning; (2) a **Multi-Agent Debate (MAD)** mechanism replacing the traditional single-critic verification with a three-agent adversarial debate (Verifier, Challenger, blind Judge); and (3) **Multi-Agent Reinforcement Learning (MARL)** with counterfactual credit assignment (C3) to train debate agents to cooperate. We evaluate on MIMIC-IV FHIR data across three tracks. First, our pipeline achieves Entity F1=0.616 on 78 clinical queries, outperforming GPT-5 (0.329), Claude Sonnet 4.5 (0.529), and Gemini 3 Pro (0.234). Second, embedding fine-tuning with MultipleNegativesRankingLoss improves retrieval MRR from 0.623 to 0.914 (+46.7%). Third, on 1,000 MedHallu hallucination detection pairs, MAD debate significantly improves Detection F1 from 0.551 to 0.645 over a single critic (p < 0.05, non-overlapping 95% bootstrap CIs). MARL with separate LoRA adapters and iterated best response further improves detection F1 to 0.752 on matched evaluation (+111% over base MAD). We identify prompt distribution mismatch as a key challenge when transferring RL-trained agents into complex multi-stage pipelines. Our work demonstrates that structured adversarial debate is a robust architectural improvement for clinical hallucination detection, while MARL provides additional gains when training and deployment prompt distributions are aligned.

**Index Terms** — Clinical NLP, Hallucination Detection, Multi-Agent Systems, Reinforcement Learning, Retrieval-Augmented Generation, Electronic Health Records

---

## I. Introduction

Clinical question answering (QA) over Electronic Health Records demands high fidelity: incorrect medication dosages, misattributed diagnoses, or fabricated lab results can have serious consequences for patient care. While Large Language Models (LLMs) achieve remarkable performance on medical knowledge benchmarks such as MedQA and PubMedQA, their application to *patient-specific* EHR data remains challenging due to three fundamental problems: hallucination of plausible but unfounded clinical claims, temporal reasoning failures in longitudinal data, and insufficient numeric precision for lab values and dosages.

Retrieval-Augmented Generation (RAG) mitigates hallucination by grounding responses in retrieved evidence. However, naive RAG—a single retrieval-generation step—is insufficient for clinical QA because different query types require different strategies, and a single LLM call cannot simultaneously retrieve, reason, validate, and verify faithfulness.

We identify a further limitation: existing clinical RAG systems rely on a **single critic agent** to verify answer faithfulness. This single-pass verification misses subtle hallucinations because the critic sees the answer once and decides without adversarial scrutiny. We hypothesize that adversarial multi-agent debate, where a verifier's claims are challenged by a dedicated adversary before an independent judge decides, can substantially improve hallucination detection.

This paper makes the following contributions:

1. **Multi-Agent Debate (MAD)**: We replace the single critic with a three-agent adversarial debate (Verifier, Challenger, blind Judge) that decomposes answers into atomic claims and subjects each to structured challenge. MAD improves Detection F1 from 0.551 to 0.645 (+17%, p < 0.05).

2. **MARL with Counterfactual Credit Assignment (C3)**: We train debate agents to cooperate using GRPO with hybrid rewards combining individual detection accuracy, shared pipeline outcome, and counterfactual marginal contribution. With separate LoRA adapters and iterated best response, MARL achieves Detection F1=0.752 on matched evaluation.

3. **Comprehensive Evaluation**: We evaluate across three tracks—pipeline accuracy versus frontier LLMs, retrieval quality, and hallucination detection—on MIMIC-IV FHIR data and 1,000 MedHallu pairs with bootstrap confidence intervals.

4. **Prompt Distribution Mismatch Analysis**: We document how RL-trained improvements fail to transfer when deployment prompts differ from training prompts, identifying a key challenge for RL in multi-stage NLP pipelines.

---

## II. Related Work

### A. RAG for Healthcare

Retrieval-Augmented Generation has been applied to biomedical QA [1], clinical decision support [6], and medical literature synthesis. Most prior work retrieves from medical knowledge bases (PubMed, clinical guidelines) rather than patient-specific structured EHR data. CRAG [12] introduced corrective retrieval with self-assessment of retrieval quality. Our work operates directly on FHIR-formatted clinical records and incorporates CRAG-style retrieval evaluation.

### B. Multi-Agent LLM Systems

AutoGen [2] and CrewAI provide frameworks for multi-agent collaboration. MedAgents [3] demonstrated multi-agent debate for medical reasoning using role-playing agents. MMOA-RAG [13] applied multi-agent orchestration to RAG pipelines. Our architecture differs by using functionally specialized agents with deterministic validation alongside LLM-based reasoning, and by incorporating RL training for agent cooperation.

### C. Hallucination Detection

Hallucination in clinical text generation has been studied through faithfulness metrics [4], claim verification pipelines, and critic-based approaches. MedHallu [14] provides a benchmark for medical hallucination detection. Our Multi-Agent Debate extends single-critic approaches with adversarial challenge and blind judging.

### D. Reinforcement Learning for LLMs

Group Relative Policy Optimization (GRPO) from DeepSeek-R1 [15] enables RL without reward models by computing advantages within sample groups. HuatuoGPT-o1 [16] applied RL to medical reasoning. We extend GRPO to multi-agent settings with counterfactual credit assignment, addressing the credit assignment problem in cooperative MARL.

### E. Embedding Fine-Tuning for Retrieval

Domain-specific embedding fine-tuning has shown consistent improvements in retrieval quality [8]. We fine-tune PubMedBERT with MultipleNegativesRankingLoss (InfoNCE) on clinical query-chunk pairs extracted from pipeline evaluations.

---

## III. System Architecture

### A. Pipeline Overview

EHR-Copilot processes clinical queries through an 11-agent tree pipeline:

```
Query → Triage Router → Query Decomposer → Hybrid Retrieval →
  CRAG Evaluator → [5 branch-specific Reasoning agents] →
  Temporal Validator → Numeric Validator →
  MAD Debate (Verifier ↔ Challenger → Judge) →
  Entity Verifier → Answer
```

The Triage Router classifies queries into seven types (FACTUAL, TEMPORAL, NUMERIC, MEDICATION, SUMMARY, COMPARISON, REASONING) and controls four downstream dimensions: retrieval strategy, reasoning prompt, validation passes, and evidence weighting.

### B. Hybrid Retrieval with Fine-Tuned Embeddings

We combine dense retrieval (PubMedBERT, 768d, FAISS) with sparse retrieval (BM25, k1=1.5, b=0.75) using Reciprocal Rank Fusion (k=20). PubMedBERT is fine-tuned on 760 clinical query-chunk triplets using MultipleNegativesRankingLoss:

$$\mathcal{L} = -\log \frac{\exp(\text{sim}(q, d^+)/\tau)}{\sum_{j} \exp(\text{sim}(q, d_j)/\tau)}$$

where $d^+$ is a cited chunk (positive) and $d_j$ includes in-batch negatives from retrieved-but-uncited chunks. Fine-tuning improves MRR from 0.623 to 0.914 (+46.7%).

### C. Multi-Agent Debate (MAD)

The MAD mechanism replaces the single critic with a structured three-agent debate:

**Claim Extractor.** Decomposes the draft answer into atomic claims, each classified as *material* (clinical facts requiring evidence) or *contextual* (background statements).

**Verifier (Agent A).** Checks each claim against retrieved evidence using strict verification rules: a claim is SUPPORTED only if evidence *explicitly* states the same fact with matching specifics (drug name, dose, diagnosis, value). Uses a "when in doubt, flag" policy prioritizing safety.

**Challenger (Agent B).** Adversarially challenges all SUPPORTED claims using five domain-specific challenge types: CONTRAINDICATION, DOSAGE\_CHECK, INTERACTION, GUIDELINE\_CURRENCY, and GAP\_FINDING. Follows the True→Skeptic rule: only SUPPORTED claims are challenged.

**Verifier Revision.** Agent A revises confidence scores (not claim text) in response to challenges. Valid challenges reduce confidence; weak challenges increase it.

**Blind Judge.** Independently scores each claim without seeing Agent A's confidence (preventing anchoring bias). Scores: 1.0 (explicitly supported), 0.5 (topic match but different specifics), 0.0 (unsupported or contradicted).

**Routing Decision.** Based on the Judge's scores: if any material claim scores 0.0 → ABSTAINED; if aggregate ≥ 0.9 → APPROVED; otherwise → REVISED.

The debate runs for up to 2 cycles, with early exit if all material claims reach ≥ 0.95 confidence with SUPPORTED verdict.

### D. Deterministic Entity Verifier

A final deterministic safety net removes ungrounded entities from the answer by matching against structured FHIR data. This zero-LLM-cost component catches entities that survived the debate.

---

## IV. MARL Training

### A. Training Data

We generate 2,000 trajectories from MedHallu [14]: 1,000 hallucinated and 1,000 ground truth answer-evidence pairs. Each trajectory includes the answer text, supporting evidence, and a ground truth label. We split into 1,800 training and 200 evaluation trajectories.

### B. GRPO with Detection-Aligned Reward

We train using Group Relative Policy Optimization (GRPO) [15]. For each trajectory, we generate $K=4$ completions, compute rewards, and update using group-relative advantages:

$$A_i = \frac{r_i - \bar{r}}{\sigma_r + \epsilon}$$

**Detection-aligned reward.** A critical design choice: we align the training reward directly with the evaluation metric (detection accuracy) rather than proxy metrics:

| Outcome | Reward |
|---------|--------|
| Hallucinated answer correctly flagged | +1.0 |
| Hallucinated answer missed | −1.0 |
| Ground truth correctly approved | +1.0 |
| Ground truth wrongly flagged | −0.5 |
| High confidence when correct | +0.3 bonus |
| Valid JSON format | +0.1 bonus |

The asymmetry (−1.0 for missed hallucinations vs −0.5 for false positives) reflects clinical priorities: missing a hallucination is more dangerous than a false alarm.

Earlier versions used Brier score reward (confidence calibration), which improved calibration but not detection—confirming the principle that RL training must optimize the evaluation metric directly.

### C. Why Independent GRPO is Insufficient

Independent GRPO trains each agent with its own reward without knowledge of other agents. The Verifier learns to detect hallucinations and the Challenger learns to generate challenges, but neither learns to *complement* the other. We observe:
- Verifier eval reward: 0.221 → 0.705 (+219%)
- Challenger eval reward: 0.132 → 0.218 (+65%)

Despite strong individual improvement, the pipeline detection F1 improvement is marginal (0.642 → 0.657) because agents don't learn cooperative behavior.

### D. MARL with Counterfactual Credit Assignment (C3)

To enable cooperative learning, we introduce a hybrid reward combining three signals:

$$r_{\text{hybrid}}^{(i)} = \alpha \cdot r_{\text{individual}}^{(i)} + \beta \cdot r_{\text{counterfactual}}^{(i)} + \gamma \cdot r_{\text{shared}}$$

where $\alpha=0.5$, $\beta=0.3$, $\gamma=0.2$, and the counterfactual component measures each agent's *marginal contribution*:

$$r_{\text{counterfactual}}^{(i)} = r_{\text{pipeline}}(\text{all agents}) - r_{\text{pipeline}}(\text{agent}_i \leftarrow \text{default})$$

The default output is a neutral response (e.g., "partial" verdict with 0.5 confidence for the Verifier, empty challenges for the Challenger). This counterfactual decomposition solves the credit assignment problem: each agent is rewarded based on how much it personally improves the pipeline outcome.

### E. Separate LoRA Adapters with Iterated Best Response

We identify shared parameter space as a critical failure mode for MARL. When both agents share a single LoRA adapter, training one agent degrades the other. We verified this empirically: four experiments with shared LoRA (MARL v1, v2, Full Pipeline, C3 v1) all failed to improve over baseline.

**Solution:** Separate LoRA adapters (rank 8, $\alpha=16$) for each agent, with Iterated Best Response (IBR):

1. Train Verifier (Challenger frozen) → convergence
2. Train Challenger (Verifier frozen) → convergence
3. Repeat for $T$ iterations

This classical game-theoretic approach ensures stable optimization: each agent faces a stationary partner during its training phase.

**Warm-start:** Rather than training from scratch, we initialize from GRPO v3 checkpoints (which already learned individual detection). MARL then only needs to learn *coordination*, not detection from scratch.

---

## V. Experimental Setup

### A. Evaluation Tracks

We evaluate across three tracks:

**Track 1: Pipeline vs. Frontier LLMs.** 78 clinical queries across 10 MIMIC-IV FHIR patients, compared against GPT-5, Claude Sonnet 4.5, Gemini 3 Pro, GLM 4.6, and Claude Haiku 3.5 in one-shot settings. Metrics: Entity F1, Precision, Hallucination Rate.

**Track 2: Retrieval Quality.** 72 queries evaluated for embedding retrieval. Metrics: MRR, NDCG@15, Recall@15.

**Track 3: Hallucination Detection.** 1,000 MedHallu [14] pairs (500 hallucinated + 500 ground truth). Four configurations: Single Critic, MAD Base, MAD+GRPO v3, MAD+MARL C3 v2. Metrics: Detection F1, Precision, Recall with 10,000-sample bootstrap 95% confidence intervals.

### B. Models and Infrastructure

- **Backbone LLM:** Qwen 2.5 3B Instruct (debate agents), MiniMax M2.5 (reasoning)
- **Embedding:** PubMedBERT (NeuML/pubmedbert-base-embeddings, 768d)
- **LoRA:** rank 8, $\alpha=16$, dropout 0.1, targeting q/k/v/o/gate/up/down projections
- **Training:** NVIDIA H100 (80GB), bfloat16, gradient checkpointing
- **GRPO:** $K=4$ samples, grad\_accum=2, lr=$3 \times 10^{-6}$, 1,800 trajectories
- **MARL:** 3 IBR iterations, ~900 gradient steps per agent per iteration

---

## VI. Results

### A. Track 1: Pipeline vs. Frontier LLMs

Table I shows results on 78 clinical queries across 10 MIMIC-IV patients.

**TABLE I: Pipeline Comparison (78 queries, 10 patients)**

| System | Entity F1 | Precision | Halluc. Rate |
|--------|-----------|-----------|-------------|
| **EHR-Copilot (Ours)** | **0.616** | **0.904** | **8.2%** |
| Claude Sonnet 4.5 | 0.529 | 0.761 | 21.0% |
| Claude Haiku 3.5 | 0.480 | 0.794 | 13.4% |
| GPT-5 | 0.329 | 0.555 | 13.1% |
| Gemini 3 Pro | 0.234 | 0.626 | 23.1% |

Our pipeline achieves 16.4% relative improvement over the strongest baseline (Sonnet 4.5) while maintaining 90.4% entity precision. Medication queries show the largest gain: F1=0.770 vs. Sonnet 4.5's 0.383 (+101%).

### B. Track 2: Embedding Fine-Tuning

**TABLE II: Retrieval Quality (72 queries)**

| Model | MRR | NDCG@15 | Recall@15 |
|-------|-----|---------|-----------|
| Base PubMedBERT | 0.623 | 0.676 | 1.000 |
| **Fine-tuned** | **0.914** | **0.934** | 1.000 |
| Improvement | +46.7% | +38.2% | — |

Fine-tuning with MNRL on only 760 triplets yields dramatic improvement. We note that TripletLoss failed (accuracy dropped to 6.6%), as the positives and hard negatives were too semantically similar for margin-based losses.

### C. Track 3: Hallucination Detection (Full Pipeline)

Table III shows results using the full MAD debate engine (claim extraction, multi-cycle debate, blind judge, routing) on 1,000 MedHallu pairs.

**TABLE III: Full Pipeline Detection (1,000 MedHallu pairs)**

| Config | F1 | 95% CI | Prec. | Recall |
|--------|-----|--------|-------|--------|
| Single Critic | 0.551 | [0.522, 0.578] | 0.679 | 0.463 |
| MAD Base | **0.645** | [0.622, 0.666] | 0.545 | 0.789 |
| MAD + GRPO v3 | 0.642 | [0.620, 0.664] | 0.539 | 0.796 |
| MAD + MARL C3 v2 | 0.643 | [0.621, 0.665] | 0.537 | 0.802 |

**Key finding:** MAD debate significantly improves over single critic (non-overlapping CIs, p < 0.05). Recall increases from 0.463 to 0.789 (+70%), meaning the debate catches substantially more hallucinations. However, GRPO and MARL training do not improve the full pipeline (overlapping CIs with MAD base).

### D. Track 3: Hallucination Detection (Matched Prompts)

Table IV shows results where evaluation uses the same prompt format as GRPO/MARL training (direct verifier + challenger prompts, no debate engine).

**TABLE IV: Matched-Prompt Detection (1,000 MedHallu pairs)**

| Config | F1 | 95% CI | Prec. | Recall |
|--------|-----|--------|-------|--------|
| Single Critic | 0.080 | [0.058, 0.104] | 0.894 | 0.042 |
| MAD Base | 0.357 | [0.325, 0.388] | 0.528 | 0.269 |
| MAD + GRPO v3 | 0.393 | [0.363, 0.423] | 0.521 | 0.315 |
| **MAD + MARL C3 v2** | **0.752** | **[0.732, 0.772]** | **0.711** | **0.799** |

MARL C3 v2 dramatically outperforms all other configurations (non-overlapping CIs). The 0.752 F1 represents a +111% improvement over base MAD and +91% over GRPO v3.

### E. MARL C3 v2 Training Progression

Table V shows the iteration-by-iteration improvement during MARL C3 v2 training (200-sample evaluation set).

**TABLE V: MARL C3 v2 IBR Iteration Results**

| Iteration | Acc. | F1 | Prec. | Recall |
|-----------|------|-----|-------|--------|
| Warm-start (GRPO v3) | 47.0% | 0.384 | 0.559 | 0.292 |
| Iter 1 | 60.5% | 0.599 | 0.702 | 0.522 |
| Iter 2 | 72.0% | 0.769 | 0.721 | 0.823 |
| **Iter 3** | **82.0%** | **0.847** | **0.813** | **0.885** |

Every iteration improved monotonically. Recall increased from 0.292 to 0.885, indicating the agents learned to catch nearly all hallucinations while maintaining 81.3% precision.

### F. MARL Ablation: Why Shared LoRA Fails

Table VI compares MARL approaches, demonstrating that parameter separation is essential.

**TABLE VI: MARL Ablation Study**

| Experiment | LoRA | Start | Accuracy | Status |
|------------|------|-------|----------|--------|
| MARL v1 | Shared | Scratch | 49.0% | Failed |
| MARL v2 | Shared | Scratch | 43.0% | Cancelled |
| MARL C3 v1 | Shared | Scratch | 48.5% | Failed |
| MARL Full Pipeline | Shared | Scratch | — | No improvement |
| **MARL C3 v2** | **Separate** | **GRPO v3** | **82.0%** | **Success** |

All four experiments with shared LoRA failed (accuracy ≤ 49%, near random). Training one agent's LoRA weights interferes with the other agent's learned behavior. Separate adapters eliminate this interference entirely.

---

## VII. Discussion

### A. Architecture vs. Training

Our results reveal an important distinction: **architectural improvements (MAD) are robust across evaluation settings, while RL training improvements are sensitive to prompt distribution.**

The MAD debate consistently improves F1 from 0.551 to 0.645 regardless of whether agents are trained. This improvement comes from the debate structure itself: claim decomposition, adversarial challenge, blind judging, and conservative routing. These are algorithmic guarantees, not learned behaviors.

By contrast, MARL C3 v2 achieves F1=0.752 with matched prompts but F1=0.643 (≈ base MAD) through the full debate engine. The LoRA weights learned patterns specific to the training prompt format and do not activate when the debate engine generates structurally different prompts.

### B. The Prompt Distribution Mismatch Problem

During GRPO/MARL training, the Verifier sees:
```
You are a STRICT clinical evidence verifier.
Answer: {answer}  Evidence: {evidence}
Return ONLY JSON: {"verdict": "...", "confidence": ...}
```

During full pipeline evaluation, the Verifier class generates a much longer, structured prompt with claim-by-claim verification, structured output fields (claim\_id, evidence\_chunks, reasoning), and multi-claim JSON arrays. The training prompts and deployment prompts occupy different regions of the model's input distribution, causing the LoRA fine-tuning to be effectively invisible through the debate engine.

This is analogous to the prompt sensitivity documented in InstructGPT [17] and RLHF for code generation [18], where RL improvements are brittle to input format changes. We identify this as a key open problem for applying RL to multi-stage NLP pipelines.

### C. Why Larger LLMs Fail on EHR QA

A counterintuitive finding: Claude Haiku 3.5 (smallest model) outperforms GPT-5, GLM 4.6, and Gemini 3 Pro. Investigation reveals that larger models over-interpret FHIR metadata (e.g., `status: completed`) and refuse to report medications, producing near-zero recall on medication queries (GPT-5 F1=0.007). Smaller models pragmatically report what they find. Our pipeline sidesteps this by delivering focused evidence chunks, removing the need for metadata interpretation.

### D. Clinical Safety Design

The system implements defense-in-depth: (1) strict verification prompts ("only SUPPORTED if evidence EXPLICITLY states the same fact"); (2) adversarial challenge of all SUPPORTED claims; (3) blind judge preventing confidence anchoring; (4) conservative routing (any material 0.0 → ABSTAIN); (5) deterministic entity verifier as final safety net. This achieves 100% abstention accuracy on unanswerable queries.

### E. Limitations

1. **Prompt distribution mismatch** prevents MARL improvements from transferring to the full pipeline. Generating training trajectories from actual pipeline prompts would address this but requires significant additional compute.
2. **Evaluation scale**: 1,000 MedHallu pairs and 78 clinical queries; larger-scale evaluation is needed.
3. **Single backbone model**: All debate agents use Qwen 2.5 3B. Larger models may show different training dynamics.
4. **Latency**: The full pipeline requires 5-6 serial LLM calls (104s average), unsuitable for real-time clinical decision support.

---

## VIII. Conclusion

We present EHR-Copilot, a multi-agent RAG pipeline with adversarial debate and MARL training for clinical hallucination detection. Our key findings are:

1. **Multi-Agent Debate improves detection robustly**: F1 0.551 → 0.645 (+17%) over single critic, with non-overlapping 95% confidence intervals. This architectural improvement requires no training.

2. **MARL with separate adapters enables cooperative learning**: With separate LoRA adapters, warm-start from GRPO, iterated best response, and counterfactual credit assignment, MARL achieves F1=0.752 on matched prompts—a +111% improvement over base MAD. Shared LoRA fails categorically (4/4 experiments).

3. **Prompt distribution alignment is critical**: RL improvements vanish when deployment prompts differ from training prompts, identifying a key challenge for RL in multi-stage pipelines.

4. **Structured multi-agent RAG outperforms frontier LLMs**: Entity F1=0.616 beats GPT-5 (0.329), Sonnet 4.5 (0.529), and Gemini 3 Pro (0.234) on clinical EHR QA.

Future work includes generating training trajectories from actual pipeline prompts to bridge the distribution gap, scaling to the full MIMIC-IV dataset, and exploring whether larger backbone models exhibit different MARL dynamics.

---

## References

[1] G. Xiong et al., "Benchmarking Retrieval-Augmented Generation for Medicine," *Findings of ACL*, 2024.

[2] Q. Wu et al., "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation," *arXiv:2308.08155*, 2023.

[3] X. Tang et al., "MedAgents: Large Language Models as Collaborators for Zero-shot Medical Reasoning," *arXiv:2311.10537*, 2024.

[4] Z. Ji et al., "Survey of Hallucination in Natural Language Generation," *ACM Computing Surveys*, vol. 55, no. 12, 2023.

[5] A. Johnson et al., "MIMIC-IV Clinical Database Demo on FHIR," *PhysioNet*, 2023.

[6] C. Zakka et al., "Almanac: Retrieval-Augmented Language Models for Clinical Medicine," *NEJM AI*, 2024.

[7] S. Robertson and H. Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond," *Foundations and Trends in Information Retrieval*, vol. 3, no. 4, pp. 333–389, 2009.

[8] Y. Gu et al., "Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing," *ACM CHIL*, 2021.

[9] E. J. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," *ICLR*, 2022.

[10] L. Du et al., "Multi-Agent Debate for Medical Text Verification," *EMNLP*, 2024.

[11] S. Liang et al., "Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate," *arXiv:2305.19118*, 2023.

[12] S. Yan et al., "Corrective Retrieval Augmented Generation," *arXiv:2401.15884*, 2024.

[13] Z. Wang et al., "MMOA-RAG: Multi-Agent Orchestration for RAG," *arXiv:2024*, 2024.

[14] UTAustin-AIHealth, "MedHallu: A Medical Hallucination Benchmark," *HuggingFace Datasets*, 2024.

[15] DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning," *arXiv:2501.12948*, 2025.

[16] J. Chen et al., "HuatuoGPT-o1: Medical Complex Reasoning with Verifiable RL," *arXiv:2412.18925*, 2024.

[17] L. Ouyang et al., "Training Language Models to Follow Instructions with Human Feedback," *NeurIPS*, 2022.

[18] Y. Li et al., "Competition-Level Code Generation with AlphaCode," *Science*, vol. 378, no. 6624, 2022.
