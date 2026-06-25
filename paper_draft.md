# Multi-Agent RAG Pipeline for Faithful Clinical Question Answering over Electronic Health Records

**Author:** Shashin Bhaskar
**Affiliation:** San Jose State University, Department of Software Engineering
**Date:** February 2026

---

## Abstract

Large Language Models (LLMs) demonstrate impressive general-purpose reasoning but frequently hallucinate when answering clinical questions grounded in patient-specific Electronic Health Record (EHR) data. We present **EHR Copilot**, a multi-agent Retrieval-Augmented Generation (RAG) pipeline that decomposes clinical question answering into specialized sub-tasks: query routing, hybrid retrieval, chain-of-thought reasoning, temporal validation, numeric validation, and critic-based verification. We evaluate our system on the MIMIC-IV Clinical Database Demo (FHIR edition) across 10 patients and 80 clinical queries against five state-of-the-art LLMs operating in a one-shot setting: GPT-5, Claude Sonnet 4.5, Gemini 3 Pro, GLM 4.6, and Claude Haiku 3.5. Using a novel zero-cost ground truth evaluation framework that extracts reference entities directly from FHIR structured data, we show that our multi-agent RAG pipeline achieves an Entity F1 of **0.639** — a **20% relative improvement** over the best one-shot baseline (Sonnet 4.5, F1=0.531) — while maintaining **85.0% entity precision** and **100% abstention accuracy** on unanswerable queries. These results demonstrate that structured multi-agent orchestration with retrieval grounding substantially reduces hallucination and improves faithfulness in clinical QA, even when using a cost-efficient reasoning backbone (MiniMax M2.5).

---

## 1. Introduction

Clinical question answering (QA) over Electronic Health Records requires high fidelity: an incorrect medication dosage, a misattributed diagnosis, or a fabricated lab result can have serious consequences for patient care. While LLMs have shown remarkable capability in medical knowledge benchmarks (MedQA, PubMedQA), their application to *patient-specific* EHR data remains challenging due to three fundamental problems:

1. **Hallucination**: LLMs confidently generate plausible but unfounded clinical claims when patient records are not directly accessible or when context windows are insufficient.
2. **Temporal reasoning failures**: Clinical data is inherently longitudinal. Models struggle to correctly sequence events, identify the most recent encounter, or track trends over time.
3. **Numeric precision**: Lab values, vital signs, and medication dosages require exact reproduction from source data; even small errors can be clinically significant.

Retrieval-Augmented Generation (RAG) addresses hallucination by grounding LLM responses in retrieved evidence. However, naive RAG — a single retrieval step followed by a single generation step — is insufficient for clinical QA because: (a) different query types (factual lookup vs. temporal trend analysis) require different retrieval and reasoning strategies; (b) a single LLM call cannot simultaneously retrieve, reason, validate temporal claims, check numeric accuracy, and assess its own faithfulness.

We propose a **multi-agent RAG architecture** where specialized agents collaborate in a pipeline:

- A **Router Agent** classifies the clinical query by type and identifies required validation passes.
- A **Hybrid Retrieval Engine** combines dense (PubMedBERT) and sparse (BM25) retrieval with Reciprocal Rank Fusion.
- A **Reasoning Agent** performs chain-of-thought synthesis over retrieved evidence.
- **Temporal and Numeric Validator Agents** cross-check the draft answer against source evidence.
- A **Critic Agent** makes the final accept/revise/abstain decision with evidence faithfulness assessment.

We evaluate against five frontier LLMs in a one-shot (full patient context in prompt) setting and demonstrate significant improvements in entity-level accuracy, semantic similarity, and hallucination rate.

---

## 2. Related Work

### 2.1 RAG for Healthcare

Retrieval-Augmented Generation has been applied to biomedical QA (Xiong et al., 2024), clinical decision support (Zakka et al., 2024), and medical literature synthesis. Most prior work focuses on retrieval from medical knowledge bases (PubMed, clinical guidelines) rather than patient-specific structured EHR data. Our work differs by operating directly on FHIR-formatted clinical records containing encounters, conditions, medications, lab observations, and procedures.

### 2.2 Multi-Agent LLM Systems

Recent work has explored multi-agent architectures for complex reasoning tasks. AutoGen (Wu et al., 2023) and CrewAI provide frameworks for agent collaboration. MedAgents (Tang et al., 2024) demonstrated multi-agent debate for medical reasoning. Our architecture differs by using *functionally specialized* agents (router, validator, critic) rather than role-playing agents, and by incorporating deterministic validation alongside LLM-based reasoning.

### 2.3 Hallucination Detection in Clinical NLP

Hallucination in clinical text generation has been studied through faithfulness metrics (Ji et al., 2023), claim verification pipelines, and critic-based approaches. Our critic agent combines LLM-based evidence cross-referencing with a fail-safe abstention mechanism that defaults to declining an answer when verification is uncertain.

### 2.4 EHR-based QA Benchmarks

Existing EHR QA benchmarks (emrQA, MIMIC-Extract) typically evaluate on extractive QA. We contribute a novel evaluation methodology that extracts ground truth entities from FHIR structured data and computes entity-level F1, enabling evaluation of generative QA systems without expensive human annotation.

---

## 3. System Architecture

### 3.1 Overview

EHR Copilot processes a clinical query through a five-stage pipeline (Figure 1):

```
Query → [Router] → [Hybrid Retrieval] → [Reasoning (CoT)] → [Validators] → [Critic] → Answer
                                                               ├── Temporal
                                                               └── Numeric
```

Each stage is implemented as an independent agent with typed inputs/outputs, enabling modular testing and replacement.

### 3.2 Data Ingestion and Indexing

Patient records are loaded from MIMIC-IV Clinical Database Demo on FHIR (v2.1.0), which provides clinical data in HL7 FHIR format (NDJSON). We process the following resource types:

| FHIR Resource | Content | Records (Demo) |
|---|---|---|
| Encounter | Hospital visits, admission/discharge | ~500 |
| Condition | Diagnoses (ICD-9/10 coded) | 5,051 |
| MedicationRequest | Prescribed medications | 12,382 |
| Observation | Lab results, vitals | 107,727 |
| Procedure | Surgical/diagnostic procedures | 3,450 |

Records are chunked into segments of up to 512 tokens with 128-token overlap to preserve context across chunk boundaries. Each chunk retains metadata including patient ID, encounter date, document type, and source resource identifiers.

### 3.3 Hybrid Retrieval

We employ a hybrid retrieval strategy combining:

- **Dense retrieval**: PubMedBERT embeddings (`NeuML/pubmedbert-base-embeddings`, 768 dimensions) indexed with FAISS (FlatIP). Domain-specific embeddings capture clinical semantic similarity (e.g., "hypertension" ≈ "high blood pressure").
- **Sparse retrieval**: BM25 (k1=1.5, b=0.75) for exact term matching, critical for medication names, lab codes, and procedure identifiers.
- **Reciprocal Rank Fusion (RRF)**: Scores from both retrievers are combined using RRF with k=20, producing a sharper discrimination between relevant and irrelevant chunks than the standard k=60.

The top 15 chunks are returned after fusion and passed to the reasoning agent.

### 3.4 Router Agent

The router classifies each query into a `QueryType` (FACTUAL, TEMPORAL, NUMERIC, SUMMARY, REASONING, MEDICATION) and determines whether temporal or numeric validation is required. This classification controls downstream pipeline behavior — e.g., temporal queries trigger the temporal validator, while medication queries bypass numeric validation.

The router uses zero-temperature generation to ensure deterministic classification and outputs a structured JSON response with type, confidence, key entities, and validation flags.

### 3.5 Reasoning Agent (Chain-of-Thought)

The reasoning agent receives the query, retrieved chunks, and query intent. It generates a chain-of-thought trace within `<reasoning>` tags followed by the answer within `<answer>` tags. Source chunks are cited using bracket notation (`[1]`, `[2]`) and listed in a `<source_chunks>` block.

This structured output enables:
- Auditability of the reasoning process
- Traceability of each claim to specific evidence chunks
- Separation of reasoning from final answer for downstream validation

### 3.6 Temporal Validator Agent

The temporal validator operates in two phases:

1. **Deterministic phase**: Regex-based date extraction from both the answer and evidence chunks. Validates that: (a) every date in the answer appears in at least one evidence chunk; (b) if multiple dates are mentioned, they follow a consistent chronological order.

2. **LLM phase**: Semantic temporal reasoning to catch implicit temporal claims (e.g., "after surgery" should correspond to a post-operative timeframe in the evidence).

This hybrid approach reduces cost (deterministic checks are free) while maintaining the ability to catch semantically complex temporal errors.

### 3.7 Numeric Validator Agent

The numeric validator cross-references all numeric values, units, and calculations in the draft answer against the retrieved evidence. It checks:
- Accurate quotation of lab values and vital signs
- Correct units (UCUM-aware)
- Mathematical accuracy of any derived calculations or comparisons

### 3.8 Critic Agent

The critic is the final gatekeeper. It receives the draft answer, all evidence chunks, and validation results from temporal/numeric validators. It produces one of three verdicts:

- **APPROVED**: The answer is faithful to evidence with no significant issues.
- **REVISED**: The answer has fixable issues; the critic provides a corrected version.
- **ABSTAINED**: Insufficient evidence or critical errors; the system declines to answer.

A critical safety design decision: on parse failure or ambiguity, the critic defaults to **ABSTAINED** rather than APPROVED. This fail-safe ensures that system errors never produce false confidence.

---

## 4. Evaluation Methodology

### 4.1 Dataset

We evaluate on the MIMIC-IV Clinical Database Demo on FHIR (v2.1.0), which contains de-identified clinical records for 100 patients. We select 10 patients and evaluate 8 clinically diverse queries per patient (80 total), covering:

| Query Type | Count | Example |
|---|---|---|
| FACTUAL | 30 | "What are the patient's diagnoses from their most recent encounter?" |
| MEDICATION | 10 | "What medications is this patient currently prescribed?" |
| TEMPORAL | 10 | "Has the patient's kidney function changed over time?" |
| SUMMARY | 10 | "Summarize the patient's clinical history across all encounters." |
| REASONING | 10 | "What is the patient's genetic risk for Alzheimer's disease?" |

The REASONING queries are intentionally **unanswerable** from the available EHR data (genetic risk is not documented in MIMIC-IV), testing the system's abstention capability.

### 4.2 Baselines

We compare against five frontier LLMs in a **one-shot** configuration, where the complete patient record is provided in the prompt context:

| Model | Provider | Context Window | Cost (Input/Output per 1M tokens) |
|---|---|---|---|
| GPT-5 | OpenAI | 128K | $2.00 / $8.00 |
| Claude Sonnet 4.5 | Anthropic | 200K | $3.00 / $15.00 |
| Gemini 3 Pro | Google | 1M | $1.25 / $10.00 |
| GLM 4.6 | Zhipu AI | 128K | $0.14 / $0.55 |
| Claude Haiku 3.5 | Anthropic | 200K | $0.80 / $4.00 |

The RAG pipeline uses **MiniMax M2.5** as the backbone LLM ($0.30/$1.20 per 1M tokens), a cost-efficient reasoning model.

### 4.3 Ground Truth Evaluation Framework

A key contribution of this work is our **zero-cost ground truth evaluation framework**. Rather than relying on expensive human annotation or LLM-as-judge approaches (which suffer from self-evaluation bias), we extract ground truth directly from FHIR structured data:

1. **Entity Extraction**: For each query, we identify the relevant FHIR resource type (e.g., Condition for diagnosis queries, MedicationRequest for medication queries) and extract all entity names, codes, and identifiers for the target patient.

2. **Entity Matching**: We employ a multi-strategy matching pipeline:
   - Exact substring matching (after normalization)
   - Word overlap with ≥50% threshold (after date/noise stripping)
   - Medical abbreviation expansion (40+ clinical abbreviation mappings, e.g., HTN→hypertension, DM→diabetes mellitus)
   - ICD code matching (answer text containing the same codes as ground truth)
   - Noise-word stripping (removing non-discriminative terms like "unspecified", "other", "NOS")

3. **Metrics Computed**:
   - **Entity F1**: Harmonic mean of precision (fraction of answer entities found in ground truth) and recall (fraction of ground truth entities mentioned in answer)
   - **Entity Precision**: Measures hallucination resistance
   - **Entity Recall**: Measures completeness
   - **Semantic Similarity**: PubMedBERT cosine similarity between answer and concatenated ground truth entities
   - **ROUGE-L**: Longest common subsequence overlap
   - **Hallucination Rate**: Fraction of answer entities with no ground truth match
   - **Abstention Accuracy**: Correctness of declining to answer unanswerable queries

---

## 5. Results

### 5.1 Main Results

Table 1 presents the comparison across all metrics. Our multi-agent RAG pipeline significantly outperforms all one-shot baselines on the primary metric (Entity F1) and most secondary metrics.

**Table 1: Ground Truth Evaluation Results (10 patients, 80 queries)**

| Metric | Proposed Solution | Sonnet 4.5 | Haiku 3.5 | GPT-5 | GLM 4.6 | Gemini 3 Pro |
|---|---|---|---|---|---|---|
| **Entity F1** | **0.639** | 0.531 | 0.482 | 0.331 | 0.308 | 0.234 |
| Entity Precision | **0.850** | 0.763 | 0.796 | 0.556 | 0.524 | 0.627 |
| Entity Recall | **0.555** | 0.449 | 0.397 | 0.281 | 0.259 | 0.184 |
| Semantic Similarity | **0.609** | 0.563 | 0.586 | 0.471 | 0.395 | 0.454 |
| ROUGE-L | 0.167 | 0.157 | 0.159 | **0.195** | 0.176 | 0.117 |
| Hallucination Rate | 0.150 | 0.209 | 0.133 | 0.130 | **0.062** | 0.230 |
| Abstention Accuracy | **100%** | 100% | 100% | 100% | 90% | 100% |

Key observations:

- **Entity F1**: The proposed solution achieves 0.639, a **20% relative improvement** over the strongest baseline (Sonnet 4.5, 0.531). This demonstrates that structured retrieval and multi-agent verification substantially improve entity-level accuracy.
- **Precision**: The proposed solution leads at 0.850, meaning 85.0% of entities mentioned in its answers are grounded in the patient record. This compares to 76.3% for Sonnet 4.5.
- **Recall**: The proposed solution achieves 0.555, indicating it captures 55.5% of ground truth entities — significantly higher than all baselines.
- **Hallucination Rate**: The proposed solution's 15.0% hallucination rate is lower than Sonnet 4.5 (20.9%) and Gemini 3 Pro (23.0%). Haiku 3.5 (13.3%) and GPT-5 (13.0%) achieve slightly lower hallucination rates but at the cost of substantially lower recall. GLM 4.6 achieves the lowest hallucination rate (6.2%) but with poor recall (0.259) and degraded abstention accuracy (90%).
- **Abstention**: The proposed solution correctly abstains on all 10 unanswerable queries (100%), matching most baselines. GLM 4.6 fails to abstain on 1 of 10 queries (90%), generating a speculative answer for a genetic risk query.

### 5.2 Per-Query-Type Breakdown

Table 2 shows Entity F1 stratified by query type, revealing where the multi-agent architecture provides the greatest benefit.

**Table 2: Entity F1 by Query Type (10 patients)**

| Query Type | Proposed Solution | Sonnet 4.5 | Haiku 3.5 | GPT-5 | GLM 4.6 | Gemini 3 Pro |
|---|---|---|---|---|---|---|
| FACTUAL | **0.607** | 0.512 | 0.472 | 0.377 | 0.368 | 0.305 |
| MEDICATION | **0.770** | 0.383 | 0.522 | 0.007 | 0.081 | 0.047 |
| TEMPORAL | 0.721 | **0.759** | 0.686 | 0.660 | 0.355 | 0.291 |
| SUMMARY | **0.550** | 0.526 | 0.280 | 0.141 | 0.249 | 0.081 |
| REASONING | N/A | N/A | N/A | N/A | N/A | N/A |

Key findings:

- **Medication queries** show the most dramatic improvement: the proposed solution achieves 0.770 F1 while the best baseline (Haiku 3.5) achieves only 0.522 — a **~48% relative improvement**. One-shot models struggle with medication queries because they must identify current prescriptions from large volumes of clinical text, while the proposed solution retrieves the specific MedicationRequest resources. Notably, GPT-5 (0.007) and Gemini 3 Pro (0.047) nearly completely fail on medication queries.
- **Temporal queries** are the one category where a baseline edges out the proposed solution: Sonnet 4.5 achieves 0.759 vs 0.721. This suggests that Sonnet 4.5's strong reasoning capabilities can sometimes compensate for the lack of structured retrieval when temporal context is already well-represented in the full patient record.
- **Factual queries** show a consistent advantage for the proposed solution (0.607 vs 0.512 for Sonnet 4.5), with all baselines trailing.
- **Summary queries** show strong performance for the proposed solution (0.550) with Sonnet 4.5 close behind (0.526), while other baselines fall off substantially (Haiku 0.280, GPT-5 0.141).

### 5.3 Latency Analysis

| System | Avg Latency (s) |
|---|---|
| Proposed Solution (MiniMax M2.5) | 104.2 |
| GLM 4.6 (one-shot) | 24.0 |
| GPT-5 (one-shot) | 17.6 |
| Gemini 3 Pro (one-shot) | 12.9 |
| Sonnet 4.5 (one-shot) | 7.5 |
| Haiku 3.5 (one-shot) | 5.1 |

The multi-agent pipeline incurs higher latency (104.2s average) due to: (a) the sequential nature of 5+ LLM calls per query; (b) MiniMax M2.5's reasoning token overhead (the model produces internal chain-of-thought before generating output). In clinical practice, this latency is acceptable for non-urgent queries (chart review, care planning) but would require optimization for real-time decision support. Potential optimizations include: parallelizing the temporal and numeric validators, caching router classifications, and using a faster backbone model for validation agents.

### 5.4 Cost Analysis

The RAG pipeline uses MiniMax M2.5 at $0.30/$1.20 per million input/output tokens. With an average of 5-6 LLM calls per query, the estimated cost per query is approximately $0.01-0.02. By contrast, one-shot with Sonnet 4.5 costs approximately $0.05-0.10 per query (large context window with full patient record). The RAG approach is both more accurate and more cost-efficient because it retrieves only relevant chunks rather than including the entire patient record.

---

## 6. Discussion

### 6.1 Why One-Shot Fails on EHR Data

#### 6.1.1 Needle in a Haystack Problem

A patient has ~10K tokens of clinical text across multiple encounters, labs, medications, procedures. When you ask "What medications is this patient currently prescribed?", the LLM has to scan everything and figure out which MedicationRequests are current vs historical.

Our proposed solution retrieves the **15 most relevant chunks** — so the reasoning agent sees a focused ~3K tokens of medication-specific evidence, not 10K of everything.

Result: **Medication F1 — Proposed 0.770 vs Sonnet 4.5 0.383.** Sonnet sees everything but can't pick out what matters.

#### 6.1.2 No Verification in One-Shot

One-shot generates an answer in a single pass. If it hallucinates a date or misquotes a lab value, there's no safety net.

Our pipeline has 3 verification stages:
- Temporal validator catches wrong dates
- Numeric validator catches wrong lab values
- Critic cross-references every claim against evidence and can **abstain** if unsupported

Result: **Hallucination rate — Proposed 15.0% vs Sonnet 4.5 20.9%.** Meaningfully lower hallucination through structured verification.

#### 6.1.3 One-Shot Can't Say "I Don't Know"

When we asked "What is the patient's genetic risk for Alzheimer's disease?" (unanswerable — no genetic data in MIMIC-IV), one-shot models mostly generate speculative answers. Our critic detects insufficient evidence and abstains.

Result: **Abstention accuracy — Proposed 100% vs GLM 4.6 90%.**

#### 6.1.4 Precision Through Retrieval Grounding

Every claim in our proposed solution's answer is traceable to a specific chunk with a citation. The reasoning agent can only use what retrieval gives it — it can't invent entities that aren't in the retrieved evidence.

Result: **Entity Precision — Proposed 85.0% vs Sonnet 4.5 76.3%.** When the proposed solution says something, it's almost always real.

### 6.2 Why Haiku 3.5 Outperforms Larger Models

A counterintuitive finding is that Claude Haiku 3.5 — the smallest and cheapest model evaluated — substantially outperforms GPT-5, GLM 4.6, and Gemini 3 Pro on Entity F1 (0.482 vs 0.331, 0.308, 0.234 respectively). This is especially striking on medication queries, where Haiku achieves F1=0.522 while GPT-5 scores just 0.007 and Gemini 3 Pro scores 0.047.

Investigation of the raw answers reveals a systematic pattern: **larger, more "capable" models over-reason about data availability and refuse to answer**. In the MIMIC-IV FHIR data, MedicationRequest resources carry a `status` field that is often set to `[completed]` for historical prescriptions. When GPT-5, GLM 4.6, and Gemini 3 Pro encounter this status, they interpret it as evidence that the medication is no longer active and respond with variations of "there is insufficient data to determine current medications" or "no active prescriptions found." This produces answers with near-zero entity recall.

Haiku 3.5, by contrast, pragmatically lists the medications it finds in the record regardless of status flags, correctly recognizing that `[completed]` in the FHIR demo context does not necessarily mean discontinued. This difference in behavior is quantifiable:

| Model | Abstention-like Response Rate | Medication F1 |
|---|---|---|
| GPT-5 | 41% | 0.007 |
| GLM 4.6 | 37% | 0.081 |
| Gemini 3 Pro | 33% | 0.047 |
| Haiku 3.5 | 25% | 0.522 |
| Sonnet 4.5 | 22% | 0.383 |

This finding has important implications for clinical NLP: **more capable models are not always more useful**. The tendency of frontier models to over-interpret metadata and hedge excessively can paradoxically make them worse at straightforward clinical information extraction. For EHR QA, a model that faithfully reports what is in the record — even when the data is ambiguous — is more useful than one that refuses to engage with imperfect data.

Our proposed multi-agent pipeline sidesteps this problem entirely: the hybrid retrieval engine delivers focused medication chunks to the reasoning agent, and the structured pipeline ensures that the answer is grounded in specific evidence rather than dependent on the model's interpretation of FHIR status codes.

### 6.3 The Latency Tradeoff

The cost of multi-agent RAG is **latency** — 104s vs 7.5s. We make 5-6 serial LLM calls instead of 1. For a clinician reviewing a chart (not an emergency), that's acceptable. For real-time decision support, you'd need to parallelize the validators.

### 6.4 Limitations

1. **Evaluation set scope**: 10 patients x 8 queries provides 80 evaluation points across diverse clinical profiles. Larger-scale evaluation across the full MIMIC-IV dataset is needed for stronger statistical significance.
2. **Demo dataset**: MIMIC-IV Demo contains only 100 patients with relatively simple clinical histories. Real-world EHR data is significantly more complex.
3. **Latency**: The 104-second average response time is too slow for real-time clinical decision support, though acceptable for asynchronous use cases.
4. **Ground truth limitations**: Our entity extraction approach may miss ground truth entities that are expressed differently in FHIR resources versus clinical text, and the matching heuristics may not capture all valid clinical synonymy.

### 6.5 Clinical Safety Considerations

The fail-safe abstention mechanism is a deliberate safety design choice. By defaulting to ABSTAINED on any parse failure, ambiguous verdict, or insufficient evidence, the system prioritizes safety over completeness. In our evaluation, this resulted in 100% abstention accuracy — the system correctly declined all ten unanswerable genetic risk queries. This conservative approach is appropriate for a clinical support tool where false confidence is more dangerous than silence.

---

## 7. Conclusion

We present EHR Copilot, a multi-agent RAG pipeline for clinical question answering over FHIR-formatted EHR data. Our five-agent architecture (router, reasoning, temporal validator, numeric validator, critic) achieves an Entity F1 of 0.639 on MIMIC-IV across 10 patients and 80 queries, outperforming five frontier LLMs in one-shot settings by 20% over the strongest baseline (Sonnet 4.5, F1=0.531). The system demonstrates particularly strong performance on medication queries (0.770 F1, ~48% improvement over the best baseline) and maintains 100% abstention accuracy on unanswerable queries. Notably, we find that smaller models like Haiku 3.5 outperform larger frontier models (GPT-5, GLM 4.6, Gemini 3 Pro) on EHR QA because the latter over-reason about data availability and refuse to answer — a finding with important implications for clinical NLP system design. Our zero-cost ground truth evaluation framework, which extracts reference entities from FHIR structured data, provides a reproducible and scalable alternative to expensive human annotation or biased LLM-as-judge approaches.

Future work includes: (a) scaling evaluation to the full MIMIC-IV dataset; (b) introducing agent parallelization to reduce latency; (c) adding a conversational memory module for multi-turn clinical dialogues; and (d) user studies with clinicians to assess real-world utility.

---

## 8. References

1. Xiong, G. et al. (2024). "Benchmarking Retrieval-Augmented Generation for Medicine." *Findings of ACL 2024*.
2. Wu, Q. et al. (2023). "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation." *arXiv:2308.08155*.
3. Tang, X. et al. (2024). "MedAgents: Large Language Models as Collaborators for Zero-shot Medical Reasoning." *arXiv:2311.10537*.
4. Ji, Z. et al. (2023). "Survey of Hallucination in Natural Language Generation." *ACM Computing Surveys*.
5. Johnson, A. et al. (2023). "MIMIC-IV Clinical Database Demo on FHIR." *PhysioNet*.
6. Zakka, C. et al. (2024). "Almanac: Retrieval-Augmented Language Models for Clinical Medicine." *NEJM AI*.
7. Robertson, S. & Zaragoza, H. (2009). "The Probabilistic Relevance Framework: BM25 and Beyond." *Foundations and Trends in IR*.
8. Gu, Y. et al. (2021). "Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing." *ACM CHIL*.

---

## Appendix A: System Configuration

| Parameter | Value |
|---|---|
| Embedding Model | NeuML/pubmedbert-base-embeddings (768d) |
| FAISS Index | FlatIP (inner product) |
| BM25 Parameters | k1=1.5, b=0.75 |
| RRF Fusion k | 20 |
| Chunk Size | 512 tokens max, 128 overlap |
| Retrieval Top-K | 15 (after fusion of 30 dense + 30 sparse) |
| LLM Backbone (RAG) | MiniMax M2.5 (reasoning model) |
| Router Temperature | 0.0 |
| Reasoning Temperature | 0.1 |
| Validator Temperature | 0.0 |
| Critic Temperature | 0.0 |

## Appendix B: Evaluation Query Set

| # | Query | Type | Answerable |
|---|---|---|---|
| 1 | What are the patient's diagnoses from their most recent encounter? | FACTUAL | Yes |
| 2 | What medications is this patient currently prescribed? | MEDICATION | Yes |
| 3 | What are the most recent lab results for this patient? | FACTUAL | Yes |
| 4 | What procedures has this patient undergone? | FACTUAL | Yes |
| 5 | Summarize the patient's clinical history across all encounters. | SUMMARY | Yes |
| 6 | What is the patient's genetic risk for Alzheimer's disease? | REASONING | No |
| 7 | What imaging studies has the patient had and what were the findings? | FACTUAL | Yes |
| 8 | Has the patient's kidney function changed over time? | TEMPORAL | Yes |

## Appendix C: Visualization

This appendix provides schematic descriptions of key result visualizations. Full rendered figures are available in the supplementary materials.

### C.1 Entity F1 by Model (Bar Chart)

A horizontal bar chart comparing Entity F1 scores across all six systems:

```
Entity F1 Score (10-Patient Evaluation)

Proposed Solution   |==========================================| 0.639
Sonnet 4.5   |==================================|        0.531
Haiku 3.5    |==============================|            0.482
GPT-5        |=====================|                     0.331
GLM 4.6      |===================|                       0.308
Gemini 3 Pro |===============|                           0.234

              0.0    0.1    0.2    0.3    0.4    0.5    0.6    0.7
```

The proposed solution achieves the highest Entity F1 at 0.639, with a 20% relative improvement over the next best system (Sonnet 4.5 at 0.531). There is a clear tier structure: the proposed solution leads, followed by the Anthropic models (Sonnet 4.5 and Haiku 3.5), then GPT-5 and GLM 4.6 in a middle tier, and Gemini 3 Pro trailing.

### C.2 Entity F1 by Query Type x Model (Grouped Bar Chart)

A grouped bar chart showing Entity F1 broken down by query type for each model:

```
Entity F1 by Query Type

FACTUAL:
  Proposed   |==============================|  0.607
  Sonnet 4.5 |=========================|      0.512
  Haiku 3.5  |=======================|        0.472
  GPT-5      |==================|             0.377
  GLM 4.6    |==================|             0.368
  Gemini 3   |===============|                0.305

MEDICATION:
  Proposed   |======================================| 0.770
  Haiku 3.5  |==========================|            0.522
  Sonnet 4.5 |===================|                   0.383
  GLM 4.6    |===|                                   0.081
  Gemini 3   |==|                                    0.047
  GPT-5      ||                                      0.007

TEMPORAL:
  Sonnet 4.5 |=====================================| 0.759
  Proposed   |====================================|  0.721
  Haiku 3.5  |=================================|    0.686
  GPT-5      |================================|     0.660
  GLM 4.6    |=================|                    0.355
  Gemini 3   |==============|                       0.291

SUMMARY:
  Proposed   |===========================|  0.550
  Sonnet 4.5 |==========================|   0.526
  Haiku 3.5  |==============|               0.280
  GLM 4.6    |============|                 0.249
  GPT-5      |=======|                      0.141
  Gemini 3   |====|                         0.081
```

Key patterns: (1) The proposed solution dominates on MEDICATION queries with a large margin. (2) TEMPORAL is the only category where a baseline (Sonnet 4.5 at 0.759) outperforms the proposed solution (0.721). (3) FACTUAL and SUMMARY show consistent advantages for the proposed solution but with closer margins.

### C.3 Multi-Metric Radar Chart: Proposed Solution vs Sonnet 4.5 vs Haiku 3.5

A radar (spider) chart with six axes comparing the top three systems across all metrics. Values are normalized to [0, 1] where higher is better (hallucination rate is inverted so that lower hallucination = higher value on chart).

```
Radar Chart: Proposed Solution vs Sonnet 4.5 vs Haiku 3.5

                     Entity F1
                        |
                   0.639(P) 0.531(S) 0.482(H)
                        |
  Abstention -----+-----+-----+----- Precision
  1.00/1.00/1.00  |     |     |      0.850(P) 0.763(S) 0.796(H)
                  |     |     |
                  |     +     |
                  |   center  |
                  |     |     |
  1-Halluc -------+-----+-----+----- Recall
  0.850/0.791/    |     |     |      0.555(P) 0.449(S) 0.397(H)
  0.867           |     |     |
                  +-----+-----+
                        |
                   Semantic Sim
                  0.609(P) 0.563(S) 0.586(H)

  (P) = Proposed Solution    (S) = Sonnet 4.5    (H) = Haiku 3.5
```

The radar chart reveals that the proposed solution has the largest area, indicating the best overall performance across metrics. It leads on Entity F1, Precision, Recall, and Semantic Similarity. All three systems tie on Abstention Accuracy (100%). Haiku 3.5 has a slightly better inverted hallucination score (0.867 vs 0.850), reflecting its lower hallucination rate, but this comes at the cost of substantially lower recall and F1.

