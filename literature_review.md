# Literature Review: Multi-Agent RAG with Citations for EHR Question Answering

**Date:** 2026-02-12
**Scope:** 2024-2026 papers across five research areas

---

## Table of Contents

1. [Clinical QA with RAG](#1-clinical-qa-with-rag)
2. [Multi-Agent Medical AI](#2-multi-agent-medical-ai)
3. [Citation / Grounding in Clinical NLP](#3-citation--grounding-in-clinical-nlp)
4. [MIMIC-Based Evaluation](#4-mimic-based-evaluation)
5. [One-Shot vs RAG Comparison](#5-one-shot-vs-rag-comparison)
6. [Standard Metrics in the Field](#6-standard-metrics-in-the-field)
7. [Key Takeaways for Our System](#7-key-takeaways-for-our-system)

---

## 1. Clinical QA with RAG

### 1.1 MIRAGE Benchmark / MedRAG Toolkit

| Field | Detail |
|-------|--------|
| **Title** | Benchmarking Retrieval-Augmented Generation for Medicine |
| **Authors** | Guangzhi Xiong, Qiao Jin, Zhiyong Lu, Aidong Zhang |
| **Venue/Year** | Findings of ACL 2024 |
| **Method** | Introduces MIRAGE (Medical Information Retrieval-Augmented Generation Evaluation), a comprehensive benchmark, alongside the MedRAG toolkit supporting 5 corpora (PubMed, StatPearls, Textbooks, Wikipedia, MedCorp), 4 retrievers (BM25, Contriever, SPECTER, MedCPT), and 6 LLMs. Retrieves snippets, concatenates them as context, and applies chain-of-thought prompting. |
| **Datasets** | 7,663 questions from 5 datasets: MedQA-US, MedMCQA, MMLU-Med, PubMedQA, BioASQ-Y/N |
| **Key Metrics** | Accuracy (multi-choice); standard deviation across runs |
| **Key Results** | MedRAG improves accuracy of 6 LLMs by up to **18%** over chain-of-thought prompting. Elevates GPT-3.5 and Mixtral to GPT-4-level (~70%). Discovered log-linear scaling with number of retrieved snippets (k <= 32) and a "lost-in-the-middle" U-shaped accuracy effect. Best results from combining multiple corpora and retrievers. |
| **Relevance** | Directly establishes RAG as superior to vanilla prompting for medical QA. Provides the standard benchmark and toolkit our system should evaluate against. The multi-corpus finding supports our design of retrieving from diverse EHR note types. |

**Source:** [ACL Anthology](https://aclanthology.org/2024.findings-acl.372/)

---

### 1.2 i-MedRAG (Iterative Follow-up Queries)

| Field | Detail |
|-------|--------|
| **Title** | Improving Retrieval-Augmented Generation in Medicine with Iterative Follow-up Questions |
| **Authors** | Guangzhi Xiong, Qiao Jin, Xiao Wang, Minjia Zhang, Zhiyong Lu, Aidong Zhang |
| **Venue/Year** | Pacific Symposium on Biocomputing (PSB) 2025, Vol 30, pp. 199-214 |
| **Method** | Extends MedRAG with iterative follow-up query generation. Rather than a single retrieval pass, the LLM reasons through complex medical questions step-by-step, generating contextual queries per iteration based on prior search results. Each follow-up query is answered by a conventional RAG system. |
| **Datasets** | MedQA (USMLE), MMLU-Med (6 subsets); corpora: Textbooks, StatPearls |
| **Key Metrics** | Accuracy |
| **Key Results** | Zero-shot i-MedRAG on GPT-3.5 achieves **69.68%** on MedQA (surpassing all prompt engineering and fine-tuning methods). Llama-3.1-8B reaches **75.02%** with i-MedRAG. Shows flexible reasoning chains via iterative queries. |
| **Relevance** | Our multi-agent pipeline can be seen as a form of iterative retrieval -- the planner agent decomposes questions into sub-queries, similar to i-MedRAG's follow-up mechanism. Validates iterative retrieval as better than single-shot. |

**Source:** [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11997844/)

---

### 1.3 RAG-squared (Rationale-Guided RAG)

| Field | Detail |
|-------|--------|
| **Title** | Rationale-Guided Retrieval Augmented Generation for Medical Question Answering |
| **Authors** | Jiwoong Sohn, Yein Park, Chanwoong Yoon, Sihyeon Park, Hyeon Hwang, Mujeen Sung, Hyunjae Kim, Jaewoo Kang |
| **Venue/Year** | NAACL 2025 (Long Paper) |
| **Method** | RAG-squared introduces three innovations: (1) a small filtering model trained on perplexity-based labels to selectively augment informative snippets while filtering distractors, (2) LLM-generated rationales as queries to improve utility of retrieved snippets, (3) balanced retrieval from 4 biomedical corpora to mitigate retriever bias. |
| **Datasets** | MedQA, MedMCQA, MMLU-Med |
| **Key Metrics** | Accuracy |
| **Key Results** | Improves state-of-the-art LLMs by up to **6.1%**. Outperforms previous best medical RAG (MedRAG) by up to **5.6%** across three benchmarks. |
| **Relevance** | The filtering model for distractor removal is directly relevant to our audit agent's role. Our citation system could adopt similar rationale-based retrieval to improve snippet quality. |

**Source:** [ACL Anthology](https://aclanthology.org/2025.naacl-long.635/)

---

### 1.4 MedRAG (Knowledge Graph-Enhanced, ACM WWW 2025)

| Field | Detail |
|-------|--------|
| **Title** | MedRAG: Enhancing Retrieval-augmented Generation with Knowledge Graph-Elicited Reasoning for Healthcare Copilot |
| **Authors** | (Nanyang Technological University et al.) |
| **Venue/Year** | Proceedings of the ACM Web Conference 2025, pp. 4442-4457 |
| **Method** | Constructs a 4-tier hierarchical diagnostic knowledge graph (KG) encoding critical diagnostic differences between diseases. Dynamically integrates KG information with similar EHRs retrieved from an EHR database, reasoned within an LLM. Also generates follow-up questions for personalized decision-making. |
| **Datasets** | DDXPlus (public differential diagnosis dataset); CPDD (private chronic pain diagnostic dataset from Tan Tock Seng Hospital) |
| **Key Metrics** | Diagnostic accuracy, misdiagnosis rate |
| **Key Results** | Outperforms state-of-the-art models in reducing misdiagnosis rates. KG integration provides more specific diagnostic insights than flat retrieval. |
| **Relevance** | Demonstrates the value of structured medical knowledge (KGs) alongside EHR retrieval. Our system could benefit from incorporating structured clinical ontologies to improve retrieval precision. |

**Source:** [ACM DL](https://dl.acm.org/doi/10.1145/3696410.3714782)

---

### 1.5 MKRAG (Medical Knowledge RAG)

| Field | Detail |
|-------|--------|
| **Title** | MKRAG: Medical Knowledge Retrieval Augmented Generation for Medical Question Answering |
| **Authors** | Shi, Xu et al. |
| **Venue/Year** | AMIA 2024 |
| **Method** | Employs transparent RAG with comprehensive medical fact retrieval from external knowledge bases, injecting them into LLM query prompts without fine-tuning. Evaluates impact of different retrieval models (Contriever, SapBert) and number of facts. |
| **Datasets** | MedQA-SMILE |
| **Key Metrics** | Accuracy |
| **Key Results** | Retrieval-augmented Vicuna-7B improved from **44.46% to 48.54%** accuracy. Contriever slightly outperformed SapBert as retriever. |
| **Relevance** | Shows RAG benefits even for smaller open-source models. Validates that retriever choice matters -- relevant to our system's retriever component selection. |

**Source:** [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12099378/)

---

### 1.6 MEGA-RAG (Multi-Evidence Guided Answer Refinement)

| Field | Detail |
|-------|--------|
| **Title** | MEGA-RAG: A Retrieval-Augmented Generation Framework with Multi-Evidence Guided Answer Refinement for Mitigating Hallucinations of LLMs in Public Health |
| **Authors** | Xu, Yan, Dai, Wu |
| **Venue/Year** | Frontiers in Public Health, 2025 |
| **Method** | Integrates multi-source evidence retrieval (dense via FAISS, sparse via BM25, and biomedical knowledge graphs), uses cross-encoder reranker for semantic relevance, and a discrepancy-aware refinement module for factual accuracy. |
| **Datasets** | Public health QA corpus (custom) |
| **Key Metrics** | Accuracy, Precision, Recall, F1, hallucination rate |
| **Key Results** | Accuracy **0.7913**, Precision **0.7541**, Recall **0.8304**, F1 **0.7904**. Reduces hallucination rate by over **40%** vs baselines (PubMedBERT, PubMedGPT, standalone LLM, standard RAG). |
| **Relevance** | Multi-evidence retrieval with discrepancy-aware refinement is architecturally similar to our multi-agent approach. The combination of dense + sparse + KG retrieval is a validated pattern. |

**Source:** [Frontiers](https://www.frontiersin.org/journals/public-health/articles/10.3389/fpubh.2025.1635381/full)

---

### 1.7 JAMIA Systematic Review and Meta-Analysis

| Field | Detail |
|-------|--------|
| **Title** | Improving Large Language Model Applications in Biomedicine with Retrieval-Augmented Generation: A Systematic Review, Meta-Analysis, and Clinical Development Guidelines |
| **Authors** | Wright, Liu et al. |
| **Venue/Year** | Journal of the American Medical Informatics Association (JAMIA), 2025 |
| **Method** | Systematic review of 20 studies (all published 2024) comparing baseline LLM vs RAG performance in biomedical domains. Meta-analysis with pooled effect sizes. Develops clinical deployment guidelines (GUIDED-RAG). |
| **Datasets** | Aggregates across 20 studies |
| **Key Metrics** | Pooled odds ratio |
| **Key Results** | RAG shows **1.35 odds ratio** improvement over baseline LLMs (95% CI: 1.19-1.53, P = .001). Statistically significant benefit across domains. |
| **Relevance** | Provides meta-analytic evidence that RAG consistently improves clinical LLM performance. The GUIDED-RAG clinical deployment guidelines are directly applicable to our system design. |

**Source:** [Oxford Academic / JAMIA](https://academic.oup.com/jamia/article/32/4/605/7954485)

---

## 2. Multi-Agent Medical AI

### 2.1 MAC Framework (Multi-Agent Conversation for Diagnosis)

| Field | Detail |
|-------|--------|
| **Title** | Enhancing Diagnostic Capability with Multi-Agents Conversational Large Language Models |
| **Authors** | (Multiple authors) |
| **Venue/Year** | npj Digital Medicine, 2025 (published March 2025) |
| **Method** | Multi-Agent Conversation (MAC) framework inspired by clinical Multi-Disciplinary Team (MDT) discussions. Architecture: admin agent (provides patient info), supervisor agent (initiates/supervises), 3 doctor agents (joint discussion). Optimal: 4 doctor agents + 1 supervisor using GPT-4. Consensus-based or max 13 conversation rounds. |
| **Datasets** | 302 rare disease cases, each in primary and follow-up consultation settings |
| **Key Metrics** | Accuracy of most likely diagnosis, possible diagnoses, suggested diagnostic tests |
| **Key Results** | **Primary consultation:** MAC achieves 28% (most likely diagnosis), 47.3% (possible diagnoses), 83.3% (diagnostic tests). **Follow-up:** 48.0% (most likely), 66.7% (possible diagnoses). Outperforms single GPT-4, GPT-3.5, Chain-of-Thought, Self-Refine, and Self-Consistency. |
| **Relevance** | Directly validates multi-agent architecture for clinical tasks. The supervisor-doctor pattern maps to our planner-specialist agent design. Shows multi-agent consensus outperforms single-model approaches. |

**Source:** [npj Digital Medicine](https://www.nature.com/articles/s41746-025-01550-0) / [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11906805/)

---

### 2.2 KG4Diagnosis (Hierarchical Multi-Agent with KG)

| Field | Detail |
|-------|--------|
| **Title** | KG4Diagnosis: A Hierarchical Multi-Agent LLM Framework with Knowledge Graph Enhancement for Medical Diagnosis |
| **Authors** | Kaiwen Zuo, Yirui Jiang, Fan Mo, Pietro Lio |
| **Venue/Year** | arXiv, December 2024 |
| **Method** | Two-tier hierarchical multi-agent architecture mirroring real-world medical systems: (1) GP-LLM agent for initial assessment/triage with confidence threshold tau=0.7, (2) Consultant-LLMs (4 specialized agents: cardiology, neurology, endocrinology, rheumatology). Automated KG construction covering 362 diseases. Multi-agent verification to counter hallucinations. |
| **Datasets** | 362 common diseases; benchmarks under development |
| **Key Metrics** | Diagnostic accuracy, hallucination prevention, multi-agent coordination efficiency |
| **Key Results** | Methodological contribution (no empirical results reported yet). KG constraints reduce hallucinations through multi-agent verification. |
| **Relevance** | Hierarchical GP-to-specialist routing is directly analogous to our planner-to-specialist pipeline. The confidence threshold for escalation is a pattern we can adopt. |

**Source:** [arXiv](https://arxiv.org/abs/2412.16833)

---

### 2.3 Agentic Memory-Augmented Retrieval and Evidence Grounding

| Field | Detail |
|-------|--------|
| **Title** | Agentic Memory-Augmented Retrieval and Evidence Grounding for Medical Question-Answering Tasks |
| **Authors** | Shuyue Jia, Subhrangshu Bit, Varuna H. Jasodanand, Yi Liu, Vijaya B. Kolachalama |
| **Venue/Year** | medRxiv 2025 / International Journal of Medical Informatics |
| **Method** | Unified open-source agentic system integrating: (1) lightweight RAG pipeline (SPECTER retriever + gte-Qwen2-7B-instruct reranker), (2) LLM-based agent (Qwen2.5-72B-Instruct) orchestrating diagnostic workflows with specialized tools, (3) cache-and-prune memory bank for efficient long-context inference. Draws evidence from PubMed, ClinicalTrials.gov, NEJM case reports, medical textbooks, and Wikipedia. |
| **Datasets** | USMLE Steps 1-3, MedQA, MedExpQA |
| **Key Metrics** | Accuracy (multiple-choice); Cosine similarity, BERTScore F1 (open-ended) |
| **Key Results** | USMLE Step 1: **82.98%**, Step 2: **86.24%**, Step 3: **88.52%**, MedQA: **73.29%**, MedExpQA: **78.40%**. Surpasses GPT-4's 80.67% (Step 1) and 81.67% (Step 2). |
| **Relevance** | **Most directly relevant paper.** Combines agentic architecture + RAG + evidence grounding + memory management -- all components of our system. The cache-and-prune memory bank is analogous to our context management strategy. Open-source design aligns with our goals. |

**Source:** [medRxiv](https://www.medrxiv.org/content/10.1101/2025.08.06.25333160v1.full) / [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1386505626000791)

---

## 3. Citation / Grounding in Clinical NLP

### 3.1 MedRAGChecker (Claim-Level Verification)

| Field | Detail |
|-------|--------|
| **Title** | MedRAGChecker: Claim-Level Verification for Biomedical Retrieval-Augmented Generation |
| **Authors** | Yuelyu Ji, Min Gu Kwak, Hang Zhang, Xizhi Wu, Chenyu Li, Yanshan Wang |
| **Venue/Year** | arXiv, January 2025 (University of Pittsburgh) |
| **Method** | Decomposes long-form answers into atomic claims, assigns each a calibrated support confidence by combining: (i) textual NLI verification (supervised by GPT-4.1/4o), and (ii) KG-based support via Drug Repurposing Knowledge Graph (DRKG). Aggregates claim decisions for answer-level diagnostics. Distills teacher supervision into compact biomedical student models. |
| **Datasets** | PubMedQA (1,000), MedQuAD (1,000), TREC LiveQA Medical (104), MedRedQA (799) |
| **Key Metrics** | **Faithfulness** (fraction of Entail claims), **Hallucination** (fraction of Contradict claims), **ClaimRecall** (reference claims supported by evidence), **ContextPrecision** (retrieved passages cited for supported claims), **SafetyError** (contradiction on safety-critical relations) |
| **Key Results** | Med42-Llama3-8B: Faithfulness **85.3%**, Hallucination **6.3%**, Safety Error **6.8%**. PMC-LLaMA-13B worst: Faithfulness **60.1%**, Hallucination **10.7%**. |
| **Relevance** | **Highest relevance for our citation/audit system.** Defines exactly the metrics we need: faithfulness, hallucination rate, claim recall, context precision, safety error. The claim decomposition + NLI verification pattern maps directly to our audit agent. |

**Source:** [arXiv](https://arxiv.org/html/2601.06519)

---

### 3.2 MedTrust-RAG (Evidence Verification and Trust Alignment)

| Field | Detail |
|-------|--------|
| **Title** | MedTrust-RAG: Evidence Verification and Trust Alignment for Biomedical Question Answering |
| **Authors** | Yingpeng Ning, Yuanyuan Sun, Ling Luo, Yanhua Wang, Yuchen Pan, Hongfei Lin |
| **Venue/Year** | arXiv, October 2025 |
| **Method** | Dual-agent system: Verifier agent evaluates medical validity of evidence and generates citation-grounded reasoning; Generator agent produces answers from validated inputs. Enforces citation-aware reasoning with inline citations [Doc j]. Uses structured Negative Knowledge Assertions when evidence is insufficient. Iterative retrieval-verification (max 3 iterations) with Medical Gap Analysis. MedTrust-Align Module (MTAM) with Direct Preference Optimization (DPO). Hybrid retrieval: BM25 + MedCPT + Contriever with Reciprocal Rank Fusion. |
| **Datasets** | MedMCQA, MedQA, MMLU-Med |
| **Key Metrics** | Exact Match (EM); Hallucination analysis: Faulty Reasoning, Over-Refusal, Missing Answer, Misattribution |
| **Key Results** | Average accuracy gains of **2.7%** and **2.4%** over strongest standard RAG baselines. Qwen3-8B (DPO): MedMCQA 63.6%, MedQA 70.1%, MMLU-Med 84.3% (avg 72.6%). |
| **Relevance** | **Directly relevant to our citation architecture.** The dual Verifier-Generator agent pattern, citation-aware reasoning with [Doc j] references, and Negative Knowledge Assertions (abstention) align precisely with our audit agent and citation system. |

**Source:** [arXiv](https://arxiv.org/abs/2510.14400)

---

### 3.3 MedHallu (Hallucination Detection Benchmark)

| Field | Detail |
|-------|--------|
| **Title** | MedHallu: A Comprehensive Benchmark for Detecting Medical Hallucinations in Large Language Models |
| **Authors** | Shrey Pandit, Jiawei Xu, Junyuan Hong, Zhangyang Wang, Tianlong Chen, Kaidi Xu, Ying Ding |
| **Venue/Year** | EMNLP 2025 (Main Conference) |
| **Method** | Constructed 10,000 QA pairs from PubMedQA with controlled hallucinated answers generated via temperature sampling, ensemble quality filtering (GPT-4o mini, Gemma2-9B, Qwen2.5-7B), bidirectional entailment checking (DeBERTa), and TextGrad optimization. Four hallucination categories: Misinterpretation (76%), Incomplete Information (23%), Mechanism Misattribution (11%), Evidence Fabrication (0.5%). |
| **Datasets** | PubMedQA (1,000 annotated + 9,000 artificial split) |
| **Key Metrics** | F1 score (macro-averaged), Precision, per-difficulty analysis |
| **Key Results** | GPT-4o: F1 **0.737** overall, **0.625** on hard cases (zero-shot). With knowledge: F1 **0.877** overall, **0.811** on hard cases. Medical fine-tuned models averaged F1 **0.522** (worse than general LLMs). Adding "not sure" option improved precision by 10-15%. |
| **Relevance** | Establishes the difficulty tiers for hallucination detection. The finding that medically fine-tuned models perform WORSE at detection is critical for our audit agent design -- we should use general-purpose models for verification. |

**Source:** [ACL Anthology / EMNLP 2025](https://aclanthology.org/2025.emnlp-main.143/)

---

### 3.4 Clinical Safety Framework for Hallucination Assessment

| Field | Detail |
|-------|--------|
| **Title** | A Framework to Assess Clinical Safety and Hallucination Rates of LLMs for Medical Text Summarisation |
| **Authors** | Elham Asgari, Nina Montana-Brown, Magda Dubois, Saleh Khalil, Jasmine Balloch, Joshua Au Yeung, Dominic Pimenta |
| **Venue/Year** | npj Digital Medicine, May 2025 |
| **Method** | Four-component framework: (1) Error taxonomy (fabrication, negation, causality, contextual hallucinations; omissions by clinical section), (2) Experimental structure for iterative testing, (3) Clinical safety assessment (likelihood x consequence severity, inspired by medical device certification), (4) CREOLA annotation platform. Tested across 18 experimental configurations. |
| **Datasets** | PriMock (450 consultation transcript-note pairs), ACI Bench |
| **Key Metrics** | Hallucination rate, omission rate, clinical severity classification |
| **Key Results** | Hallucination rate **1.47%** (191/12,999 sentences); Major hallucinations: 44% of total. Omission rate **3.45%** (1,712/49,590 sentences); Major omissions: 16.7%. Best experiments achieved sub-human error rates. Addition of provenance notes improved clinician trust by **12%**. |
| **Relevance** | Provides the clinical safety evaluation framework our system should adopt. The error taxonomy (fabrication, negation, causality, contextual) is directly applicable to our audit agent's classification scheme. Provenance notes align with our citation approach. |

**Source:** [npj Digital Medicine](https://www.nature.com/articles/s41746-025-01670-7)

---

### 3.5 VERIRAG (Statistical Audit for RAG)

| Field | Detail |
|-------|--------|
| **Title** | VERIRAG: Healthcare Claim Verification via Statistical Audit in Retrieval-Augmented Generation |
| **Authors** | (Multiple authors) |
| **Venue/Year** | arXiv, July 2025 |
| **Method** | Post-retrieval auditing framework with three components: (1) Veritable -- 11-point checklist evaluating source methodological rigor (data integrity, statistical validity), (2) Hard-to-Vary (HV) Score -- quantitative aggregator weighting evidence by quality and diversity, (3) Dynamic Acceptance Threshold -- calibrates required evidence based on claim extraordinariness. |
| **Datasets** | Four corpora: retracted science, conflicting science, comprehensive science, settled science |
| **Key Metrics** | F1 score |
| **Key Results** | F1 scores **0.53-0.65** across datasets, representing **10-14 point improvement** over next-best baselines. Over 80% of generated audit trails rated useful by human testers. |
| **Relevance** | **Directly relevant to our audit pipeline.** The statistical auditing approach -- checking source quality, not just text matching -- is a dimension our system should incorporate. The dynamic acceptance threshold concept could improve our abstention mechanism. |

**Source:** [arXiv](https://arxiv.org/abs/2507.17948)

---

### 3.6 RAG-HAT (Hallucination-Aware Tuning)

| Field | Detail |
|-------|--------|
| **Title** | RAG-HAT: A Hallucination-Aware Tuning Pipeline for LLM in Retrieval-Augmented Generation |
| **Authors** | Juntong Song, Xingguang Wang, Juno Zhu, Yuanhao Wu, Xuxin Cheng, Randy Zhong, Cheng Niu |
| **Venue/Year** | EMNLP 2024 (Industry Track) |
| **Method** | Trains hallucination detection models that generate detection labels with detailed descriptions. Uses GPT-4 Turbo to correct detected hallucinations. Creates preference datasets from corrected vs original outputs for Direct Preference Optimization (DPO) training. |
| **Datasets** | Not specified in search results (industry application) |
| **Key Metrics** | Hallucination rate, answer quality |
| **Key Results** | DPO fine-tuning leads to reduced hallucination rates and improved answer quality. |
| **Relevance** | The detect-correct-prefer pipeline could be used to train our audit agent. DPO alignment for hallucination reduction is a proven approach. |

**Source:** [ACL Anthology](https://aclanthology.org/2024.emnlp-industry.113/)

---

### 3.7 MedGraphRAG (Evidence-Based Graph RAG)

| Field | Detail |
|-------|--------|
| **Title** | Medical Graph RAG: Evidence-based Medical Large Language Model via Graph Retrieval-Augmented Generation |
| **Authors** | Junde Wu, Jiayuan Zhu, Yunli Qi, Jingkun Chen, Min Xu, Filippo Menolascina, Yueming Jin, Vicente Grau |
| **Venue/Year** | ACL 2025 (Long Paper), Vienna |
| **Method** | Graph-based RAG framework with Triple Graph Construction and U-Retrieval (combining Top-down Precise Retrieval with Bottom-up Response Refinement). Connects user documents to credible medical sources for evidence-based response generation. |
| **Datasets** | 9 medical QA benchmarks, 2 health fact-checking datasets, 1 long-form generation test set |
| **Key Metrics** | Accuracy, fact-checking scores |
| **Key Results** | Outperforms state-of-the-art models across all evaluated benchmarks while ensuring credible sourcing. |
| **Relevance** | The U-Retrieval strategy (top-down + bottom-up) for connecting responses to credible sources is a pattern our citation system should consider. |

**Source:** [ACL Anthology](https://aclanthology.org/2025.acl-long.1381/)

---

## 4. MIMIC-Based Evaluation

### 4.1 EHRNoteQA

| Field | Detail |
|-------|--------|
| **Title** | EHRNoteQA: An LLM Benchmark for Real-World Clinical Practice Using Discharge Summaries |
| **Authors** | Ji-Youn Kim et al. |
| **Venue/Year** | NeurIPS 2024 (Datasets and Benchmarks Track) |
| **Method** | Constructed 962 QA pairs from MIMIC-IV discharge summaries. Questions generated by GPT-4, then manually reviewed/refined by 3 clinicians. Unique requirement: questions need information from 2+ clinical notes per patient. Available in open-ended and multi-choice formats. 27 LLMs evaluated. |
| **Datasets** | MIMIC-IV EHR (patient records 2008-2019), 962 QA pairs across 8 clinical topics |
| **Key Metrics** | Accuracy (multi-choice), correlation with clinician evaluation |
| **Key Results** | Spearman correlation **0.78**, Kendall correlation **0.62** with clinician-evaluated performance -- higher than other benchmarks. Multi-note questions reveal real-world clinical reasoning challenges. |
| **Relevance** | **Primary evaluation benchmark for our system.** Built on MIMIC-IV, requires multi-note reasoning (matching our multi-agent retrieval), and has validated correlation with clinician judgment. |

**Source:** [NeurIPS 2024](https://neurips.cc/virtual/2024/poster/97643) / [PhysioNet](https://physionet.org/content/ehr-notes-qa-llms/1.0.0/) / [GitHub](https://github.com/ji-youn-kim/EHRNoteQA)

---

### 4.2 EHR QA Scoping Review

| Field | Detail |
|-------|--------|
| **Title** | Question Answering for Electronic Health Records: Scoping Review of Datasets and Models |
| **Authors** | (Multiple authors) |
| **Venue/Year** | Journal of Medical Internet Research (JMIR), October 2024 |
| **Method** | Systematic review of 47 papers (2005-2023) following PRISMA guidelines. Searched Google Scholar, ACL Anthology, ACM Digital Library, PubMed. |
| **Datasets** | Reviews all major EHR QA datasets |
| **Key Metrics** | N/A (review paper) |
| **Key Results** | emrQA is the most cited EHR QA dataset. MIMIC-III and n2c2 are the most popular EHR databases. Key challenges: limited clinical annotations, concept normalization, realistic dataset generation. 53% of papers (n=25) focused on datasets, 79% (n=37) on models. |
| **Relevance** | Provides comprehensive landscape of EHR QA datasets and identifies gaps. Confirms MIMIC as the dominant evaluation platform and emrQA as the standard dataset. |

**Source:** [JMIR](https://www.jmir.org/2024/1/e53636)

---

### 4.3 Revisiting MIMIC-IV Benchmark

| Field | Detail |
|-------|--------|
| **Title** | Revisiting the MIMIC-IV Benchmark: Experiments Using Large Language Models |
| **Authors** | (Multiple authors) |
| **Venue/Year** | CL4Health Workshop 2024 |
| **Method** | Integrated MIMIC-IV data into Hugging Face datasets library. Investigated template-based conversion of EHR tabular data to text. Evaluated fine-tuned and zero-shot LLMs on patient mortality prediction. |
| **Datasets** | MIMIC-IV |
| **Key Metrics** | Mortality prediction accuracy |
| **Key Results** | Fine-tuned text-based models are competitive against robust tabular classifiers. Template design significantly impacts LLM performance on EHR tasks. |
| **Relevance** | Shows that text-based LLM approaches to EHR data are viable. Template design for converting structured EHR to text is relevant to our data preprocessing pipeline. |

**Source:** [ACL Anthology](https://aclanthology.org/2024.cl4health-1.23.pdf)

---

### 4.4 RAG Evaluation on MIMIC (NL2SQL and RAG-QA)

| Field | Detail |
|-------|--------|
| **Title** | (NL2SQL and RAG-QA Evaluation on MIMIC) |
| **Authors** | (Multiple authors) |
| **Venue/Year** | 2024-2025 |
| **Method** | Evaluates two clinical NLP workflows: NL2SQL for EHR querying and RAG-QA for clinical question answering, with privacy-preserving deployment focus. Benchmarks 9 LLMs including DeepSeek V3, Llama-3.3-70B, Qwen2.5-32B, GPT-4o, GPT-5. |
| **Datasets** | MIMICSQL (27,000 generations across 9 models x 3 runs) |
| **Key Metrics** | Execution accuracy (EX) for NL2SQL |
| **Key Results** | Best NL2SQL accuracy: GPT-4o at **66.1%**, GPT-5 at **64.6%**. |
| **Relevance** | Demonstrates the complementary NL2SQL approach for structured EHR querying alongside RAG for unstructured notes. Our system could combine both strategies. |

**Source:** [medRxiv](https://www.medrxiv.org/content/10.64898/2025.12.22.25342863v1.full.pdf)

---

## 5. One-Shot vs RAG Comparison

### 5.1 Radiology RaR (Multi-Step Retrieval vs Zero-Shot)

| Field | Detail |
|-------|--------|
| **Title** | Multi-Step Retrieval and Reasoning Improves Radiology Question Answering with Large Language Models |
| **Authors** | (Multiple authors) |
| **Venue/Year** | npj Digital Medicine, December 2025 |
| **Method** | Radiology Retrieval and Reasoning (RaR): multi-step framework that iteratively summarizes clinical questions, retrieves evidence, and synthesizes answers. Evaluated 25 LLMs spanning 0.5B-670B parameters across general-purpose, reasoning-optimized, and clinically fine-tuned models. |
| **Datasets** | 104 expert-curated radiology questions + 65 real radiology board-exam questions |
| **Key Metrics** | Diagnostic accuracy, hallucination rate |
| **Key Results** | RaR vs zero-shot: **75% vs 67%** (P = 1.1 x 10^-7). RaR vs conventional RAG: **75% vs 69%** (P = 1.9 x 10^-6). Largest gains in mid-sized models (Mistral Large: 72% -> 81%). RaR provided clinically relevant evidence in **46%** of cases. |
| **Relevance** | Strongest evidence that multi-step RAG beats both zero-shot AND single-pass RAG. The iterative retrieve-reason-retrieve pattern validates our multi-agent approach where agents can request additional context. |

**Source:** [npj Digital Medicine](https://www.nature.com/articles/s41746-025-02250-5)

---

### 5.2 MIRAGE Benchmark (RAG vs Chain-of-Thought)

| Field | Detail |
|-------|--------|
| **Title** | (Same as Section 1.1 -- MIRAGE/MedRAG) |
| **Venue/Year** | Findings of ACL 2024 |
| **Key Comparison** | RAG (MedRAG) vs zero-shot Chain-of-Thought across 6 LLMs |
| **Key Results** | RAG improves accuracy by up to **18%** over CoT. GPT-3.5 with MedRAG matches GPT-4 zero-shot performance (~70%). Log-linear scaling with number of retrieved snippets. |
| **Relevance** | Definitive evidence that RAG bridges the gap between smaller and larger models. Our system using RAG with open-source models can potentially match proprietary model performance. |

---

### 5.3 RAG for 10 LLMs (Medical Fitness Assessment)

| Field | Detail |
|-------|--------|
| **Title** | Retrieval Augmented Generation for 10 Large Language Models and Its Generalizability in Assessing Medical Fitness |
| **Authors** | (Multiple authors) |
| **Venue/Year** | npj Digital Medicine, 2025 |
| **Method** | GPT-4 LLM-RAG model with international guidelines as knowledge base. Compared human responses vs RAG-augmented LLM responses. |
| **Datasets** | Medical fitness assessment cases |
| **Key Metrics** | Accuracy |
| **Key Results** | GPT-4-based LLM-RAG with international guidelines achieved highest accuracy, **significantly outperforming human-generated responses**. |
| **Relevance** | Shows RAG with authoritative guideline retrieval can exceed human expert performance. Validates the importance of high-quality retrieval sources. |

**Source:** [npj Digital Medicine](https://www.nature.com/articles/s41746-025-01519-z)

---

### 5.4 RAG Variants for Clinical Decision Support

| Field | Detail |
|-------|--------|
| **Title** | Evaluating Retrieval-Augmented Generation Variants for Clinical Decision Support: Hallucination Mitigation and Secure On-Premises Deployment |
| **Authors** | (Multiple authors) |
| **Venue/Year** | Electronics (MDPI), 2024 |
| **Method** | Tested 12 RAG variants (dense, sparse, hybrid, graph-based, multimodal, self-reflective, adaptive, security-focused) on de-identified patient vignettes. Evaluated hallucination mitigation with retrieval confidence thresholds, chain-of-thought verification, and external fact-checking. |
| **Datasets** | 250 de-identified patient vignettes |
| **Key Metrics** | P@5, nDCG@10, hallucination rate, response latency |
| **Key Results** | Best retrieval: Haystack (DPR + BM25 + cross-encoder) with P@5 >= **0.68**, nDCG@10 >= **0.67**. Self-reflective RAG lowest hallucination at **5.8%**. Sparse retrieval fastest (120ms) but less accurate. |
| **Relevance** | Comprehensive comparison of RAG variants for clinical use. Self-reflective RAG's low hallucination rate validates our audit-agent-as-self-reflection approach. Hybrid retrieval (DPR + BM25 + reranker) confirmed as best pattern. |

**Source:** [MDPI](https://www.mdpi.com/2079-9292/14/21/4227)

---

## 6. Standard Metrics in the Field

Based on the surveyed literature, the following metrics are standard for evaluating clinical RAG systems with citation/grounding:

### 6.1 Answer Quality Metrics

| Metric | Description | Used In |
|--------|-------------|---------|
| **Accuracy** | Exact match on multiple-choice questions | MIRAGE, i-MedRAG, RAG-squared, Agentic RAG, MEGA-RAG |
| **Exact Match (EM)** | Answer exactly matches ground truth | MedTrust-RAG |
| **F1 Score** | Token-level overlap with reference answer | MEGA-RAG, MedHallu, MedRAGChecker |
| **BERTScore F1** | Semantic similarity using BERT embeddings | Agentic Memory-Augmented RAG |
| **Cosine Similarity** | Embedding-based semantic similarity | Agentic Memory-Augmented RAG |
| **ROUGE-L** | Longest common subsequence overlap | MedHallu, general QA |

### 6.2 Faithfulness and Citation Metrics

| Metric | Description | Used In |
|--------|-------------|---------|
| **Faithfulness** | Fraction of generated claims entailed by evidence | MedRAGChecker (primary) |
| **Hallucination Rate** | Fraction of claims contradicting evidence | MedRAGChecker, Clinical Safety Framework, RAG variants |
| **Citation Precision / ContextPrecision** | Fraction of cited passages that actually support claims | MedRAGChecker |
| **ClaimRecall** | Fraction of reference claims supported by retrieved evidence | MedRAGChecker |
| **SafetyError** | Contradiction rate on safety-critical biomedical relations | MedRAGChecker |
| **Omission Rate** | Fraction of relevant information missed from source | Clinical Safety Framework |

### 6.3 Retrieval Quality Metrics

| Metric | Description | Used In |
|--------|-------------|---------|
| **P@K (Precision at K)** | Fraction of top-K retrieved docs that are relevant | RAG Variants evaluation |
| **nDCG@K** | Normalized discounted cumulative gain | RAG Variants evaluation |
| **Context Recall** | Fraction of relevant chunks retrieved | RAG evaluation frameworks |

### 6.4 Hallucination Detection Metrics

| Metric | Description | Used In |
|--------|-------------|---------|
| **Detection F1** | F1 for binary hallucination detection | MedHallu (primary) |
| **Per-Difficulty F1** | F1 stratified by easy/medium/hard hallucinations | MedHallu |
| **Clinical Severity Classification** | Likelihood x consequence (inspired by medical device certification) | Clinical Safety Framework |

### 6.5 Abstention and Calibration Metrics

| Metric | Description | Used In |
|--------|-------------|---------|
| **Abstention Accuracy** | Correctly refusing to answer when evidence is insufficient | MedTrust-RAG (Negative Knowledge Assertions) |
| **Over-Refusal Rate** | Incorrectly refusing when evidence IS sufficient | MedTrust-RAG |
| **Confidence Calibration** | Alignment between model confidence and actual correctness | LLM confidence evaluation studies |

---

## 7. Key Takeaways for Our System

### Architecture Validation

1. **Multi-agent pipelines outperform single models.** MAC framework shows multi-agent consensus beats GPT-4 single-model by significant margins on rare diseases (Section 2.1).
2. **Iterative retrieval beats single-pass RAG.** Both i-MedRAG (Section 1.2) and RaR (Section 5.1) demonstrate that multi-step retrieve-reason-retrieve achieves 6-8% absolute improvement over conventional RAG.
3. **Hybrid retrieval (dense + sparse + reranker) is the confirmed best pattern.** MEGA-RAG, RAG Variants study, and MedTrust-RAG all converge on BM25 + dense retriever + cross-encoder reranker.

### Citation and Audit Design

4. **Claim-level decomposition is the emerging standard** for verification (MedRAGChecker, MedTrust-RAG). Our audit agent should decompose answers into atomic claims and verify each independently.
5. **Use general-purpose LLMs for verification, not medically fine-tuned ones.** MedHallu shows medical fine-tuned models perform WORSE at hallucination detection (F1 0.522 vs 0.737 for GPT-4o).
6. **Negative Knowledge Assertions / structured abstention** when evidence is insufficient is a validated pattern (MedTrust-RAG).
7. **Provenance notes with source IDs improve clinician trust by 12%** (Clinical Safety Framework).

### Evaluation Strategy

8. **Primary benchmarks for our system:**
   - EHRNoteQA (NeurIPS 2024) -- MIMIC-IV-based, multi-note, clinician-validated
   - MIRAGE (ACL 2024) -- Standard medical RAG benchmark (7,663 questions)
   - MedQA/MMLU-Med -- For comparison with literature baselines

9. **Metrics we should report:**
   - **Answer quality:** Accuracy, F1, BERTScore
   - **Faithfulness:** Fraction of entailed claims (per MedRAGChecker)
   - **Citation precision:** Fraction of citations that support their claims
   - **Citation recall / ClaimRecall:** Fraction of claims with valid citations
   - **Hallucination rate:** Fraction of contradicted claims
   - **Safety error rate:** Contradictions on safety-critical relations
   - **Abstention accuracy:** Correct refusal when evidence is insufficient
   - **Retrieval quality:** P@5, nDCG@10

10. **Target baselines from literature:**
    - RAG accuracy on MedQA: ~70% (MedRAG with GPT-3.5) to ~86% (Agentic RAG)
    - Faithfulness: 85% (best, Med42-Llama3-8B per MedRAGChecker)
    - Hallucination rate: <6% (best, Med42-Llama3-8B) to 5.8% (self-reflective RAG)
    - Clinician trust improvement with citations: ~12%

---

## References (Ordered by Section)

1. Xiong et al. "Benchmarking Retrieval-Augmented Generation for Medicine." Findings of ACL, 2024.
2. Xiong et al. "Improving RAG in Medicine with Iterative Follow-up Questions." PSB, 2025.
3. Sohn et al. "Rationale-Guided RAG for Medical QA." NAACL, 2025.
4. (NTU et al.) "MedRAG: KG-Elicited Reasoning for Healthcare Copilot." ACM WWW, 2025.
5. Shi & Xu et al. "MKRAG: Medical Knowledge RAG for Medical QA." AMIA, 2024.
6. Xu et al. "MEGA-RAG: Multi-Evidence Guided Answer Refinement." Frontiers in Public Health, 2025.
7. Wright & Liu et al. "RAG in Biomedicine: Systematic Review and Meta-Analysis." JAMIA, 2025.
8. (Multiple) "Enhancing Diagnostic Capability with Multi-Agents Conversational LLMs." npj Digital Medicine, 2025.
9. Zuo et al. "KG4Diagnosis: Hierarchical Multi-Agent LLM with KG." arXiv, 2024.
10. Jia et al. "Agentic Memory-Augmented Retrieval and Evidence Grounding." medRxiv/IJMI, 2025.
11. Ji et al. "MedRAGChecker: Claim-Level Verification for Biomedical RAG." arXiv, 2025.
12. Ning et al. "MedTrust-RAG: Evidence Verification and Trust Alignment." arXiv, 2025.
13. Pandit et al. "MedHallu: Benchmark for Detecting Medical Hallucinations." EMNLP, 2025.
14. Asgari et al. "Clinical Safety and Hallucination Rates of LLMs." npj Digital Medicine, 2025.
15. (Multiple) "VERIRAG: Healthcare Claim Verification via Statistical Audit." arXiv, 2025.
16. Song et al. "RAG-HAT: Hallucination-Aware Tuning Pipeline." EMNLP Industry, 2024.
17. Wu et al. "Medical Graph RAG: Evidence-based Medical LLM." ACL, 2025.
18. Kim et al. "EHRNoteQA: LLM Benchmark for Clinical Practice." NeurIPS D&B, 2024.
19. (Multiple) "QA for EHRs: Scoping Review of Datasets and Models." JMIR, 2024.
20. (Multiple) "Revisiting the MIMIC-IV Benchmark." CL4Health, 2024.
21. (Multiple) "Multi-Step Retrieval and Reasoning for Radiology QA." npj Digital Medicine, 2025.
22. (Multiple) "RAG for 10 LLMs: Medical Fitness Assessment." npj Digital Medicine, 2025.
23. (Multiple) "Evaluating RAG Variants for Clinical Decision Support." Electronics/MDPI, 2024.
