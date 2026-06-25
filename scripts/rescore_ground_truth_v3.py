#!/usr/bin/env python3
"""Re-score all saved answers using 4 quality axes (v3).

Evaluation axes (inspired by ArchEHR-QA, ACL 2025):
  Axis 1: Factuality   — Entity Precision, Hallucination Rate
  Axis 2: Relevance    — Entity Recall, Entity F1, Semantic Similarity, ROUGE-L
  Axis 3: Grounding    — Context Precision, Context Recall, Citation Accuracy
  Axis 4: Safety       — Abstention Accuracy, Safety-Critical Error Rate

Zero API cost — uses only local computation.

Usage:
    uv run python scripts/rescore_ground_truth_v3.py
"""

from __future__ import annotations

import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MIMIC_DATA_DIR = Path("data/mimic-fhir/mimic-iv-clinical-database-demo-on-fhir-2.1.0/fhir")
RAG_RESULTS_PATH = Path("results/eval_results_10patients_merged.json")
BENCHMARK_RESULTS_PATH = Path("results/multi_model_benchmark_10patients.json")
OUTPUT_PATH = Path("results/ground_truth_eval_v3.json")

EVAL_QUERIES = [
    {"query": "What are the patient's diagnoses from their most recent encounter?", "gt_resource": "conditions"},
    {"query": "What medications is this patient currently prescribed?", "gt_resource": "medications"},
    {"query": "What are the most recent lab results for this patient?", "gt_resource": "labs"},
    {"query": "What procedures has this patient undergone?", "gt_resource": "procedures"},
    {"query": "Summarize the patient's clinical history across all encounters.", "gt_resource": "all"},
    {"query": "What is the patient's genetic risk for Alzheimer's disease?", "gt_resource": None},
    {"query": "What imaging studies has the patient had and what were the findings?", "gt_resource": "procedures"},
    {"query": "Has the patient's kidney function changed over time?", "gt_resource": "labs_kidney"},
]

# Safety-critical resource types: medication errors are the most dangerous
SAFETY_CRITICAL_RESOURCES = {"medications"}

# Kidney-related lab codes
KIDNEY_LAB_NAMES = {"creatinine", "bun", "urea nitrogen", "gfr", "glomerular",
                    "egfr", "cystatin", "uric acid", "potassium", "bicarbonate",
                    "anion gap", "phosphate", "calcium"}

# Medical abbreviation → expansion mapping for fuzzy matching
MEDICAL_SYNONYMS = {
    "htn": "hypertension",
    "dm": "diabetes mellitus",
    "dm2": "diabetes mellitus type 2",
    "t2dm": "diabetes mellitus type 2",
    "ckd": "chronic kidney disease",
    "chf": "congestive heart failure",
    "cad": "coronary artery disease",
    "copd": "chronic obstructive pulmonary disease",
    "afib": "atrial fibrillation",
    "a fib": "atrial fibrillation",
    "mi": "myocardial infarction",
    "dvt": "deep vein thrombosis",
    "pe": "pulmonary embolism",
    "uti": "urinary tract infection",
    "gerd": "gastroesophageal reflux",
    "bph": "benign prostatic hyperplasia",
    "ra": "rheumatoid arthritis",
    "osa": "obstructive sleep apnea",
    "esrd": "end stage renal disease",
    "aki": "acute kidney injury",
    "ards": "acute respiratory distress",
    "sirs": "systemic inflammatory response",
    "tia": "transient ischemic attack",
    "cva": "cerebrovascular accident",
    "pna": "pneumonia",
    "acs": "acute coronary syndrome",
    "nstemi": "non st elevation myocardial infarction",
    "stemi": "st elevation myocardial infarction",
    "cabg": "coronary artery bypass",
    "pci": "percutaneous coronary intervention",
    "iv": "intravenous",
    "po": "oral",
    "prn": "as needed",
    "bid": "twice daily",
    "tid": "three times daily",
    "qid": "four times daily",
    "high blood pressure": "hypertension",
    "high cholesterol": "hypercholesterolemia",
    "blood sugar": "glucose",
    "kidney function": "creatinine",
    "liver function": "bilirubin",
    "blood thinner": "anticoagulant",
}

# Noise words to strip for softer matching
NOISE_WORDS = {"unspecified", "other", "nos", "not otherwise specified",
               "without mention of", "not stated as", "or unspecified",
               "uncontrolled", "controlled", "type", "acute", "chronic",
               "primary", "secondary", "acquired", "personal history of",
               "history of", "status", "code", "finding"}


# ---------------------------------------------------------------------------
# FHIR Ground Truth Extraction (unchanged from v2)
# ---------------------------------------------------------------------------

def read_ndjson_gz(filepath: Path) -> list[dict]:
    records = []
    if not filepath.exists():
        return records
    with gzip.open(filepath, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_patient_ref(resource: dict) -> str | None:
    ref = resource.get("subject", {}).get("reference", "")
    if ref.startswith("Patient/"):
        return ref.split("/")[1]
    return None


def extract_encounter_ref(resource: dict) -> str | None:
    ref = resource.get("encounter", {}).get("reference", "")
    if ref.startswith("Encounter/"):
        return ref.split("/")[1]
    return None


class FHIRGroundTruth:
    """Extracts ground truth entities from FHIR structured data."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._patients = {}
        self._encounters = {}
        self._conditions = {}
        self._medications = {}
        self._labs = {}
        self._procedures = {}
        self._med_lookup = {}
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        print("Loading FHIR resources...")

        # Medications lookup
        for med in read_ndjson_gz(self.data_dir / "MimicMedication.ndjson.gz"):
            med_id = med.get("id", "")
            name = ""
            for ident in med.get("identifier", []):
                if "medication-name" in ident.get("system", ""):
                    name = ident.get("value", "")
                    break
            if not name:
                for coding in med.get("code", {}).get("coding", []):
                    if coding.get("display"):
                        name = coding["display"]
                        break
            if not name:
                name = med.get("code", {}).get("text", f"Unknown-{med_id[:8]}")
            self._med_lookup[med_id] = name

        # Patients
        for p in read_ndjson_gz(self.data_dir / "MimicPatient.ndjson.gz"):
            self._patients[p["id"]] = p

        # Encounters
        enc_by_patient = defaultdict(list)
        for fname in ["MimicEncounter.ndjson.gz", "MimicEncounterED.ndjson.gz", "MimicEncounterICU.ndjson.gz"]:
            for enc in read_ndjson_gz(self.data_dir / fname):
                pid = extract_patient_ref(enc)
                if pid:
                    enc_by_patient[pid].append(enc)
        for pid in enc_by_patient:
            enc_by_patient[pid].sort(key=lambda e: e.get("period", {}).get("start", ""), reverse=True)
        self._encounters = dict(enc_by_patient)

        # Conditions
        cond_by_patient = defaultdict(list)
        for fname in ["MimicCondition.ndjson.gz", "MimicConditionED.ndjson.gz"]:
            for c in read_ndjson_gz(self.data_dir / fname):
                pid = extract_patient_ref(c)
                if pid:
                    coding = c.get("code", {}).get("coding", [{}])[0]
                    display = coding.get("display", "")
                    code = coding.get("code", "")
                    enc_ref = extract_encounter_ref(c)
                    if display:
                        cond_by_patient[pid].append({
                            "display": display, "code": code,
                            "encounter_id": enc_ref,
                        })
        self._conditions = dict(cond_by_patient)

        # Medications
        med_by_patient = defaultdict(list)
        for mr in read_ndjson_gz(self.data_dir / "MimicMedicationRequest.ndjson.gz"):
            pid = extract_patient_ref(mr)
            if pid:
                med_ref = mr.get("medicationReference", {}).get("reference", "")
                med_id = med_ref.split("/")[-1] if "/" in med_ref else ""
                name = self._med_lookup.get(med_id, "")
                if not name:
                    name = mr.get("medicationCodeableConcept", {}).get("text", "")
                if name:
                    med_by_patient[pid].append({
                        "name": name,
                        "status": mr.get("status", ""),
                        "encounter_id": extract_encounter_ref(mr),
                    })
        self._medications = dict(med_by_patient)

        # Labs
        lab_by_patient = defaultdict(list)
        for obs in read_ndjson_gz(self.data_dir / "MimicObservationLabevents.ndjson.gz"):
            pid = extract_patient_ref(obs)
            if pid:
                display = obs.get("code", {}).get("coding", [{}])[0].get("display", "")
                vq = obs.get("valueQuantity", {})
                dt = obs.get("effectiveDateTime", "")
                if display:
                    lab_by_patient[pid].append({
                        "display": display,
                        "value": vq.get("value"),
                        "unit": vq.get("unit", ""),
                        "date": dt,
                        "encounter_id": extract_encounter_ref(obs),
                    })
        self._labs = dict(lab_by_patient)

        # Procedures
        proc_by_patient = defaultdict(list)
        for fname in ["MimicProcedure.ndjson.gz", "MimicProcedureED.ndjson.gz", "MimicProcedureICU.ndjson.gz"]:
            for p in read_ndjson_gz(self.data_dir / fname):
                pid = extract_patient_ref(p)
                if pid:
                    display = p.get("code", {}).get("coding", [{}])[0].get("display", "")
                    if display:
                        proc_by_patient[pid].append({
                            "display": display,
                            "date": p.get("performedDateTime", ""),
                            "encounter_id": extract_encounter_ref(p),
                        })
        self._procedures = dict(proc_by_patient)

        self._loaded = True
        print(f"  Loaded: {len(self._patients)} patients, "
              f"{sum(len(v) for v in self._conditions.values())} conditions, "
              f"{sum(len(v) for v in self._medications.values())} medications, "
              f"{sum(len(v) for v in self._labs.values())} labs, "
              f"{sum(len(v) for v in self._procedures.values())} procedures")

    def get_ground_truth(self, patient_id: str, resource_type: str | None) -> dict:
        self.load()

        if resource_type is None:
            return {"entities": [], "codes": []}

        if resource_type == "conditions":
            encounters = self._encounters.get(patient_id, [])
            conditions = self._conditions.get(patient_id, [])
            if encounters:
                most_recent_enc_id = encounters[0]["id"]
                recent = [c for c in conditions if c["encounter_id"] == most_recent_enc_id]
                if recent:
                    return {
                        "entities": [c["display"] for c in recent],
                        "codes": [c["code"] for c in recent if c["code"]],
                    }
            return {
                "entities": list(set(c["display"] for c in conditions)),
                "codes": list(set(c["code"] for c in conditions if c["code"])),
            }

        elif resource_type == "medications":
            meds = self._medications.get(patient_id, [])
            return {"entities": list(set(m["name"] for m in meds)), "codes": []}

        elif resource_type == "labs":
            labs = self._labs.get(patient_id, [])
            latest = {}
            for lab in labs:
                name = lab["display"]
                if name not in latest or lab["date"] > latest[name]["date"]:
                    latest[name] = lab
            entities = []
            entity_names = []
            for v in latest.values():
                entity_names.append(v["display"])
                if v["value"] is not None:
                    entities.append(f"{v['display']} {v['value']} {v['unit']}")
                else:
                    entities.append(v["display"])
            return {"entities": entities, "codes": [], "entity_names": entity_names}

        elif resource_type == "labs_kidney":
            labs = self._labs.get(patient_id, [])
            kidney = [l for l in labs if any(k in l["display"].lower() for k in KIDNEY_LAB_NAMES)]
            entities = []
            entity_names = []
            for l in kidney:
                entity_names.append(l["display"])
                if l["value"] is not None:
                    entities.append(f"{l['display']} {l['value']} {l['unit']}")
                else:
                    entities.append(l["display"])
            return {"entities": entities, "codes": [], "entity_names": entity_names}

        elif resource_type == "procedures":
            procs = self._procedures.get(patient_id, [])
            return {"entities": list(set(p["display"] for p in procs)), "codes": []}

        elif resource_type == "all":
            entities = set()
            entities.update(c["display"] for c in self._conditions.get(patient_id, []))
            entities.update(m["name"] for m in self._medications.get(patient_id, []))
            entities.update(p["display"] for p in self._procedures.get(patient_id, []))
            entities.update(l["display"] for l in self._labs.get(patient_id, []))
            codes = list(set(c["code"] for c in self._conditions.get(patient_id, []) if c["code"]))
            return {"entities": list(entities), "codes": codes}

        return {"entities": [], "codes": []}


# ---------------------------------------------------------------------------
# Text normalization and matching (unchanged from v2)
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_entity(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(?:on\s+)?\d{4}[-/]\d{2}[-/]\d{2}\b", "", text)
    text = re.sub(r"\b\d{4}\b", "", text)
    text = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_noise(text: str) -> str:
    normed = normalize(text)
    for noise in NOISE_WORDS:
        normed = normed.replace(noise, " ")
    return re.sub(r"\s+", " ", normed).strip()


def expand_abbreviations(text: str) -> str:
    normed = normalize(text)
    for abbr, expansion in MEDICAL_SYNONYMS.items():
        pattern = r'\b' + re.escape(abbr) + r'\b'
        normed = re.sub(pattern, expansion, normed)
    return normed


def entity_matches(entity_a: str, entity_b: str) -> bool:
    na = normalize_entity(entity_a)
    nb = normalize_entity(entity_b)

    if na == nb:
        return True
    if na in nb or nb in na:
        return True

    words_a = set(na.split())
    words_b = set(nb.split())
    if words_a and words_b:
        overlap = len(words_a & words_b)
        min_len = min(len(words_a), len(words_b))
        if min_len <= 2:
            if overlap >= min_len:
                return True
        else:
            if overlap / min_len >= 0.5:
                return True

    sa = strip_noise(entity_a)
    sb = strip_noise(entity_b)
    if sa and sb and (sa in sb or sb in sa):
        return True
    if sa and sb:
        sw_a = set(sa.split())
        sw_b = set(sb.split())
        if sw_a and sw_b:
            overlap = len(sw_a & sw_b)
            min_len = min(len(sw_a), len(sw_b))
            if min_len > 0 and overlap / min_len >= 0.5:
                return True

    ea = expand_abbreviations(entity_a)
    eb = expand_abbreviations(entity_b)
    if ea != na or eb != nb:
        if ea in eb or eb in ea:
            return True
        ew_a = set(ea.split())
        ew_b = set(eb.split())
        if ew_a and ew_b:
            overlap = len(ew_a & ew_b)
            min_len = min(len(ew_a), len(ew_b))
            if min_len > 0 and overlap / min_len >= 0.5:
                return True

    return False


def entity_mentioned_in_text(entity: str, text: str) -> bool:
    norm_entity = normalize_entity(entity)
    norm_text = normalize(text)

    if norm_entity in norm_text:
        return True

    entity_words = set(norm_entity.split())
    if not entity_words:
        return False
    matching = sum(1 for w in entity_words if w in norm_text)
    if len(entity_words) <= 2:
        if matching >= len(entity_words):
            return True
    else:
        if matching / len(entity_words) >= 0.5:
            return True

    stripped = strip_noise(entity)
    if stripped and len(stripped) > 3:
        stripped_text = strip_noise(text)
        if stripped in stripped_text:
            return True
        sw = set(stripped.split())
        if sw:
            matching = sum(1 for w in sw if w in stripped_text)
            if len(sw) <= 2:
                if matching >= len(sw):
                    return True
            elif matching / len(sw) >= 0.5:
                return True

    expanded = expand_abbreviations(entity)
    if expanded != norm_entity:
        expanded_text = expand_abbreviations(text)
        if expanded in expanded_text:
            return True

    return False


# ---------------------------------------------------------------------------
# Entity extraction from answers (unchanged from v2)
# ---------------------------------------------------------------------------

def extract_answer_entities(answer_text: str) -> list[str]:
    entities = []

    list_patterns = [
        r"(?:^|\n)\s*\d+[\.\)]\s*(.+?)(?:\[.*?\])?\s*(?:\n|$)",
        r"(?:^|\n)\s*[-•–]\s*(.+?)(?:\[.*?\])?\s*(?:\n|$)",
        r"(?:^|\n)\s*\*\s+(.+?)(?:\[.*?\])?\s*(?:\n|$)",
    ]
    for pattern in list_patterns:
        matches = re.findall(pattern, answer_text, re.MULTILINE)
        for m in matches:
            cleaned = _clean_entity(m)
            if cleaned:
                entities.append(cleaned)

    for intro in [r"including\s+", r"such as\s+", r"consists? of\s+", r"diagnos(?:es|ed with)\s+"]:
        m = re.search(intro + r"(.+?)(?:\.|$)", answer_text, re.IGNORECASE)
        if m:
            items = re.split(r",\s*(?:and\s+)?|;\s*", m.group(1))
            for item in items:
                cleaned = _clean_entity(item)
                if cleaned:
                    entities.append(cleaned)

    kv_matches = re.findall(r"(?:^|\n)\s*([A-Z][^:\n]{2,40}):\s*(\S.+?)(?:\n|$)", answer_text)
    for key, val in kv_matches:
        cleaned = _clean_entity(key.strip())
        if cleaned and not any(skip in cleaned.lower() for skip in
                               ["question", "answer", "note", "summary", "instruction",
                                "source", "reference", "finding", "result", "date"]):
            entities.append(cleaned)

    seen = set()
    unique = []
    for e in entities:
        ne = normalize(e)
        if ne not in seen and len(ne) > 2:
            seen.add(ne)
            unique.append(e)

    return unique


def _clean_entity(text: str) -> str | None:
    cleaned = re.sub(r"\[.*?\]", "", text).strip()
    cleaned = re.sub(r"\((?:ICD|code).*?\)", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s*:$", "", cleaned).strip()
    cleaned = re.sub(r"\s*\((?:completed|active|stopped|final|routine)\)\s*$", "",
                     cleaned, flags=re.IGNORECASE).strip()

    if len(cleaned) < 3:
        return None
    if cleaned.lower().startswith(("the ", "this ", "note:", "based on", "there ", "no ", "n/a")):
        return None
    return cleaned


# ---------------------------------------------------------------------------
# Core Metrics (shared computations)
# ---------------------------------------------------------------------------

def compute_entity_metrics(answer_text: str, gt_entities: list[str], gt_codes: list[str],
                           gt_entity_names: list[str] | None = None) -> dict:
    """Compute entity-level precision, recall, F1, hallucination count."""
    if not gt_entities:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "found": 0, "total_gt": 0, "hallucinated": 0, "total_answer_entities": 0}

    # --- RECALL ---
    found = 0
    found_gt = set()
    for idx, entity in enumerate(gt_entities):
        if idx in found_gt:
            continue
        if entity_mentioned_in_text(entity, answer_text):
            found += 1
            found_gt.add(idx)
            continue
        if gt_entity_names and idx < len(gt_entity_names):
            name_only = gt_entity_names[idx]
            if name_only and entity_mentioned_in_text(name_only, answer_text):
                found += 1
                found_gt.add(idx)
                continue
        if gt_codes and idx < len(gt_codes):
            code = gt_codes[idx]
            if code and len(code) >= 3 and code in answer_text:
                found += 1
                found_gt.add(idx)
                continue

    recall = found / len(gt_entities) if gt_entities else 0

    # --- PRECISION ---
    answer_entities = extract_answer_entities(answer_text)
    if not answer_entities:
        precision = 1.0 if found > 0 else 0.0
        hallucinated = 0
        total_ae = 0
    else:
        matched = 0
        for ae in answer_entities:
            for gt in gt_entities:
                if entity_matches(ae, gt):
                    matched += 1
                    break
            else:
                if gt_entity_names:
                    for gtn in gt_entity_names:
                        if entity_matches(ae, gtn):
                            matched += 1
                            break

        precision = matched / len(answer_entities) if answer_entities else 0
        hallucinated = len(answer_entities) - matched
        total_ae = len(answer_entities)

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "found": found,
        "total_gt": len(gt_entities),
        "hallucinated": max(0, hallucinated),
        "total_answer_entities": total_ae,
    }


def rouge_l(answer_text: str, reference_text: str) -> float:
    answer_tokens = normalize(answer_text).split()
    ref_tokens = normalize(reference_text).split()
    if not answer_tokens or not ref_tokens:
        return 0.0
    m, n = min(len(ref_tokens), 2000), min(len(answer_tokens), 2000)
    ref_tokens = ref_tokens[:m]
    answer_tokens = answer_tokens[:n]
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == answer_tokens[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    lcs_len = prev[n]
    prec = lcs_len / n if n > 0 else 0
    rec = lcs_len / m if m > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return round(f1, 4)


def compute_semantic_similarity(answer_texts: list[str], reference_texts: list[str]) -> list[float]:
    try:
        from ehr_copilot.config import get_settings
        from ehr_copilot.indexing.embedding import EmbeddingModel
        settings = get_settings()
        model = EmbeddingModel(settings.embedding)
        print("  Computing semantic similarity with PubMedBERT...")
        scores = []
        batch_size = 16
        for i in range(0, len(answer_texts), batch_size):
            batch_a = answer_texts[i:i + batch_size]
            batch_r = reference_texts[i:i + batch_size]
            a_embs = model.encode(batch_a)
            r_embs = model.encode(batch_r)
            for a_emb, r_emb in zip(a_embs, r_embs):
                sim = float(np.dot(a_emb, r_emb))
                scores.append(round(max(0.0, sim), 4))
        return scores
    except Exception as e:
        print(f"  [WARN] Semantic similarity failed: {e}")
        return [-1.0] * len(answer_texts)


# ---------------------------------------------------------------------------
# Axis 3: Grounding — Context Precision / Recall / Citation Accuracy
# ---------------------------------------------------------------------------

def compute_grounding_metrics(result: dict, gt_entities: list[str],
                              gt_entity_names: list[str] | None = None) -> dict:
    """Compute grounding metrics from citation data.

    Context Precision: What fraction of retrieved chunks contain at least one GT entity?
    Context Recall: What fraction of GT entities appear in at least one retrieved chunk?
    Citation Accuracy: What fraction of citations have evidence that supports their claim?
    """
    citations = result.get("citations", [])
    if not citations and not gt_entities:
        return {"context_precision": 0.0, "context_recall": 0.0, "citation_accuracy": 0.0}

    # Collect all evidence texts from citations
    evidence_texts = []
    for cit in citations:
        for span in cit.get("evidence_spans", []):
            text = span.get("text", "")
            if text:
                evidence_texts.append(text)

    combined_evidence = " ".join(evidence_texts)

    # Context Precision: fraction of evidence chunks containing a GT entity
    if evidence_texts:
        chunks_with_gt = 0
        for ev in evidence_texts:
            for gt_e in gt_entities:
                if entity_mentioned_in_text(gt_e, ev):
                    chunks_with_gt += 1
                    break
            else:
                if gt_entity_names:
                    for gtn in gt_entity_names:
                        if entity_mentioned_in_text(gtn, ev):
                            chunks_with_gt += 1
                            break
        context_precision = chunks_with_gt / len(evidence_texts)
    else:
        context_precision = 0.0

    # Context Recall: fraction of GT entities found in at least one evidence chunk
    if gt_entities and evidence_texts:
        gt_found = 0
        for idx, gt_e in enumerate(gt_entities):
            if entity_mentioned_in_text(gt_e, combined_evidence):
                gt_found += 1
                continue
            if gt_entity_names and idx < len(gt_entity_names):
                name_only = gt_entity_names[idx]
                if name_only and entity_mentioned_in_text(name_only, combined_evidence):
                    gt_found += 1
                    continue
        context_recall = gt_found / len(gt_entities)
    else:
        context_recall = 0.0

    # Citation Accuracy: fraction of citations where claim text matches evidence
    if citations:
        accurate_cits = 0
        for cit in citations:
            claim = cit.get("claim_text", "")
            if not claim:
                continue
            cit_evidence = " ".join(
                span.get("text", "") for span in cit.get("evidence_spans", [])
            )
            if cit_evidence and entity_mentioned_in_text(claim, cit_evidence):
                accurate_cits += 1
            elif not cit_evidence:
                # No evidence at all — cannot verify, count as inaccurate
                pass
        citation_accuracy = accurate_cits / len(citations)
    else:
        citation_accuracy = 0.0

    return {
        "context_precision": round(context_precision, 4),
        "context_recall": round(context_recall, 4),
        "citation_accuracy": round(citation_accuracy, 4),
    }


# ---------------------------------------------------------------------------
# Axis 4: Safety — Abstention Accuracy + Safety-Critical Error Rate
# ---------------------------------------------------------------------------

def compute_safety_critical_hallucination(answer_text: str, gt_entities: list[str],
                                          gt_codes: list[str],
                                          gt_entity_names: list[str] | None = None) -> dict:
    """Compute hallucination rate specifically for safety-critical (medication) queries.

    This is reported separately because a hallucinated medication is far more
    dangerous than a hallucinated diagnosis.
    """
    metrics = compute_entity_metrics(answer_text, gt_entities, gt_codes, gt_entity_names)
    if metrics["total_answer_entities"] > 0:
        safety_error_rate = metrics["hallucinated"] / metrics["total_answer_entities"]
    else:
        safety_error_rate = 0.0
    return {
        "safety_hallucinated": metrics["hallucinated"],
        "safety_total_entities": metrics["total_answer_entities"],
        "safety_error_rate": round(safety_error_rate, 4),
    }


# ---------------------------------------------------------------------------
# Load saved results
# ---------------------------------------------------------------------------

def load_rag_results() -> list[dict]:
    if not RAG_RESULTS_PATH.exists():
        print(f"  [SKIP] {RAG_RESULTS_PATH} not found")
        return []
    with open(RAG_RESULTS_PATH) as f:
        data = json.load(f)
    results = []
    for pr in data["patient_results"]:
        pid = pr["patient_id"]
        for i, r in enumerate(pr.get("rag_results", [])):
            if "error" in r:
                continue
            results.append({
                "model": "Proposed Solution",
                "patient_id": pid,
                "query_index": i,
                "query": r.get("query", EVAL_QUERIES[i]["query"] if i < len(EVAL_QUERIES) else ""),
                "answer_text": r.get("answer_text", ""),
                "latency_ms": r.get("latency_ms", 0),
                "verdict": r.get("verdict", ""),
                "citations": r.get("citations", []),
            })
    return results


def load_benchmark_results() -> list[dict]:
    if not BENCHMARK_RESULTS_PATH.exists():
        print(f"  [SKIP] {BENCHMARK_RESULTS_PATH} not found")
        return []
    with open(BENCHMARK_RESULTS_PATH) as f:
        data = json.load(f)
    results = []
    for model_name, model_results in data.get("raw_results", {}).items():
        for i, r in enumerate(model_results):
            if "error" in r:
                continue
            results.append({
                "model": model_name,
                "patient_id": r.get("patient_id", ""),
                "query_index": i % len(EVAL_QUERIES),
                "query": r.get("query", ""),
                "answer_text": r.get("answer_text", ""),
                "latency_ms": r.get("latency_ms", 0),
                "citations": [],  # benchmark models don't have citations
            })
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("GROUND TRUTH EVALUATION v3 — 4-Axis Quality Framework")
    print("  Axes: Factuality | Relevance | Grounding | Safety")
    print("  Ref:  ArchEHR-QA (BioNLP @ ACL 2025)")
    print("=" * 70)

    gt = FHIRGroundTruth(MIMIC_DATA_DIR)
    gt.load()

    print("\nLoading saved results...")
    rag_results = load_rag_results()
    benchmark_results = load_benchmark_results()
    all_results = rag_results + benchmark_results
    print(f"  Proposed Solution results: {len(rag_results)}")
    print(f"  Benchmark results: {len(benchmark_results)}")
    print(f"  Total: {len(all_results)}")

    if not all_results:
        print("No results to evaluate!")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Score each result
    # ------------------------------------------------------------------
    print("\nScoring across 4 quality axes...")
    answer_texts_for_sim = []
    reference_texts_for_sim = []

    for r in all_results:
        qi = r["query_index"]
        if qi >= len(EVAL_QUERIES):
            r["entity_metrics"] = {}
            r["grounding_metrics"] = {}
            r["safety_metrics"] = {}
            r["rouge_l"] = 0.0
            continue

        eq = EVAL_QUERIES[qi]
        gt_resource = eq.get("gt_resource")
        gt_data = gt.get_ground_truth(r["patient_id"], gt_resource)
        gt_ents = gt_data["entities"]
        gt_codes = gt_data.get("codes", [])
        gt_names = gt_data.get("entity_names")
        r["gt_entity_count"] = len(gt_ents)

        # --- Unanswerable query (Alzheimer's genetics — no GT) ---
        if gt_resource is None:
            answer_lower = r["answer_text"].lower()
            abstained = any(p in answer_lower for p in [
                "insufficient data", "not available", "no information",
                "cannot determine", "no data", "not found", "no evidence",
                "not possible", "cannot be determined", "no genetic",
                "does not contain", "not present", "not included",
            ])
            r["entity_metrics"] = {"precision": 0, "recall": 0, "f1": 0, "abstained": abstained}
            r["grounding_metrics"] = {"context_precision": 0, "context_recall": 0, "citation_accuracy": 0}
            r["safety_metrics"] = {}
            r["rouge_l"] = 0.0
            r["hallucination_rate"] = 0.0
            continue

        # Axis 1 & 2: Entity metrics (precision=factuality, recall/f1=relevance)
        em = compute_entity_metrics(r["answer_text"], gt_ents, gt_codes, gt_names)
        r["entity_metrics"] = em

        # Hallucination rate (Axis 1: Factuality)
        if em["total_answer_entities"] > 0:
            r["hallucination_rate"] = round(em["hallucinated"] / em["total_answer_entities"], 4)
        else:
            r["hallucination_rate"] = 0.0

        # Axis 2: ROUGE-L (Relevance)
        ref_text = " ".join(gt_ents)
        r["rouge_l"] = rouge_l(r["answer_text"], ref_text)

        # Axis 3: Grounding
        r["grounding_metrics"] = compute_grounding_metrics(r, gt_ents, gt_names)

        # Axis 4: Safety-critical hallucination (medication queries only)
        if gt_resource in SAFETY_CRITICAL_RESOURCES:
            r["safety_metrics"] = compute_safety_critical_hallucination(
                r["answer_text"], gt_ents, gt_codes, gt_names
            )
        else:
            r["safety_metrics"] = {}

        # Collect for semantic similarity
        answer_texts_for_sim.append(r["answer_text"][:1500])
        reference_texts_for_sim.append(ref_text[:1500])

    # Semantic similarity in batch (Axis 2: Relevance)
    if answer_texts_for_sim:
        sim_scores = compute_semantic_similarity(answer_texts_for_sim, reference_texts_for_sim)
        sim_idx = 0
        for r in all_results:
            qi = r["query_index"]
            if qi < len(EVAL_QUERIES) and EVAL_QUERIES[qi].get("gt_resource") is not None:
                r["semantic_sim"] = sim_scores[sim_idx]
                sim_idx += 1
            else:
                r["semantic_sim"] = -1.0

    # ------------------------------------------------------------------
    # Aggregate by model
    # ------------------------------------------------------------------
    print("\nAggregating metrics by model across 4 axes...")
    model_results = defaultdict(list)
    for r in all_results:
        model_results[r["model"]].append(r)

    model_metrics = {}
    for model_name, results in model_results.items():
        answerable = [r for r in results if r.get("entity_metrics", {}).get("abstained") is None]
        unanswerable = [r for r in results if r.get("entity_metrics", {}).get("abstained") is not None]

        # Collect raw values
        precs = [r["entity_metrics"]["precision"] for r in answerable
                 if r.get("entity_metrics", {}).get("precision") is not None]
        recs = [r["entity_metrics"]["recall"] for r in answerable
                if r.get("entity_metrics", {}).get("recall") is not None]
        f1s = [r["entity_metrics"]["f1"] for r in answerable
               if r.get("entity_metrics", {}).get("f1") is not None]
        halluc = [r["hallucination_rate"] for r in answerable
                  if r.get("hallucination_rate") is not None]
        rouges = [r["rouge_l"] for r in answerable if r.get("rouge_l", 0) > 0]
        sims = [r["semantic_sim"] for r in answerable if r.get("semantic_sim", -1) >= 0]
        lats = [r["latency_ms"] for r in results if r.get("latency_ms", 0) > 0]

        # Grounding
        ctx_precs = [r["grounding_metrics"]["context_precision"] for r in answerable
                     if r.get("grounding_metrics", {}).get("context_precision") is not None]
        ctx_recs = [r["grounding_metrics"]["context_recall"] for r in answerable
                    if r.get("grounding_metrics", {}).get("context_recall") is not None]
        cit_accs = [r["grounding_metrics"]["citation_accuracy"] for r in answerable
                    if r.get("grounding_metrics", {}).get("citation_accuracy") is not None]

        # Safety
        safety_results = [r for r in answerable if r.get("safety_metrics", {}).get("safety_error_rate") is not None]
        safety_rates = [r["safety_metrics"]["safety_error_rate"] for r in safety_results]

        correct_abs = sum(1 for r in unanswerable if r.get("entity_metrics", {}).get("abstained"))

        model_metrics[model_name] = {
            "count": len(results),
            "answerable_count": len(answerable),
            # Axis 1: Factuality
            "entity_precision": round(np.mean(precs), 4) if precs else 0,
            "hallucination_rate": round(np.mean(halluc), 4) if halluc else 0,
            # Axis 2: Relevance
            "entity_recall": round(np.mean(recs), 4) if recs else 0,
            "entity_f1": round(np.mean(f1s), 4) if f1s else 0,
            "semantic_sim": round(np.mean(sims), 4) if sims else 0,
            "rouge_l": round(np.mean(rouges), 4) if rouges else 0,
            # Axis 3: Grounding
            "context_precision": round(np.mean(ctx_precs), 4) if ctx_precs else 0,
            "context_recall": round(np.mean(ctx_recs), 4) if ctx_recs else 0,
            "citation_accuracy": round(np.mean(cit_accs), 4) if cit_accs else 0,
            # Axis 4: Safety
            "abstention_accuracy": round(correct_abs / len(unanswerable), 3) if unanswerable else 0,
            "safety_error_rate": round(np.mean(safety_rates), 4) if safety_rates else 0,
            # Operational
            "avg_latency_ms": round(np.mean(lats)) if lats else 0,
        }

    # ------------------------------------------------------------------
    # Print 4-axis report
    # ------------------------------------------------------------------
    model_order = sorted(model_metrics.keys(),
                         key=lambda x: (0 if x.startswith("Proposed") else 1, x))

    print(f"\n{'=' * 130}")
    print("4-AXIS QUALITY EVALUATION (ArchEHR-QA Framework)")
    print(f"{'=' * 130}")

    # Helper to print a section
    def print_axis(title: str, rows: list[tuple[str, str, str]]):
        print(f"\n  {title}")
        print(f"  {'-' * (len(title) + 4)}")
        for label, key, fmt in rows:
            row = f"    {label:<24}"
            for m in model_order:
                val = model_metrics[m].get(key, 0)
                row += f" | {fmt.format(val):>15}"
            print(row)

    # Column headers
    header = f"    {'Metric':<24}"
    for m in model_order:
        header += f" | {m:>15}"
    print(header)

    # Axis 1: Factuality
    print_axis("AXIS 1: FACTUALITY — Are the generated facts correct?", [
        ("Entity Precision", "entity_precision", "{:.4f}"),
        ("Hallucination Rate", "hallucination_rate", "{:.4f}"),
    ])

    # Axis 2: Relevance
    print_axis("AXIS 2: RELEVANCE — Does the answer cover the ground truth?", [
        ("Entity Recall", "entity_recall", "{:.4f}"),
        ("Entity F1", "entity_f1", "{:.4f}"),
        ("Semantic Similarity", "semantic_sim", "{:.4f}"),
        ("ROUGE-L", "rouge_l", "{:.4f}"),
    ])

    # Axis 3: Grounding
    print_axis("AXIS 3: GROUNDING — Are claims traceable to source evidence?", [
        ("Context Precision", "context_precision", "{:.4f}"),
        ("Context Recall", "context_recall", "{:.4f}"),
        ("Citation Accuracy", "citation_accuracy", "{:.4f}"),
    ])

    # Axis 4: Safety
    print_axis("AXIS 4: SAFETY — Does the system avoid harmful errors?", [
        ("Abstention Accuracy", "abstention_accuracy", "{:.1%}"),
        ("Safety Error Rate", "safety_error_rate", "{:.4f}"),
        ("Avg Latency (ms)", "avg_latency_ms", "{:.0f}"),
    ])

    print(f"\n{'=' * 130}")

    # ------------------------------------------------------------------
    # Compact summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print("COMPACT SUMMARY (per-axis averages)")
    print(f"{'=' * 80}")

    for m in model_order:
        mm = model_metrics[m]
        factuality_avg = np.mean([mm["entity_precision"], 1 - mm["hallucination_rate"]])
        relevance_avg = np.mean([mm["entity_recall"], mm["entity_f1"],
                                 mm["semantic_sim"] if mm["semantic_sim"] > 0 else 0,
                                 mm["rouge_l"]])
        grounding_avg = np.mean([mm["context_precision"], mm["context_recall"],
                                 mm["citation_accuracy"]])
        safety_avg = np.mean([mm["abstention_accuracy"],
                              1 - mm["safety_error_rate"]])

        print(f"\n  {m}:")
        print(f"    Factuality:  {factuality_avg:.3f}  (precision={mm['entity_precision']:.3f}, 1-halluc={1-mm['hallucination_rate']:.3f})")
        print(f"    Relevance:   {relevance_avg:.3f}  (recall={mm['entity_recall']:.3f}, F1={mm['entity_f1']:.3f}, sim={mm['semantic_sim']:.3f}, rouge={mm['rouge_l']:.3f})")
        print(f"    Grounding:   {grounding_avg:.3f}  (ctx_prec={mm['context_precision']:.3f}, ctx_rec={mm['context_recall']:.3f}, cit_acc={mm['citation_accuracy']:.3f})")
        print(f"    Safety:      {safety_avg:.3f}  (abstention={mm['abstention_accuracy']:.3f}, 1-safety_err={1-mm['safety_error_rate']:.3f})")

    print(f"\n{'=' * 80}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "evaluation_framework": "4-Axis Quality Framework (v3)",
        "reference": "ArchEHR-QA (BioNLP @ ACL 2025), MedRAGChecker (2025)",
        "axes": {
            "1_factuality": {
                "description": "Are the generated facts correct?",
                "metrics": ["Entity Precision", "Hallucination Rate"],
            },
            "2_relevance": {
                "description": "Does the answer cover the ground truth?",
                "metrics": ["Entity Recall", "Entity F1", "Semantic Similarity", "ROUGE-L"],
            },
            "3_grounding": {
                "description": "Are claims traceable to source evidence?",
                "metrics": ["Context Precision", "Context Recall", "Citation Accuracy"],
            },
            "4_safety": {
                "description": "Does the system avoid harmful errors?",
                "metrics": ["Abstention Accuracy", "Safety Error Rate (medication hallucinations)"],
            },
        },
        "api_cost": "$0.00",
        "model_metrics": model_metrics,
        "per_result": [
            {k: v for k, v in r.items() if k not in ("answer_text", "citations")}
            for r in all_results
        ],
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
