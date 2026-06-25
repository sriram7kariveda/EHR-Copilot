#!/usr/bin/env python3
"""Re-score all saved answers using ground-truth entities from FHIR structured data.

Zero API cost — uses only local computation:
  - Entity F1 (precision/recall of clinical entities against FHIR ground truth)
  - Semantic Similarity (PubMedBERT cosine similarity)
  - ROUGE-L (longest common subsequence overlap)
  - Hallucination Rate (fraction of answer entities NOT in ground truth)

Improvements over v1:
  - Medical synonym/abbreviation matching (HTN→hypertension, DM→diabetes, etc.)
  - ICD code matching (answer mentions code "4019" → matches GT with that code)
  - Better entity extraction from prose (handles colons, commas, "including" lists)
  - Noise-word stripping ("unspecified", "other", "NOS") for softer matching
  - Separate lab-name-only matching (don't penalize for missing exact values)

Usage:
    uv run python scripts/rescore_ground_truth.py
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
OUTPUT_PATH = Path("results/ground_truth_eval.json")

EVAL_QUERIES = [
    {"query": "What are the patient's diagnoses from their most recent encounter?", "type": "FACTUAL", "gt_resource": "conditions"},
    {"query": "What medications is this patient currently prescribed?", "type": "MEDICATION", "gt_resource": "medications"},
    {"query": "What are the most recent lab results for this patient?", "type": "FACTUAL", "gt_resource": "labs"},
    {"query": "What procedures has this patient undergone?", "type": "FACTUAL", "gt_resource": "procedures"},
    {"query": "Summarize the patient's clinical history across all encounters.", "type": "SUMMARY", "gt_resource": "all"},
    {"query": "What is the patient's genetic risk for Alzheimer's disease?", "type": "REASONING", "gt_resource": None},
    {"query": "What imaging studies has the patient had and what were the findings?", "type": "FACTUAL", "gt_resource": "procedures"},
    {"query": "Has the patient's kidney function changed over time?", "type": "TEMPORAL", "gt_resource": "labs_kidney"},
]

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
# FHIR Ground Truth Extraction
# ---------------------------------------------------------------------------

def read_ndjson_gz(filepath: Path) -> list[dict]:
    """Read a gzipped NDJSON file into a list of dicts."""
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
        self._conditions_with_codes = {}  # patient_id -> [{display, code, encounter_id}]
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

        # Conditions (keep codes for ICD matching)
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
        """Get ground truth entities + ICD codes for a patient and resource type.

        Returns dict with 'entities' (display names) and 'codes' (ICD codes).
        """
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
            # Return both full strings and just names (for softer matching)
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
# Text normalization and matching
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_entity(text: str) -> str:
    """Normalize an entity for matching: lowercase, strip dates, punctuation."""
    text = text.lower()
    # Remove dates like "2136-04-11", "on 2136-04-11", "2136"
    text = re.sub(r"\b(?:on\s+)?\d{4}[-/]\d{2}[-/]\d{2}\b", "", text)
    text = re.sub(r"\b\d{4}\b", "", text)  # standalone years
    text = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", "", text)  # times
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_noise(text: str) -> str:
    """Remove common medical noise words for softer matching."""
    normed = normalize(text)
    for noise in NOISE_WORDS:
        normed = normed.replace(noise, " ")
    return re.sub(r"\s+", " ", normed).strip()


def expand_abbreviations(text: str) -> str:
    """Expand medical abbreviations in text."""
    normed = normalize(text)
    for abbr, expansion in MEDICAL_SYNONYMS.items():
        # Word-boundary match to avoid partial replacements
        pattern = r'\b' + re.escape(abbr) + r'\b'
        normed = re.sub(pattern, expansion, normed)
    return normed


def entity_matches(entity_a: str, entity_b: str) -> bool:
    """Check if two clinical entities refer to the same thing.

    Uses multi-strategy matching with date-stripped normalization.
    """
    na = normalize_entity(entity_a)
    nb = normalize_entity(entity_b)

    # 1. Exact match
    if na == nb:
        return True

    # 2. Substring containment (either direction)
    if na in nb or nb in na:
        return True

    # 3. Word overlap (>=50% of the shorter entity's words)
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

    # 4. Noise-stripped match
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

    # 5. Abbreviation-expanded match
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


# ---------------------------------------------------------------------------
# Entity extraction from answers (improved)
# ---------------------------------------------------------------------------

def extract_answer_entities(answer_text: str) -> list[str]:
    """Extract clinical entity mentions from an answer.

    Handles:
    - Numbered lists: "1. Acute pancreatitis [5770]"
    - Bulleted lists: "- Pure hypercholesterolemia"
    - Starred lists: "* Colostomy status"
    - Colon-separated: "Diagnosis: acute pancreatitis"
    - Comma lists after "including": "...including diabetes, hypertension, and anemia"
    - Pipe/semicolon separated items
    """
    entities = []

    # Pattern 1: Numbered/bulleted list items
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

    # Pattern 2: "including X, Y, and Z" or "such as X, Y, Z"
    for intro in [r"including\s+", r"such as\s+", r"consists? of\s+", r"diagnos(?:es|ed with)\s+"]:
        m = re.search(intro + r"(.+?)(?:\.|$)", answer_text, re.IGNORECASE)
        if m:
            items = re.split(r",\s*(?:and\s+)?|;\s*", m.group(1))
            for item in items:
                cleaned = _clean_entity(item)
                if cleaned:
                    entities.append(cleaned)

    # Pattern 3: "Key: Value" pairs (common in lab results)
    kv_matches = re.findall(r"(?:^|\n)\s*([A-Z][^:\n]{2,40}):\s*(\S.+?)(?:\n|$)", answer_text)
    for key, val in kv_matches:
        cleaned = _clean_entity(key.strip())
        if cleaned and not any(skip in cleaned.lower() for skip in
                               ["question", "answer", "note", "summary", "instruction",
                                "source", "reference", "finding", "result", "date"]):
            entities.append(cleaned)

    # Deduplicate
    seen = set()
    unique = []
    for e in entities:
        ne = normalize(e)
        if ne not in seen and len(ne) > 2:
            seen.add(ne)
            unique.append(e)

    return unique


def _clean_entity(text: str) -> str | None:
    """Clean an extracted entity string."""
    cleaned = re.sub(r"\[.*?\]", "", text).strip()
    cleaned = re.sub(r"\((?:ICD|code).*?\)", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s*:$", "", cleaned).strip()
    # Remove trailing metadata like "(completed)", "(active)"
    cleaned = re.sub(r"\s*\((?:completed|active|stopped|final|routine)\)\s*$", "",
                     cleaned, flags=re.IGNORECASE).strip()

    if len(cleaned) < 3:
        return None
    if cleaned.lower().startswith(("the ", "this ", "note:", "based on", "there ", "no ", "n/a")):
        return None
    return cleaned


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def entity_mentioned_in_text(entity: str, text: str) -> bool:
    """Check if a ground truth entity is mentioned in the answer text.

    This is asymmetric: we check if the entity appears IN the text,
    not whether the text appears in the entity.
    """
    norm_entity = normalize_entity(entity)
    norm_text = normalize(text)

    # 1. Exact substring
    if norm_entity in norm_text:
        return True

    # 2. Word overlap: >=50% of entity words found in text
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

    # 3. Noise-stripped: strip filler words then check
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

    # 4. Abbreviation expansion
    expanded = expand_abbreviations(entity)
    if expanded != norm_entity:
        expanded_text = expand_abbreviations(text)
        if expanded in expanded_text:
            return True

    return False


def entity_f1(answer_text: str, gt_entities: list[str], gt_codes: list[str],
              gt_entity_names: list[str] | None = None) -> dict:
    """Compute entity-level precision, recall, F1 with improved matching."""
    if not gt_entities:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "found": 0, "total_gt": 0, "hallucinated": 0, "total_answer_entities": 0}

    norm_answer = normalize(answer_text)

    # --- RECALL: how many GT entities are found in the answer? ---
    found = 0
    found_gt = set()
    for idx, entity in enumerate(gt_entities):
        if idx in found_gt:
            continue

        # Strategy A: Check if entity is mentioned in the answer
        if entity_mentioned_in_text(entity, answer_text):
            found += 1
            found_gt.add(idx)
            continue

        # Strategy B: Match entity name only (without values, for labs)
        if gt_entity_names and idx < len(gt_entity_names):
            name_only = gt_entity_names[idx]
            if name_only and entity_mentioned_in_text(name_only, answer_text):
                found += 1
                found_gt.add(idx)
                continue

        # Strategy C: Check if the ICD code appears in the answer
        if gt_codes and idx < len(gt_codes):
            code = gt_codes[idx]
            if code and len(code) >= 3 and code in answer_text:
                found += 1
                found_gt.add(idx)
                continue

    recall = found / len(gt_entities) if gt_entities else 0

    # --- PRECISION: of the entities extracted from the answer, how many match GT? ---
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
                # Also check against entity names (for labs)
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
    """Compute ROUGE-L F1 score using longest common subsequence."""
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
    """Compute semantic similarity using PubMedBERT embeddings."""
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
                "model": "RAG (Ours)",
                "patient_id": pid,
                "query_index": i,
                "query": r.get("query", EVAL_QUERIES[i]["query"] if i < len(EVAL_QUERIES) else ""),
                "answer_text": r.get("answer_text", ""),
                "latency_ms": r.get("latency_ms", 0),
                "verdict": r.get("verdict", ""),
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
            })
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("GROUND TRUTH RE-EVALUATION v2 (Zero API Cost)")
    print("=" * 70)

    gt = FHIRGroundTruth(MIMIC_DATA_DIR)
    gt.load()

    print("\nLoading saved results...")
    rag_results = load_rag_results()
    benchmark_results = load_benchmark_results()
    all_results = rag_results + benchmark_results
    print(f"  RAG results: {len(rag_results)}")
    print(f"  Benchmark results: {len(benchmark_results)}")
    print(f"  Total: {len(all_results)}")

    if not all_results:
        print("No results to evaluate!")
        sys.exit(1)

    # Score each result
    print("\nScoring with ground truth entities (improved matching)...")
    answer_texts_for_sim = []
    reference_texts_for_sim = []

    for r in all_results:
        qi = r["query_index"]
        if qi >= len(EVAL_QUERIES):
            r["entity_f1"] = {}
            r["rouge_l"] = 0.0
            continue

        eq = EVAL_QUERIES[qi]
        gt_resource = eq.get("gt_resource")
        gt_data = gt.get_ground_truth(r["patient_id"], gt_resource)
        gt_ents = gt_data["entities"]
        gt_codes = gt_data.get("codes", [])
        gt_names = gt_data.get("entity_names")
        r["gt_entity_count"] = len(gt_ents)

        if gt_resource is None:
            answer_lower = r["answer_text"].lower()
            abstained = any(p in answer_lower for p in [
                "insufficient data", "not available", "no information",
                "cannot determine", "no data", "not found", "no evidence",
                "not possible", "cannot be determined", "no genetic",
                "does not contain", "not present", "not included",
            ])
            r["entity_f1"] = {"precision": 0, "recall": 0, "f1": 0, "abstained": abstained}
            r["rouge_l"] = 0.0
            r["hallucination_rate"] = 0.0
            continue

        # Entity F1 with improved matching
        ef1 = entity_f1(r["answer_text"], gt_ents, gt_codes, gt_names)
        r["entity_f1"] = ef1

        # Hallucination rate
        if ef1["total_answer_entities"] > 0:
            r["hallucination_rate"] = round(ef1["hallucinated"] / ef1["total_answer_entities"], 4)
        else:
            r["hallucination_rate"] = 0.0

        # ROUGE-L
        ref_text = " ".join(gt_ents)
        r["rouge_l"] = rouge_l(r["answer_text"], ref_text)

        # Collect for semantic similarity
        answer_texts_for_sim.append(r["answer_text"][:1500])
        reference_texts_for_sim.append(ref_text[:1500])

    # Semantic similarity in batch
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

    # Aggregate by model
    print("\nAggregating metrics by model...")
    model_results = defaultdict(list)
    for r in all_results:
        model_results[r["model"]].append(r)

    model_metrics = {}
    for model_name, results in model_results.items():
        answerable = [r for r in results if r.get("entity_f1", {}).get("abstained") is None]
        unanswerable = [r for r in results if r.get("entity_f1", {}).get("abstained") is not None]

        f1s = [r["entity_f1"]["f1"] for r in answerable if r.get("entity_f1", {}).get("f1") is not None]
        precs = [r["entity_f1"]["precision"] for r in answerable if r.get("entity_f1", {}).get("precision") is not None]
        recs = [r["entity_f1"]["recall"] for r in answerable if r.get("entity_f1", {}).get("recall") is not None]
        rouges = [r["rouge_l"] for r in answerable if r.get("rouge_l", 0) > 0]
        sims = [r["semantic_sim"] for r in answerable if r.get("semantic_sim", -1) >= 0]
        halluc = [r["hallucination_rate"] for r in answerable if r.get("hallucination_rate") is not None]
        lats = [r["latency_ms"] for r in results if r.get("latency_ms", 0) > 0]

        correct_abs = sum(1 for r in unanswerable if r.get("entity_f1", {}).get("abstained"))

        model_metrics[model_name] = {
            "count": len(results),
            "answerable_count": len(answerable),
            "entity_f1": round(np.mean(f1s), 4) if f1s else 0,
            "entity_precision": round(np.mean(precs), 4) if precs else 0,
            "entity_recall": round(np.mean(recs), 4) if recs else 0,
            "rouge_l": round(np.mean(rouges), 4) if rouges else 0,
            "semantic_sim": round(np.mean(sims), 4) if sims else 0,
            "hallucination_rate": round(np.mean(halluc), 4) if halluc else 0,
            "abstention_accuracy": round(correct_abs / len(unanswerable), 3) if unanswerable else 0,
            "avg_latency_ms": round(np.mean(lats)) if lats else 0,
        }

    # Print comparison table
    model_order = sorted(model_metrics.keys(), key=lambda x: (0 if "RAG" in x else 1, x))

    print(f"\n{'=' * 130}")
    print("GROUND TRUTH EVALUATION v2: Entity F1 / Semantic Sim / ROUGE-L / Hallucination Rate")
    print(f"{'=' * 130}")

    header = f"{'Metric':<22}"
    for m in model_order:
        header += f" | {m:>15}"
    print(header)
    print("-" * len(header))

    rows = [
        ("Entity F1", "entity_f1", "{:.4f}"),
        ("Entity Precision", "entity_precision", "{:.4f}"),
        ("Entity Recall", "entity_recall", "{:.4f}"),
        ("ROUGE-L", "rouge_l", "{:.4f}"),
        ("Semantic Sim", "semantic_sim", "{:.4f}"),
        ("Hallucination Rate", "hallucination_rate", "{:.4f}"),
        ("Abstention Acc.", "abstention_accuracy", "{:.1%}"),
        ("Avg Latency (ms)", "avg_latency_ms", "{:.0f}"),
    ]

    for label, key, fmt in rows:
        row = f"{label:<22}"
        for m in model_order:
            val = model_metrics[m].get(key, 0)
            row += f" | {fmt.format(val):>15}"
        print(row)

    print(f"{'=' * 130}")

    # Per-query-type breakdown
    print(f"\n{'=' * 100}")
    print("PER-QUERY-TYPE BREAKDOWN (Entity F1)")
    print(f"{'=' * 100}")

    query_types = sorted(set(eq["type"] for eq in EVAL_QUERIES))
    header2 = f"{'Query Type':<15}"
    for m in model_order:
        header2 += f" | {m:>15}"
    print(header2)
    print("-" * len(header2))

    for qt in query_types:
        row = f"{qt:<15}"
        for m in model_order:
            rft = [r for r in model_results[m]
                   if r["query_index"] < len(EVAL_QUERIES) and
                   EVAL_QUERIES[r["query_index"]]["type"] == qt and
                   r.get("entity_f1", {}).get("f1") is not None and
                   r.get("entity_f1", {}).get("abstained") is None]
            if rft:
                avg_f1 = np.mean([r["entity_f1"]["f1"] for r in rft])
                row += f" | {avg_f1:>15.4f}"
            else:
                row += f" | {'N/A':>15}"
        print(row)
    print(f"{'=' * 100}")

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "evaluation_method": "Ground Truth Entity Extraction from FHIR Structured Data (v2)",
        "improvements": [
            "Medical synonym/abbreviation matching",
            "ICD code matching in answer text",
            "Better entity extraction (prose, comma-lists, colon-pairs)",
            "Noise-word stripping for softer matching",
            "Separate lab-name matching (without exact values)",
        ],
        "metrics_computed": ["Entity F1", "Entity Precision", "Entity Recall",
                             "ROUGE-L", "Semantic Similarity (PubMedBERT)", "Hallucination Rate"],
        "api_cost": "$0.00",
        "model_metrics": model_metrics,
        "per_result": [{k: v for k, v in r.items() if k != "answer_text"} for r in all_results],
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
