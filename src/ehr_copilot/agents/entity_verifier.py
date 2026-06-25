"""Post-generation entity verifier -- strips hallucinated entities from answers.

Deterministic (zero LLM cost) step that extracts entities from the final answer
and checks each one against the retrieved evidence chunks. Entities that cannot
be traced to any chunk are removed from the answer text.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Noise words and abbreviations (subset of rescore_ground_truth.py)
# ---------------------------------------------------------------------------

NOISE_WORDS = [
    "unspecified", "other", "not otherwise specified", "nos", "primary",
    "secondary", "benign", "malignant", "chronic", "acute", "history of",
    "personal history", "status", "type", "without", "with", "due to",
    "initial encounter", "subsequent encounter", "sequela",
]

MEDICAL_SYNONYMS: dict[str, str] = {
    "htn": "hypertension", "dm": "diabetes mellitus", "cad": "coronary artery disease",
    "chf": "congestive heart failure", "copd": "chronic obstructive pulmonary disease",
    "ckd": "chronic kidney disease", "aki": "acute kidney injury",
    "mi": "myocardial infarction", "afib": "atrial fibrillation",
    "dvt": "deep vein thrombosis", "pe": "pulmonary embolism",
    "uti": "urinary tract infection", "gerd": "gastroesophageal reflux disease",
    "bph": "benign prostatic hyperplasia", "tia": "transient ischemic attack",
    "cva": "cerebrovascular accident", "esrd": "end stage renal disease",
    "osa": "obstructive sleep apnea", "ra": "rheumatoid arthritis",
    "sle": "systemic lupus erythematosus", "ibs": "irritable bowel syndrome",
    "gfr": "glomerular filtration rate", "egfr": "estimated glomerular filtration rate",
    "bun": "blood urea nitrogen", "hba1c": "hemoglobin a1c",
    "wbc": "white blood cell", "rbc": "red blood cell", "hgb": "hemoglobin",
    "hct": "hematocrit", "plt": "platelet", "inr": "international normalized ratio",
    "bp": "blood pressure", "hr": "heart rate",
}


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_entity(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(?:on\s+)?\d{4}[-/]\d{2}[-/]\d{2}\b", "", text)
    text = re.sub(r"\b\d{4}\b", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _expand_abbreviations(text: str) -> str:
    normed = _normalize(text)
    for abbr, expansion in MEDICAL_SYNONYMS.items():
        normed = re.sub(r'\b' + re.escape(abbr) + r'\b', expansion, normed)
    return normed


# ---------------------------------------------------------------------------
# Entity extraction (mirrors rescore_ground_truth.py logic)
# ---------------------------------------------------------------------------

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


def extract_entities(text: str) -> list[str]:
    """Extract clinical entity mentions from answer text."""
    entities: list[str] = []

    list_patterns = [
        r"(?:^|\n)\s*\d+[\.\)]\s*(.+?)(?:\[.*?\])?\s*(?:\n|$)",
        r"(?:^|\n)\s*[-•–]\s*(.+?)(?:\[.*?\])?\s*(?:\n|$)",
        r"(?:^|\n)\s*\*\s+(.+?)(?:\[.*?\])?\s*(?:\n|$)",
    ]
    for pattern in list_patterns:
        for m in re.findall(pattern, text, re.MULTILINE):
            cleaned = _clean_entity(m)
            if cleaned:
                entities.append(cleaned)

    for intro in [r"including\s+", r"such as\s+", r"consists? of\s+", r"diagnos(?:es|ed with)\s+"]:
        m = re.search(intro + r"(.+?)(?:\.|$)", text, re.IGNORECASE)
        if m:
            items = re.split(r",\s*(?:and\s+)?|;\s*", m.group(1))
            for item in items:
                cleaned = _clean_entity(item)
                if cleaned:
                    entities.append(cleaned)

    kv_matches = re.findall(r"(?:^|\n)\s*([A-Z][^:\n]{2,40}):\s*(\S.+?)(?:\n|$)", text)
    for key, _val in kv_matches:
        cleaned = _clean_entity(key.strip())
        if cleaned and not any(skip in cleaned.lower() for skip in
                               ["question", "answer", "note", "summary", "instruction",
                                "source", "reference", "finding", "result", "date"]):
            entities.append(cleaned)

    seen: set[str] = set()
    unique: list[str] = []
    for e in entities:
        ne = _normalize(e)
        if ne not in seen and len(ne) > 2:
            seen.add(ne)
            unique.append(e)
    return unique


# ---------------------------------------------------------------------------
# Entity-to-chunk matching
# ---------------------------------------------------------------------------

def _entity_in_text(entity: str, text: str) -> bool:
    """Check if an entity is mentioned in a chunk of text.

    Uses multiple matching strategies: exact substring, word overlap,
    abbreviation expansion.
    """
    ne = _normalize_entity(entity)
    nt = _normalize(text)

    # 1. Exact substring
    if ne in nt:
        return True

    # 2. Word overlap (>=50% of entity words found in chunk)
    words_e = set(ne.split())
    words_t = set(nt.split())
    if words_e:
        overlap = len(words_e & words_t)
        if len(words_e) <= 2:
            if overlap >= len(words_e):
                return True
        elif overlap / len(words_e) >= 0.5:
            return True

    # 3. Abbreviation expansion
    ee = _expand_abbreviations(entity)
    if ee != ne and ee in nt:
        return True

    return False


# ---------------------------------------------------------------------------
# Main verification function
# ---------------------------------------------------------------------------

def _is_numeric_or_lab_value(entity: str) -> bool:
    """Check if an entity is primarily a numeric/lab value rather than a clinical concept.

    We skip verification for these because lab values are often reformatted
    by the model (e.g., "Hematocrit: 38 %" vs "HCT 38.0") making string
    matching unreliable. The numeric validator handles these separately.
    """
    ne = _normalize(entity)
    # Contains significant numeric content
    digits = sum(1 for c in ne if c.isdigit())
    if len(ne) > 0 and digits / len(ne) > 0.3:
        return True
    # Looks like a key:value lab result
    if re.match(r"^[a-z\s]+\s*\d", ne):
        return True
    # Contains units
    if re.search(r"\b(?:mg|ml|dl|kg|mmol|meq|iu|bpm|mmhg|g|mcg|units?|%)\b", ne, re.IGNORECASE):
        return True
    return False


def _is_descriptive_sentence(entity: str) -> bool:
    """Check if the extracted 'entity' is actually a descriptive sentence, not a clinical entity."""
    words = entity.split()
    if len(words) > 12:
        return True
    lower = entity.lower()
    if any(w in lower for w in ["during", "throughout", "initially", "indicating",
                                 "consistent with", "improving", "decreased",
                                 "increased", "ranging from", "normalizing"]):
        return True
    return False


def verify_entities(
    answer_text: str,
    chunk_texts: list[str],
) -> tuple[str, list[str], list[str]]:
    """Verify entities in the answer against evidence chunks.

    Parameters
    ----------
    answer_text:
        The final answer text (after critic).
    chunk_texts:
        List of evidence chunk text strings.

    Returns
    -------
    tuple of (cleaned_text, grounded_entities, removed_entities)
        - cleaned_text: answer with hallucinated entity lines removed
        - grounded_entities: entities that matched at least one chunk
        - removed_entities: entities that matched no chunks
    """
    entities = extract_entities(answer_text)
    if not entities:
        return answer_text, [], []

    # Combine all chunk text for matching
    combined_evidence = " ".join(chunk_texts)

    grounded: list[str] = []
    hallucinated: list[str] = []

    for entity in entities:
        # Skip numeric/lab values and long descriptive sentences -- these are
        # handled by the numeric validator and are too noisy for string matching.
        if _is_numeric_or_lab_value(entity) or _is_descriptive_sentence(entity):
            grounded.append(entity)  # give benefit of the doubt
            continue

        # Check against combined evidence first (fast path)
        if _entity_in_text(entity, combined_evidence):
            grounded.append(entity)
        else:
            # Also check individual chunks (for cases where normalization matters)
            found = False
            for ct in chunk_texts:
                if _entity_in_text(entity, ct):
                    found = True
                    break
            if found:
                grounded.append(entity)
            else:
                hallucinated.append(entity)

    if not hallucinated:
        return answer_text, grounded, []

    # Remove lines containing hallucinated entities
    cleaned_lines: list[str] = []
    lines = answer_text.split("\n")

    for line in lines:
        line_norm = _normalize(line)
        should_remove = False
        for h_entity in hallucinated:
            h_norm = _normalize_entity(h_entity)
            # Check if the hallucinated entity appears in this line
            if h_norm in line_norm:
                should_remove = True
                break
            # Also check word overlap for fuzzy line matching
            words_h = set(h_norm.split())
            words_l = set(line_norm.split())
            if words_h and len(words_h) >= 2:
                overlap = len(words_h & words_l)
                if overlap / len(words_h) >= 0.8:
                    should_remove = True
                    break
        if not should_remove:
            cleaned_lines.append(line)

    cleaned_text = "\n".join(cleaned_lines).strip()

    # Fix numbering if we removed list items
    cleaned_text = _renumber_list(cleaned_text)

    logger.info(
        "Entity verification: %d grounded, %d removed: %s",
        len(grounded),
        len(hallucinated),
        hallucinated,
    )

    return cleaned_text, grounded, hallucinated


def _renumber_list(text: str) -> str:
    """Re-number items in a numbered list after removals."""
    lines = text.split("\n")
    counter = 0
    result: list[str] = []
    for line in lines:
        m = re.match(r"^(\s*)\d+([.\)])\s*", line)
        if m:
            counter += 1
            rest = line[m.end():]
            result.append(f"{m.group(1)}{counter}{m.group(2)} {rest}")
        else:
            result.append(line)
    return "\n".join(result)
