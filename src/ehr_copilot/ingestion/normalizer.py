"""Text cleaning and date/unit normalization utilities for clinical text."""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from typing import Optional


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_LEADING_TRAILING_WS = re.compile(r"^[ \t]+|[ \t]+$", re.MULTILINE)


def clean_clinical_text(text: str) -> str:
    """Clean clinical text by normalizing unicode and collapsing whitespace.

    * NFKC-normalise unicode (smart quotes, ligatures, etc.)
    * Replace runs of horizontal whitespace with a single space
    * Collapse 3+ consecutive newlines into 2
    * Strip leading/trailing whitespace on each line
    * Strip leading/trailing whitespace from the entire string
    """
    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)

    # Replace common unicode characters that slip through
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u00a0", " ")  # non-breaking space

    # Collapse horizontal whitespace within lines
    text = _MULTI_SPACE.sub(" ", text)

    # Strip leading/trailing whitespace on each line
    text = _LEADING_TRAILING_WS.sub("", text)

    # Collapse excessive blank lines
    text = _MULTI_NEWLINE.sub("\n\n", text)

    return text.strip()


# ---------------------------------------------------------------------------
# Date normalization
# ---------------------------------------------------------------------------

_DATE_FORMATS = [
    # ISO-like formats
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
    # US-style
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%m/%d/%Y",
    "%m-%d-%Y",
    # Common clinical formats
    "%d-%b-%Y",       # 12-Jan-2024
    "%d %b %Y",       # 12 Jan 2024
    "%b %d, %Y",      # Jan 12, 2024
    "%B %d, %Y",      # January 12, 2024
    "%d/%m/%Y",       # 12/01/2024 (day-first, tried after US)
    "%Y%m%d",         # 20240112
    "%Y%m%d%H%M%S",   # 20240112143000
]


def normalize_date(date_str: str) -> Optional[datetime]:
    """Parse a variety of clinical date formats into a datetime.

    Returns ``None`` if the string cannot be parsed.
    """
    if not date_str or not date_str.strip():
        return None

    cleaned = date_str.strip()

    # Handle timezone offset with colon (e.g., +00:00 -> +0000)
    if re.search(r"[+-]\d{2}:\d{2}$", cleaned):
        cleaned = cleaned[:-3] + cleaned[-2:]

    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue

    return None


# ---------------------------------------------------------------------------
# Unit normalization
# ---------------------------------------------------------------------------

_UNIT_MAP: dict[re.Pattern, str] = {
    re.compile(r"\bmg/dL\b", re.IGNORECASE): "mg/dL",
    re.compile(r"\bmmol/L\b", re.IGNORECASE): "mmol/L",
    re.compile(r"\bm[Ee]q/L\b"): "mEq/L",
    re.compile(r"\bng/mL\b", re.IGNORECASE): "ng/mL",
    re.compile(r"\bpg/mL\b", re.IGNORECASE): "pg/mL",
    re.compile(r"\bug/dL\b", re.IGNORECASE): "ug/dL",
    re.compile(r"\bmcg\b", re.IGNORECASE): "mcg",
    re.compile(r"\bIU/L\b", re.IGNORECASE): "IU/L",
    re.compile(r"\bU/L\b"): "U/L",
    re.compile(r"\bu/l\b"): "U/L",
    re.compile(r"\b10\*3/uL\b"): "10^3/uL",
    re.compile(r"\b10\*6/uL\b"): "10^6/uL",
    re.compile(r"\bx10E3/uL\b", re.IGNORECASE): "10^3/uL",
    re.compile(r"\bx10E6/uL\b", re.IGNORECASE): "10^6/uL",
    re.compile(r"\b%\b"): "%",
    re.compile(r"\bmmHg\b", re.IGNORECASE): "mmHg",
    re.compile(r"\bbpm\b", re.IGNORECASE): "bpm",
    re.compile(r"\bkg\b"): "kg",
    re.compile(r"\bcm\b"): "cm",
    re.compile(r"\bcelsius\b", re.IGNORECASE): "Cel",
    re.compile(r"\bdeg\s*[Cc]\b"): "Cel",
    re.compile(r"\b[Dd]egrees?\s+[Ff]ahrenheit\b"): "degF",
    re.compile(r"\bdeg\s*[Ff]\b"): "degF",
    re.compile(r"\bbeats?\s*/\s*min(?:ute)?\b", re.IGNORECASE): "bpm",
    re.compile(r"\bbreaths?\s*/\s*min(?:ute)?\b", re.IGNORECASE): "breaths/min",
}


def normalize_units(text: str) -> str:
    """Standardize common clinical unit abbreviations in *text*.

    Replaces recognized unit patterns with their canonical form while
    preserving surrounding text.
    """
    for pattern, canonical in _UNIT_MAP.items():
        text = pattern.sub(canonical, text)
    return text
