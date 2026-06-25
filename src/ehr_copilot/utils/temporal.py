"""Clinical date parsing and formatting utilities.

Handles the wide variety of date representations commonly encountered in
clinical notes and EHR free-text fields.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# Absolute date parsing
# ---------------------------------------------------------------------------

# Ordered list of strptime patterns to try when parsing an absolute date.
_DATE_FORMATS: list[str] = [
    # ISO-style
    "%Y-%m-%d",            # 2024-01-15
    "%Y/%m/%d",            # 2024/01/15
    # US-style with slashes
    "%m/%d/%Y",            # 01/15/2024
    "%m/%d/%y",            # 01/15/24
    # US-style with dashes
    "%m-%d-%Y",            # 01-15-2024
    "%m-%d-%y",            # 01-15-24
    # Long month name
    "%B %d, %Y",           # January 15, 2024
    "%B %d %Y",            # January 15 2024
    # Abbreviated month name
    "%b %d, %Y",           # Jan 15, 2024
    "%b %d %Y",            # Jan 15 2024
    # Day-first variants (common in some clinical systems)
    "%d %B %Y",            # 15 January 2024
    "%d %b %Y",            # 15 Jan 2024
    "%d-%b-%Y",            # 15-Jan-2024
    # Month-Year (day defaults to 1)
    "%B %Y",               # January 2024
    "%b %Y",               # Jan 2024
]


def parse_clinical_date(date_str: str) -> datetime | None:
    """Parse a date string using formats common in clinical notes.

    The function tries each candidate format in order and returns the
    first successful parse.  Leading/trailing whitespace is stripped
    automatically.

    Parameters
    ----------
    date_str:
        The raw date string from a clinical note.

    Returns
    -------
    datetime | None
        A :class:`~datetime.datetime` on success, or ``None`` if no
        format matched.
    """
    cleaned = date_str.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Relative time parsing
# ---------------------------------------------------------------------------

_RELATIVE_PATTERN = re.compile(
    r"(?:(?:last|past|previous|within(?: the)?)\s+)"
    r"(\d+)\s+"
    r"(day|days|week|weeks|month|months|year|years)",
    re.IGNORECASE,
)

# Map the plural/singular unit words to a timedelta-friendly keyword.
_UNIT_MAP: dict[str, str] = {
    "day": "days",
    "days": "days",
    "week": "weeks",
    "weeks": "weeks",
    "month": "months",
    "months": "months",
    "year": "years",
    "years": "years",
}


def _make_delta(amount: int, unit: str) -> timedelta:
    """Build a timedelta from a numeric amount and a clinical time unit."""
    key = _UNIT_MAP[unit.lower()]
    if key == "days":
        return timedelta(days=amount)
    if key == "weeks":
        return timedelta(weeks=amount)
    if key == "months":
        return timedelta(days=amount * 30)
    if key == "years":
        return timedelta(days=amount * 365)
    # Fallback (should not be reached).
    return timedelta(days=amount)  # pragma: no cover


def parse_relative_time(
    text: str,
    reference: datetime | None = None,
) -> tuple[datetime, datetime] | None:
    """Parse a relative time expression into an absolute date range.

    Supported phrases include ``"last 6 months"``, ``"past year"``,
    ``"last 3 days"``, ``"previous 2 weeks"``, etc.

    Parameters
    ----------
    text:
        Free-text that may contain a relative time expression.
    reference:
        The reference point for *now*.  Defaults to
        :func:`datetime.now()` when ``None``.

    Returns
    -------
    tuple[datetime, datetime] | None
        ``(start, end)`` datetime pair where *end* is the reference date
        and *start* is the computed beginning of the window, or ``None``
        if the text could not be parsed.
    """
    match = _RELATIVE_PATTERN.search(text)
    if match is None:
        # Handle the shorthand "past year" / "last year" (no digit).
        shorthand = re.search(
            r"(?:last|past|previous)\s+(day|week|month|year)",
            text,
            re.IGNORECASE,
        )
        if shorthand is None:
            return None
        amount = 1
        unit = shorthand.group(1)
    else:
        amount = int(match.group(1))
        unit = match.group(2)

    end = reference if reference is not None else datetime.now()
    delta = _make_delta(amount, unit)
    start = end - delta
    return (start, end)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_clinical_date(dt: datetime) -> str:
    """Format a datetime as ``YYYY-MM-DD``.

    Parameters
    ----------
    dt:
        The datetime to format.

    Returns
    -------
    str
        ISO-style date string.
    """
    return dt.strftime("%Y-%m-%d")


def days_between(d1: datetime, d2: datetime) -> int:
    """Return the absolute number of whole days between two datetimes.

    Parameters
    ----------
    d1:
        First datetime.
    d2:
        Second datetime.

    Returns
    -------
    int
        Non-negative integer representing the number of days separating
        *d1* and *d2*.
    """
    return abs((d1 - d2).days)
