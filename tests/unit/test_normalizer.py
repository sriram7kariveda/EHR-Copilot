"""Unit tests for the text normalizer (ingestion/normalizer.py)."""

from __future__ import annotations

from datetime import datetime

import pytest

from ehr_copilot.ingestion.normalizer import (
    clean_clinical_text,
    normalize_date,
    normalize_units,
)


# ---------------------------------------------------------------------------
# clean_clinical_text
# ---------------------------------------------------------------------------


class TestCleanClinicalText:
    def test_collapses_whitespace(self):
        text = "Blood   pressure    is    stable."
        result = clean_clinical_text(text)
        assert result == "Blood pressure is stable."

    def test_collapses_excessive_newlines(self):
        text = "Section A\n\n\n\n\nSection B"
        result = clean_clinical_text(text)
        assert result == "Section A\n\nSection B"

    def test_normalizes_unicode_smart_quotes(self):
        text = "\u201cDiagnosis\u201d: patient\u2019s condition is \u201cstable\u201d"
        result = clean_clinical_text(text)
        assert "\u201c" not in result
        assert "\u201d" not in result
        assert "\u2019" not in result
        assert '"Diagnosis"' in result

    def test_normalizes_nonbreaking_space(self):
        text = "Value:\u00a0120\u00a0mg/dL"
        result = clean_clinical_text(text)
        assert "\u00a0" not in result
        assert "Value: 120 mg/dL" in result

    def test_strips_leading_trailing_line_whitespace(self):
        text = "  Line one  \n  Line two  "
        result = clean_clinical_text(text)
        assert result == "Line one\nLine two"

    def test_empty_string(self):
        assert clean_clinical_text("") == ""
        assert clean_clinical_text("   ") == ""


# ---------------------------------------------------------------------------
# normalize_date
# ---------------------------------------------------------------------------


class TestNormalizeDate:
    def test_iso_date(self):
        result = normalize_date("2024-01-15")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_us_slash_date(self):
        result = normalize_date("01/15/2024")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_named_month_date(self):
        result = normalize_date("January 15, 2024")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1

    def test_abbreviated_month(self):
        result = normalize_date("15-Jan-2024")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_iso_datetime_with_tz(self):
        result = normalize_date("2024-01-15T10:30:00+00:00")
        assert result is not None
        assert result.year == 2024

    def test_empty_string_returns_none(self):
        assert normalize_date("") is None
        assert normalize_date("   ") is None

    def test_garbage_returns_none(self):
        assert normalize_date("not-a-date") is None


# ---------------------------------------------------------------------------
# normalize_units
# ---------------------------------------------------------------------------


class TestNormalizeUnits:
    def test_case_insensitive_mg_dl(self):
        result = normalize_units("Value: 120 MG/DL")
        assert "mg/dL" in result

    def test_celsius_normalization(self):
        result = normalize_units("Temperature: 37.5 celsius")
        assert "Cel" in result

    def test_u_per_l_lowercase(self):
        result = normalize_units("ALT: 45 u/l")
        assert "U/L" in result

    def test_preserves_surrounding_text(self):
        result = normalize_units("Heart rate: 80 bpm at rest")
        assert result.startswith("Heart rate: 80")
        assert result.endswith("at rest")
        assert "bpm" in result
