"""Unit tests for temporal utilities (utils/temporal.py)."""

from __future__ import annotations

from datetime import datetime

import pytest

from ehr_copilot.utils.temporal import (
    days_between,
    format_clinical_date,
    parse_clinical_date,
    parse_relative_time,
)


# ---------------------------------------------------------------------------
# parse_clinical_date
# ---------------------------------------------------------------------------


class TestParseClinicalDate:
    def test_iso_format(self):
        result = parse_clinical_date("2024-01-15")
        assert result is not None
        assert result == datetime(2024, 1, 15)

    def test_us_slash_format(self):
        result = parse_clinical_date("01/15/2024")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_long_month_name(self):
        result = parse_clinical_date("January 15, 2024")
        assert result is not None
        assert result.month == 1

    def test_abbreviated_month_name(self):
        result = parse_clinical_date("Jan 15, 2024")
        assert result is not None
        assert result.month == 1
        assert result.day == 15

    def test_day_first_abbreviated(self):
        result = parse_clinical_date("15-Jan-2024")
        assert result is not None
        assert result.day == 15
        assert result.month == 1

    def test_unknown_format_returns_none(self):
        result = parse_clinical_date("not a date at all")
        assert result is None

    def test_strips_whitespace(self):
        result = parse_clinical_date("  2024-01-15  ")
        assert result is not None
        assert result.year == 2024


# ---------------------------------------------------------------------------
# parse_relative_time
# ---------------------------------------------------------------------------


class TestParseRelativeTime:
    def test_last_6_months(self):
        ref = datetime(2024, 7, 1)
        result = parse_relative_time("labs from the last 6 months", reference=ref)
        assert result is not None
        start, end = result
        assert end == ref
        # ~180 days back
        delta_days = (end - start).days
        assert 179 <= delta_days <= 181

    def test_past_year_shorthand(self):
        ref = datetime(2024, 7, 1)
        result = parse_relative_time("past year", reference=ref)
        assert result is not None
        start, end = result
        delta_days = (end - start).days
        assert delta_days == 365

    def test_last_3_days(self):
        ref = datetime(2024, 7, 10)
        result = parse_relative_time("within the last 3 days", reference=ref)
        assert result is not None
        start, end = result
        assert (end - start).days == 3

    def test_previous_2_weeks(self):
        ref = datetime(2024, 7, 15)
        result = parse_relative_time("previous 2 weeks", reference=ref)
        assert result is not None
        start, end = result
        assert (end - start).days == 14

    def test_no_match_returns_none(self):
        result = parse_relative_time("what medications is the patient on?")
        assert result is None


# ---------------------------------------------------------------------------
# format_clinical_date
# ---------------------------------------------------------------------------


class TestFormatClinicalDate:
    def test_formats_as_iso_date(self):
        dt = datetime(2024, 1, 15, 10, 30, 0)
        assert format_clinical_date(dt) == "2024-01-15"

    def test_zero_padded(self):
        dt = datetime(2024, 3, 5)
        assert format_clinical_date(dt) == "2024-03-05"


# ---------------------------------------------------------------------------
# days_between
# ---------------------------------------------------------------------------


class TestDaysBetween:
    def test_same_date(self):
        dt = datetime(2024, 1, 15)
        assert days_between(dt, dt) == 0

    def test_positive_difference(self):
        d1 = datetime(2024, 1, 1)
        d2 = datetime(2024, 1, 31)
        assert days_between(d1, d2) == 30

    def test_order_independent(self):
        d1 = datetime(2024, 1, 1)
        d2 = datetime(2024, 6, 20)
        assert days_between(d1, d2) == days_between(d2, d1)
        assert days_between(d1, d2) > 0
