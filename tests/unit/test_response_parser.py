"""Unit tests for the LLM response parser (llm/response_parser.py)."""

from __future__ import annotations

import pytest

from ehr_copilot.domain.query import QueryType
from ehr_copilot.llm.response_parser import ResponseParser


# ---------------------------------------------------------------------------
# parse_json_block
# ---------------------------------------------------------------------------


class TestParseJsonBlock:
    def test_with_markdown_code_block(self):
        text = (
            "Here is the classification:\n\n"
            "```json\n"
            '{"query_type": "FACTUAL", "confidence": 0.95}\n'
            "```\n\n"
            "Let me know if you need more details."
        )
        result = ResponseParser.parse_json_block(text)
        assert result["query_type"] == "FACTUAL"
        assert result["confidence"] == 0.95

    def test_with_raw_json(self):
        text = '{"verdict": "APPROVED", "issues": []}'
        result = ResponseParser.parse_json_block(text)
        assert result["verdict"] == "APPROVED"
        assert result["issues"] == []

    def test_json_embedded_in_text(self):
        text = (
            'The query is factual in nature. My analysis:\n'
            '{"query_type": "NUMERIC", "requires_numeric": true}\n'
            "That's my classification."
        )
        result = ResponseParser.parse_json_block(text)
        assert result["query_type"] == "NUMERIC"
        assert result["requires_numeric"] is True

    def test_invalid_json_raises_value_error(self):
        text = "This is not JSON at all."
        with pytest.raises(ValueError, match="Could not extract"):
            ResponseParser.parse_json_block(text)

    def test_code_block_without_json_label(self):
        text = (
            "```\n"
            '{"key": "value"}\n'
            "```"
        )
        result = ResponseParser.parse_json_block(text)
        assert result["key"] == "value"


# ---------------------------------------------------------------------------
# parse_enum_value
# ---------------------------------------------------------------------------


class TestParseEnumValue:
    def test_exact_value_match(self):
        result = ResponseParser.parse_enum_value("factual", QueryType)
        assert result == QueryType.FACTUAL

    def test_exact_name_match(self):
        result = ResponseParser.parse_enum_value("FACTUAL", QueryType)
        assert result == QueryType.FACTUAL

    def test_case_insensitive(self):
        result = ResponseParser.parse_enum_value("Medication", QueryType)
        assert result == QueryType.MEDICATION

    def test_containment_match(self):
        result = ResponseParser.parse_enum_value(
            "I think this is a temporal query", QueryType
        )
        assert result == QueryType.TEMPORAL

    def test_no_match_raises(self):
        with pytest.raises(ValueError, match="Cannot match"):
            ResponseParser.parse_enum_value("xyzzy_not_a_type", QueryType)


# ---------------------------------------------------------------------------
# parse_list
# ---------------------------------------------------------------------------


class TestParseList:
    def test_bullet_list(self):
        text = "Key findings:\n- Diabetes\n- Hypertension\n- Obesity"
        result = ResponseParser.parse_list(text)
        assert result == ["Diabetes", "Hypertension", "Obesity"]

    def test_numbered_list(self):
        text = "1. Metformin\n2. Lisinopril\n3. Aspirin"
        result = ResponseParser.parse_list(text)
        assert result == ["Metformin", "Lisinopril", "Aspirin"]

    def test_asterisk_list(self):
        text = "* Item A\n* Item B"
        result = ResponseParser.parse_list(text)
        assert result == ["Item A", "Item B"]

    def test_mixed_non_list_lines_ignored(self):
        text = "Introduction text\n- Real item\nMore text\n- Another item"
        result = ResponseParser.parse_list(text)
        assert result == ["Real item", "Another item"]

    def test_empty_items_discarded(self):
        text = "Items:\n- \n- Actual item\n-  "
        result = ResponseParser.parse_list(text)
        # Only the non-empty "Actual item" should be returned
        assert len(result) == 1
        assert "Actual item" in result[0]


# ---------------------------------------------------------------------------
# extract_between_tags
# ---------------------------------------------------------------------------


class TestExtractBetweenTags:
    def test_extracts_answer(self):
        text = (
            "<reasoning>Step 1: Check labs.</reasoning>\n"
            "<answer>The A1c is 7.2%.</answer>"
        )
        result = ResponseParser.extract_between_tags(text, "answer")
        assert result == "The A1c is 7.2%."

    def test_extracts_reasoning(self):
        text = (
            "<reasoning>\n"
            "  I looked at chunk [1] and found the A1c value.\n"
            "</reasoning>\n"
            "<answer>The answer is 7.2%.</answer>"
        )
        result = ResponseParser.extract_between_tags(text, "reasoning")
        assert "chunk [1]" in result

    def test_missing_tag_raises(self):
        text = "No tags here."
        with pytest.raises(ValueError, match="not found"):
            ResponseParser.extract_between_tags(text, "answer")

    def test_multiline_content(self):
        text = (
            "<source_chunks>\n"
            "1, 2, 3\n"
            "</source_chunks>"
        )
        result = ResponseParser.extract_between_tags(text, "source_chunks")
        assert "1, 2, 3" in result
