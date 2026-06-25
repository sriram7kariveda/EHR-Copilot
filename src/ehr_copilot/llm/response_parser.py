"""Utilities for parsing structured data out of free-form LLM responses."""

from __future__ import annotations

import json
import logging
import re
from enum import Enum
from typing import Type

logger = logging.getLogger(__name__)


class ResponseParser:
    """Collection of static helpers for extracting structured values from LLM text."""

    # ------------------------------------------------------------------
    # JSON extraction
    # ------------------------------------------------------------------

    @staticmethod
    def parse_json_block(text: str) -> dict:
        """Extract a JSON object from *text*.

        The method tries the following strategies in order:

        1. Look for a fenced code-block (```json ... ``` or ``` ... ```)
           and parse its content.
        2. Look for the first ``{`` ... ``}`` pair and parse that.
        3. Attempt to parse the entire *text* as JSON directly.

        Returns
        -------
        dict
            The parsed JSON object.

        Raises
        ------
        ValueError
            If no valid JSON object can be extracted.
        """
        # Strategy 1: fenced code-block
        fence_pattern = re.compile(
            r"```(?:json)?\s*\n?(.*?)```", re.DOTALL
        )
        match = fence_pattern.search(text)
        if match:
            candidate = match.group(1).strip()
            try:
                result = json.loads(candidate)
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

        # Strategy 2: first { ... } pair (outermost braces)
        brace_start = text.find("{")
        if brace_start != -1:
            depth = 0
            for idx in range(brace_start, len(text)):
                if text[idx] == "{":
                    depth += 1
                elif text[idx] == "}":
                    depth -= 1
                if depth == 0:
                    candidate = text[brace_start : idx + 1]
                    try:
                        result = json.loads(candidate)
                        if isinstance(result, dict):
                            return result
                    except json.JSONDecodeError:
                        break

        # Strategy 3: parse the full text
        try:
            result = json.loads(text.strip())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

        raise ValueError(f"Could not extract a JSON object from the LLM response: {text[:200]}")

    # ------------------------------------------------------------------
    # Enum matching
    # ------------------------------------------------------------------

    @staticmethod
    def parse_enum_value(text: str, enum_class: Type[Enum]) -> Enum:
        """Fuzzy-match *text* to a member of *enum_class*.

        Matching is performed in the following order:

        1. Exact match on member *value* (case-insensitive).
        2. Exact match on member *name* (case-insensitive).
        3. Substring / containment check -- the first member whose
           lowercased name or value appears in the lowercased text wins.

        Raises
        ------
        ValueError
            If no member can be matched.
        """
        cleaned = text.strip()
        lower = cleaned.lower()

        # Pass 1: exact value match (case-insensitive)
        for member in enum_class:
            if str(member.value).lower() == lower:
                return member

        # Pass 2: exact name match (case-insensitive)
        for member in enum_class:
            if member.name.lower() == lower:
                return member

        # Pass 3: containment (value first, then name)
        for member in enum_class:
            if str(member.value).lower() in lower:
                return member
        for member in enum_class:
            if member.name.lower() in lower:
                return member

        valid = [f"{m.name}={m.value}" for m in enum_class]
        raise ValueError(
            f"Cannot match '{cleaned}' to {enum_class.__name__}. "
            f"Valid members: {valid}"
        )

    # ------------------------------------------------------------------
    # Numeric extraction
    # ------------------------------------------------------------------

    @staticmethod
    def parse_float(text: str) -> float | None:
        """Return the first floating-point number found in *text*, or ``None``.

        Handles integers, decimals, and negative numbers.
        """
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if match:
            return float(match.group())
        return None

    # ------------------------------------------------------------------
    # List extraction
    # ------------------------------------------------------------------

    @staticmethod
    def parse_list(text: str) -> list[str]:
        """Parse a bulleted or numbered list from *text*.

        Recognised formats::

            - item          * item          1. item
            - item          * item          2) item

        Lines that do not match a list marker are ignored.  Leading /
        trailing whitespace on each item is stripped.  Empty items are
        discarded.
        """
        pattern = re.compile(
            r"^\s*(?:[-*+]|\d+[.)]) \s*(.*)", re.MULTILINE
        )
        items: list[str] = []
        for match in pattern.finditer(text):
            item = match.group(1).strip()
            if item:
                items.append(item)
        return items

    # ------------------------------------------------------------------
    # XML-style tag extraction
    # ------------------------------------------------------------------

    @staticmethod
    def extract_between_tags(text: str, tag: str) -> str:
        """Return the content between ``<tag>`` and ``</tag>``.

        Parameters
        ----------
        text:
            The full text that may contain the XML-like tags.
        tag:
            The tag name **without** angle brackets (e.g. ``"answer"``).

        Returns
        -------
        str
            The trimmed content between the opening and closing tags.

        Raises
        ------
        ValueError
            If the tags are not found in *text*.
        """
        pattern = re.compile(
            rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>",
            re.DOTALL,
        )
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
        raise ValueError(f"Tags <{tag}>...</{tag}> not found in text")
