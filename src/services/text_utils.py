"""Utilities for normalising plain text into Markdown-friendly strings."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

_MARKDOWN_SIGNATURES = (
    re.compile(r"^#{1,6}\s", re.MULTILINE),
    re.compile(r"^\s*[-*+]\s+\S", re.MULTILINE),
    re.compile(r"^\s*\d+\.\s+\S", re.MULTILINE),
    re.compile(r"`{3,}", re.MULTILINE),
    re.compile(r"\[.+?\]\(.+?\)"),
)


def _normalise_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _looks_like_markdown(text: str) -> bool:
    return any(pattern.search(text) for pattern in _MARKDOWN_SIGNATURES)


def _collapse_whitespace(text: str) -> str:
    paragraphs = re.split(r"\n\s*\n", text.strip())
    cleaned: list[str] = []

    for paragraph in paragraphs:
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        if not lines:
            continue

        # Preserve list formatting when the paragraph is already a bullet/numbered list.
        if all(re.match(r"^(\d+\.\s+|[-*+]\s+)", line) for line in lines):
            cleaned.append("\n".join(lines))
        else:
            cleaned.append(" ".join(lines))

    return "\n\n".join(cleaned)


def ensure_markdown(
    text: str,
    *,
    source_name: Optional[str] = None,
    prefer_title: bool = True,
) -> str:
    """
    Convert raw text to a Markdown-friendly representation.

    If the text already appears to contain Markdown syntax the original value is returned.
    Otherwise we normalise whitespace, optionally prepend a heading, and return Markdown.
    """
    if not text:
        return ""

    normalised = _normalise_newlines(text)
    stripped = normalised.strip()
    if not stripped:
        return ""

    if _looks_like_markdown(stripped):
        return stripped

    converted = _collapse_whitespace(stripped)

    if not converted:
        return ""

    if prefer_title and source_name:
        stem = Path(source_name).stem
        if stem and not converted.startswith("# "):
            return f"# {stem}\n\n{converted}"

    return converted
