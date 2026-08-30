"""Enforce <=3 sentences, exactly one allowlisted citation, and source freshness."""

from __future__ import annotations

import re
from dataclasses import dataclass

from code.config import APPROVED_SOURCES, MAX_ANSWER_SENTENCES
from code.rag.prompt import evidence_freshness

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'({])")


class ValidationError(RuntimeError):
    """Raised when a generated answer violates the output contract."""


@dataclass(frozen=True)
class ValidatedAnswer:
    answer: str
    source: str
    last_updated: str


def allowed_urls() -> set[str]:
    return {source["canonical_url"] for source in APPROVED_SOURCES}


def count_sentences(text: str) -> int:
    return len([part for part in _SENTENCE_SPLIT.split(text.strip()) if part.strip()])


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValidationError(reason)


def validate_answer(output: dict, evidence) -> ValidatedAnswer:
    """Validate Mistral's structured output against the allowlist, length, and freshness rules."""
    _require(isinstance(output, dict), "answer is not an object")

    answer = output.get("answer")
    source = output.get("source")
    last_updated = output.get("last_updated")

    _require(isinstance(answer, str) and answer.strip(), "answer is empty")
    _require(count_sentences(answer) <= MAX_ANSWER_SENTENCES, "answer exceeds 3 sentences")

    _require(isinstance(source, str), "source is missing")
    _require(source in allowed_urls(), "source is not an approved URL")

    _require(isinstance(last_updated, str), "last_updated is missing")
    _require(_ISO_DATE.match(last_updated.strip()) is not None, "last_updated is not YYYY-MM-DD")

    if evidence:
        _require(
            last_updated.strip() == evidence_freshness(evidence),
            "last_updated does not match the cited evidence freshness",
        )

    return ValidatedAnswer(
        answer=answer.strip(),
        source=source.strip(),
        last_updated=last_updated.strip(),
    )