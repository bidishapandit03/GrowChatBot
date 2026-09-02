"""Enforce <=3 sentences, exactly one allowlisted citation, source freshness, and
that every substantive claim in the answer actually appears in the cited evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass

from code.config import APPROVED_SOURCES, MAX_ANSWER_SENTENCES
from code.rag.prompt import evidence_freshness

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'({])")
# Multi-word capitalized phrases and standalone numeric values are the two ways a
# model can smuggle in a fabricated value or entity. Sentence-initial fillers count
# as part of a phrase (e.g. "Its AUM") so they are stripped before checking.
_CAPITALIZED_SEQUENCE = re.compile(r"[A-Z][\w&.\-'’]*(?:\s+[A-Z][\w&.\-'’]+)+")
_NUMBER_TOKEN = re.compile(r"\d+(?:[.,]\d+)*")
_FIRST_TOKEN_STOPWORDS = {
    "a", "an", "and", "as", "at", "for", "in", "its", "it", "of", "on", "or",
    "the", "this", "that", "these", "those", "to", "is", "are", "was", "were",
    "yes", "no",
}


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


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _significant_claims(answer: str) -> set[str]:
    """Capitalized phrases and numeric values in the answer, normalized.

    Leading fillers on a capitalized phrase (e.g. "Its AUM") are dropped and a
    phrase reduced to a single word is not treated as a claim, so that a
    sentence-initial capitalized pronoun or article cannot make an otherwise
    supported sentence look fabricated.
    """
    terms: set[str] = set()
    for sentence in _SENTENCE_SPLIT.split(answer):
        for phrase in _CAPITALIZED_SEQUENCE.findall(sentence):
            words = phrase.split()
            if _normalize(words[0]) in _FIRST_TOKEN_STOPWORDS:
                words = words[1:]
            if len(words) >= 2:
                terms.add(_normalize(" ".join(words)))
        terms.update(_normalize(number) for number in _NUMBER_TOKEN.findall(sentence))
    return terms


def _evidence_corpus(evidence) -> str:
    parts: list[str] = []
    for row in evidence:
        parts.append(getattr(row, "chunk_text", "") or "")
        parts.append(getattr(row, "fund_name", "") or "")
    return _normalize(" ".join(parts))


def unsupported_claims(answer: str, evidence) -> list[str]:
    """Return normalized claim terms from ``answer`` missing from the evidence.

    Only structured section rows count as claim support. The free-form "Visible
    page text" chunks contain disconnected crawler noise (related-fund carousels,
    comparison tables) whose tokens must not be used to justify an assertion -
    this is what lets a model smuggle in a plausible-looking but fabricated value
    like "HDFC Bank" that happens to appear somewhere on the page.
    """
    structured = [row for row in evidence if getattr(row, "section_heading", None) != "Visible page text"]
    corpus = _evidence_corpus(structured)
    if not corpus:
        return sorted(_significant_claims(answer))
    return sorted(term for term in _significant_claims(answer) if term not in corpus)


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValidationError(reason)


def validate_answer(output: dict, evidence) -> ValidatedAnswer:
    """Validate Mistral's structured output against structure, citation, and claim-support rules."""
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
        missing = unsupported_claims(answer, evidence)
        _require(not missing, f"answer claims not supported by the evidence: {missing}")

    return ValidatedAnswer(
        answer=answer.strip(),
        source=source.strip(),
        last_updated=last_updated.strip(),
    )