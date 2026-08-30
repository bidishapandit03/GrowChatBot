"""Phase 5 steps 9-12: grounded generation, output validation, and safe fallback.

Evidence is generated only when retrieval is grounded and conflict-free. Prompt
construction, Mistral generation, and output validation each fail closed, so an
unvalidated Mistral answer never reaches the user.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from code.rag.generator import GenerationError, generate
from code.rag.prompt import build_messages, evidence_freshness
from code.rag.retriever import (
    DECISION_BLOCKED,
    DECISION_CLARIFICATION,
    DECISION_ERROR,
    DECISION_FOUND,
    DECISION_NOT_FOUND,
    MESSAGE_ERROR,
    MESSAGE_NOT_FOUND,
    Retriever,
    RetrievalResult,
)
from code.rag.validator import ValidationError, validate_answer

MESSAGE_UNABLE_TO_ANSWER = (
    "Sorry, I could not generate a verified answer from the approved pages. "
    "Please rephrase or try another question."
)


@dataclass(frozen=True)
class AnswerResult:
    question: str
    retrieval: RetrievalResult
    answer: str | None
    grounded: bool

    def to_dict(self) -> dict:
        payload = self.retrieval.to_dict()
        payload["grounded"] = self.grounded
        payload["answer"] = self.answer
        return payload


def _conflict_answer(result: RetrievalResult) -> str:
    details = "; ".join(result.conflicts)
    freshness = evidence_freshness(result.evidence)
    answer = (
        f"The approved pages contain conflicting values for: {details}. "
        "I cannot give a single figure without choosing between them."
    )
    if result.fund:
        answer = f"{answer}\nSource: {result.fund.canonical_url}"
    if freshness:
        answer = f"{answer}\nLast updated from sources: {freshness}"
    return answer


def answer(question: str, retriever: Retriever | None = None, generate_fn: Callable[[list[dict[str, str]]], dict] = generate) -> AnswerResult:
    """Run the full online path and return either a grounded answer or a safe fallback."""
    retriever = retriever or Retriever()
    result = retriever.retrieve(question)

    if result.decision == DECISION_FOUND and result.conflicts:
        return AnswerResult(question, result, _conflict_answer(result), grounded=False)

    if result.decision == DECISION_FOUND:
        try:
            messages = build_messages(result.question, result.evidence)
            raw = generate_fn(messages)
            validated = validate_answer(raw, result.evidence)
        except (GenerationError, ValidationError):
            return AnswerResult(question, result, MESSAGE_UNABLE_TO_ANSWER, grounded=False)
        answer_text = (
            f"{validated.answer}\nSource: {validated.source}\n"
            f"Last updated from sources: {validated.last_updated}"
        )
        return AnswerResult(question, result, answer_text, grounded=True)

    if result.decision == DECISION_NOT_FOUND:
        return AnswerResult(question, result, result.message, grounded=False)

    if result.decision == DECISION_ERROR:
        return AnswerResult(question, result, MESSAGE_ERROR, grounded=False)

    return AnswerResult(question, result, result.message, grounded=False)