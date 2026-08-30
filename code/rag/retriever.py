"""Phase 5 retrieval logic: PII/scope gates, fund resolution, embedding, top-k, threshold, conflicts."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from code.config import RELEVANCE_THRESHOLD, TOP_K
from code.ingestion.embedder import EmbeddingError, embed_query
from code.ingestion.indexer import ChromaVectorStore
from code.rag.fund_resolver import ResolvedFund, matched_categories, resolve_fund
from code.rag.policy import PIICheck, PolicyDecision, QueryClass, REFUSAL_MESSAGES, classify_intent

DECISION_BLOCKED = "blocked"
DECISION_FOUND = "found"
DECISION_NOT_FOUND = "not_found"
DECISION_CLARIFICATION = "clarification"
DECISION_ERROR = "error"

MESSAGE_NOT_FOUND = "I couldn't find that fact in the five approved Groww pages."
MESSAGE_CLARIFICATION = (
    "Which of the five HDFC mutual funds do you mean? Large Cap, Flexi Cap, "
    "ELSS Tax Saver, Small Cap, or Balanced Advantage (all Direct Growth)?"
)
MESSAGE_MULTI_FUND = (
    "You mentioned more than one fund. Please ask about a single fund, for example the "
    "expense ratio or minimum SIP of one of them."
)
MESSAGE_ERROR = "Sorry, something went wrong while searching the approved pages. Please try again."

# Single-value fields: two different readings in the same fund's evidence means the source is inconsistent.
CONFLICT_SENSITIVE_LABELS = (
    "expense ratio",
    "minimum sip",
    "minimum lumpsum / 1st investment",
    "minimum additional investment",
    "exit load",
    "lock-in",
    "benchmark",
    "aum",
    "nav",
    "nav date",
    "riskometer",
    "stamp duty",
    "launch date",
)


@dataclass(frozen=True)
class EvidenceRow:
    chunk_id: str
    chunk_text: str
    canonical_url: str
    fund_name: str
    fund_category: str
    section_heading: str
    ingested_at: str
    distance: float

    @classmethod
    def from_query_row(cls, row: dict) -> "EvidenceRow":
        metadata = row["metadata"]
        return cls(
            chunk_id=row["chunk_id"],
            chunk_text=row["chunk_text"],
            canonical_url=metadata["canonical_url"],
            fund_name=metadata["fund_name"],
            fund_category=metadata["fund_category"],
            section_heading=metadata["section_heading"],
            ingested_at=metadata["ingested_at"],
            distance=row["distance"],
        )


@dataclass
class RetrievalResult:
    question: str
    query_class: QueryClass
    decision: str
    blocked: bool
    embedded: bool
    fund: ResolvedFund | None
    evidence: list[EvidenceRow] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    message: str | None = None
    pii_categories: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "query_class": self.query_class.value,
            "decision": self.decision,
            "blocked": self.blocked,
            "embedded": self.embedded,
            "fund": {
                "canonical_url": self.fund.canonical_url,
                "fund_name": self.fund.fund_name,
                "fund_category": self.fund.fund_category,
            }
            if self.fund
            else None,
            "evidence": [
                {
                    "chunk_id": row.chunk_id,
                    "section_heading": row.section_heading,
                    "fund_category": row.fund_category,
                    "canonical_url": row.canonical_url,
                    "distance": round(row.distance, 4),
                }
                for row in self.evidence
            ],
            "conflicts": self.conflicts,
            "message": self.message,
        }


def _detect_conflicts(evidence: list[EvidenceRow]) -> list[str]:
    """Flag scalar fields that appear with two or more distinct values among the evidence.

    Only structured sections use clean "label: value" lines; the free-form visible page
    text uses inconsistent prefixes (e.g. "NAV:" for a date), so it is excluded.
    """
    label_values: dict[str, set[str]] = {}
    for row in evidence:
        if row.section_heading == "Visible page text":
            continue
        for line in row.chunk_text.splitlines():
            match = re.match(r"^([^:]+):\s*(.+)$", line)
            if not match:
                continue
            label = " ".join(match.group(1).lower().split())
            if label not in CONFLICT_SENSITIVE_LABELS:
                continue
            value = match.group(2).strip()
            label_values.setdefault(label, set()).add(value)
    return [f"{label}: {', '.join(sorted(values))}" for label, values in label_values.items() if len(values) > 1]


class Retriever:
    def __init__(
        self,
        store: ChromaVectorStore | None = None,
        embed_fn: Callable[[str], list[float]] = embed_query,
        top_k: int = TOP_K,
        threshold: float = RELEVANCE_THRESHOLD,
    ):
        self._store = store
        self._embed_fn = embed_fn
        self._top_k = top_k
        self._threshold = threshold

    def _get_store(self) -> ChromaVectorStore:
        if self._store is None:
            self._store = ChromaVectorStore()
        return self._store

    def retrieve(self, question: str) -> RetrievalResult:
        question = question.strip()
        if not question:
            return RetrievalResult(question, QueryClass.FACTUAL, DECISION_CLARIFICATION, True, False, None)

        pii = PIICheck.clean(question)
        if pii.flagged:
            return RetrievalResult(
                question,
                QueryClass.PII,
                DECISION_BLOCKED,
                True,
                False,
                None,
                pii_categories=pii.categories,
                message=REFUSAL_MESSAGES[QueryClass.PII],
            )

        decision: PolicyDecision = classify_intent(question)
        if decision.blocked:
            fund = self._resolve_mention(question)
            message = decision.message
            if fund and decision.query_class in (QueryClass.ADVICE, QueryClass.PERFORMANCE):
                message = f"{message}\nSource: {fund.canonical_url}"
            return RetrievalResult(question, decision.query_class, DECISION_BLOCKED, True, False, fund, message=message)

        fund = resolve_fund(question)
        if fund is None:
            mentioned = matched_categories(question)
            message = MESSAGE_MULTI_FUND if len(mentioned) > 1 else MESSAGE_CLARIFICATION
            return RetrievalResult(question, QueryClass.FACTUAL, DECISION_CLARIFICATION, True, False, None, message=message)

        try:
            vector = self._embed_fn(question)
        except EmbeddingError:
            return RetrievalResult(question, QueryClass.FACTUAL, DECISION_ERROR, True, False, fund, message=MESSAGE_ERROR)

        rows = self._get_store().query(vector, n_results=self._top_k, where={"canonical_url": fund.canonical_url})
        evidence = [EvidenceRow.from_query_row(row) for row in rows if row["distance"] <= self._threshold]
        if not evidence:
            freshness = rows[0]["metadata"]["ingested_at"][:10] if rows else None
            message = f"{MESSAGE_NOT_FOUND}\nSource: {fund.canonical_url}"
            if freshness:
                message = f"{message}\nLast updated from sources: {freshness}"
            return RetrievalResult(
                question,
                QueryClass.FACTUAL,
                DECISION_NOT_FOUND,
                False,
                True,
                fund,
                message=message,
            )

        conflicts = _detect_conflicts(evidence)
        return RetrievalResult(
            question,
            QueryClass.FACTUAL,
            DECISION_FOUND,
            False,
            True,
            fund,
            evidence=evidence,
            conflicts=conflicts,
        )

    def _resolve_mention(self, question: str) -> ResolvedFund | None:
        return resolve_fund(question)