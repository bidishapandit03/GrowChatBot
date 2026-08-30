"""Build the grounded Mistral prompt from retrieved chunk text and metadata only.

Retrieved page text is data, never instruction. Only approved URLs and the user's
question reach the model; no conversation history and no app secrets are included.
"""

from __future__ import annotations

from code.config import APPROVED_SOURCES
from code.rag.retriever import EvidenceRow

_SYSTEM_INSTRUCTION = (
    "You are a facts-only assistant for five approved HDFC mutual fund pages on Groww. "
    "Answer ONLY from the <evidence> blocks in the user message. Never use your own "
    "knowledge, never give investment advice, never predict or compare performance, never "
    "rank or recommend funds, and never mention any URL that is not in the evidence. "
    "Answer in at most three sentences. Return exactly one JSON object with the string "
    "keys \"answer\", \"source\", and \"last_updated\": \"source\" must be the exact "
    "approved URL of the most relevant evidence block, and \"last_updated\" must be the "
    "YYYY-MM-DD date shown in that block. If the evidence does not answer the question, "
    "say so plainly and still give the closest approved URL."
)


def _allowed_url_text() -> str:
    return ", ".join(source["canonical_url"] for source in APPROVED_SOURCES)


def evidence_freshness(evidence: list[EvidenceRow]) -> str | None:
    """Freshness for the displayed source date: newest ingestion date among evidence."""
    dates = [row.ingested_at[:10] for row in evidence]
    return max(dates) if dates else None


def build_messages(question: str, evidence: list[EvidenceRow]) -> list[dict[str, str]]:
    """Return Mistral messages containing only the question and approved retrieved text."""
    blocks = []
    for index, row in enumerate(evidence, start=1):
        blocks.append(
            f"[{index}] Fund: {row.fund_name} ({row.fund_category})\n"
            f"    Section: {row.section_heading}\n"
            f"    Source URL: {row.canonical_url}\n"
            f"    Last updated: {row.ingested_at[:10]}\n"
            f"    Text: {row.chunk_text}"
        )

    user_content = (
        "Approved pages: " + _allowed_url_text() + "\n\n"
        "<evidence>\n" + "\n\n".join(blocks) + "\n</evidence>\n\n"
        "Question: " + question
    )
    return [
        {"role": "system", "content": _SYSTEM_INSTRUCTION},
        {"role": "user", "content": user_content},
    ]