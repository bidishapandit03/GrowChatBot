"""Phase 1 pipeline: allowlist → fetch → validate URL → clean → source document."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from code.config import (
    APPROVED_SOURCES,
    LOAD_STATUS_SUCCESSFUL,
    LOAD_STATUS_UNAVAILABLE,
    LOAD_STATUS_VALIDATION_FAILED,
    RAW_DOCUMENTS_DIR,
    source_slug,
)
from code.ingestion.cleaner import clean_html_to_source_text
from code.ingestion.loader import fetch_approved_source, write_raw_html

REQUIRED_DOCUMENT_FIELDS = (
    "canonical_url",
    "fund_name",
    "fund_category",
    "source_text",
    "ingested_at",
    "content_hash",
    "load_status",
)


@dataclass
class SourceDocument:
    canonical_url: str
    fund_name: str
    fund_category: str
    source_text: str
    ingested_at: str
    content_hash: str
    load_status: str

    def as_dict(self) -> dict:
        return asdict(self)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def document_path(canonical_url: str) -> Path:
    return RAW_DOCUMENTS_DIR / f"{source_slug(canonical_url)}.json"


def attempt_path(canonical_url: str) -> Path:
    return RAW_DOCUMENTS_DIR / f"{source_slug(canonical_url)}.attempt.json"


def html_path(canonical_url: str) -> Path:
    return RAW_HTML_DIR / f"{source_slug(canonical_url)}.html"


def read_existing_document(canonical_url: str) -> SourceDocument | None:
    path = document_path(canonical_url)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return SourceDocument(**{field: payload.get(field, "") for field in REQUIRED_DOCUMENT_FIELDS})


def write_document(document: SourceDocument) -> None:
    RAW_DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    document_path(document.canonical_url).write_text(
        json.dumps(document.as_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_attempt(canonical_url: str, payload: dict) -> None:
    RAW_DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    attempt_path(canonical_url).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _unavailable_or_invalid(source: dict, load_status: str, error: str, ingested_at: str) -> SourceDocument:
    existing = read_existing_document(source["canonical_url"])
    if existing and existing.load_status == LOAD_STATUS_SUCCESSFUL and existing.source_text:
        return existing
    return SourceDocument(
        canonical_url=source["canonical_url"],
        fund_name=source["fund_name"],
        fund_category=source["fund_category"],
        source_text="",
        ingested_at=ingested_at,
        content_hash=hash_text(""),
        load_status=load_status,
    )


def load_one_source(source: dict, fetch_result=None) -> SourceDocument:
    ingested_at = utc_now_iso()
    result = fetch_result if fetch_result is not None else fetch_approved_source(source)

    if not result.ok or not result.html:
        status = result.load_status or LOAD_STATUS_UNAVAILABLE
        document = _unavailable_or_invalid(source, status, result.error or "Fetch failed", ingested_at)
        if not (document.load_status == LOAD_STATUS_SUCCESSFUL and document.source_text):
            write_document(document)
        write_attempt(
            source["canonical_url"],
            {
                "canonical_url": source["canonical_url"],
                "load_status": status,
                "error": result.error,
                "attempted_at": ingested_at,
                "retained_previous_successful": document.load_status == LOAD_STATUS_SUCCESSFUL
                and bool(document.source_text)
                and document.ingested_at != ingested_at,
            },
        )
        return document

    source_text = clean_html_to_source_text(result.html, source["fund_name"], source["fund_category"])
    if not source_text:
        document = _unavailable_or_invalid(
            source,
            LOAD_STATUS_VALIDATION_FAILED,
            "Cleaning produced empty source text.",
            ingested_at,
        )
        if not (document.load_status == LOAD_STATUS_SUCCESSFUL and document.source_text):
            write_document(document)
        write_attempt(
            source["canonical_url"],
            {
                "canonical_url": source["canonical_url"],
                "load_status": LOAD_STATUS_VALIDATION_FAILED,
                "error": "Cleaning produced empty source text.",
                "attempted_at": ingested_at,
                "retained_previous_successful": document.load_status == LOAD_STATUS_SUCCESSFUL
                and bool(document.source_text),
            },
        )
        return document

    write_raw_html(source["canonical_url"], result.html)
    document = SourceDocument(
        canonical_url=source["canonical_url"],
        fund_name=source["fund_name"],
        fund_category=source["fund_category"],
        source_text=source_text,
        ingested_at=ingested_at,
        content_hash=hash_text(source_text),
        load_status=LOAD_STATUS_SUCCESSFUL,
    )
    write_document(document)
    write_attempt(
        source["canonical_url"],
        {
            "canonical_url": source["canonical_url"],
            "load_status": LOAD_STATUS_SUCCESSFUL,
            "error": None,
            "attempted_at": ingested_at,
            "content_hash": document.content_hash,
        },
    )
    return document


def load_approved_corpus() -> list[SourceDocument]:
    """Attempt exactly the five allowlisted sources. Does not accept a user-supplied URL."""
    documents = [load_one_source(source) for source in APPROVED_SOURCES]
    summary = {
        "attempted": [source["canonical_url"] for source in APPROVED_SOURCES],
        "attempt_count": len(APPROVED_SOURCES),
        "results": [
            {
                "canonical_url": document.canonical_url,
                "load_status": document.load_status,
                "content_hash": document.content_hash,
                "ingested_at": document.ingested_at,
            }
            for document in documents
        ],
        "run_at": utc_now_iso(),
    }
    RAW_DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DOCUMENTS_DIR / "_run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return documents
