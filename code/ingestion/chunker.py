"""Phase 2 chunking: semantic sections into 300-500 token chunks with 50-75 token overlap."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass

from code.config import APPROVED_SOURCES, CHUNK_OVERLAP_TOKENS, CHUNK_SIZE_TOKENS, CHUNKS_DIR, source_slug
from code.ingestion.pipeline import SourceDocument, read_existing_document

HEADING_RE = re.compile(r"^#{1,6}\s+.+")
HEADING_PREFIX_RE = re.compile(r"^#{1,6}\s+")
OVERVIEW_HEADING = "Overview"
SEGMENT_HEADING_LINE = "## {heading}"


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 characters per subword token, MiniLM-style)."""
    return max(1, (len(text) + 3) // 4)


@dataclass(frozen=True)
class Section:
    heading: str
    lines: list[str]

    def token_count(self) -> int:
        return sum(estimate_tokens(line) for line in self.lines)


def split_sections(source_text: str) -> list[Section]:
    """Group non-blank lines into sections by markdown heading. The title heading maps to Overview."""
    sections: list[Section] = []
    current_heading: str | None = None
    current_lines: list[str] = []
    first_heading_seen = False

    for raw_line in source_text.splitlines():
        if HEADING_RE.match(raw_line):
            if current_heading is not None and current_lines:
                sections.append(Section(current_heading, current_lines))
            if not first_heading_seen:
                current_heading = OVERVIEW_HEADING
                first_heading_seen = True
            else:
                current_heading = HEADING_PREFIX_RE.sub("", raw_line).strip()
            current_lines = []
            continue
        compact = raw_line.strip()
        if compact:
            current_lines.append(compact)

    if current_heading is not None and current_lines:
        sections.append(Section(current_heading, current_lines))
    elif not sections and current_lines:
        sections.append(Section(OVERVIEW_HEADING, current_lines))
    return sections


def _window_spans(tokens: list[int], max_tokens: int, overlap_low: int, overlap_high: int) -> list[tuple[int, int]]:
    """Greedy windows capped at max_tokens with overlap closest to the midpoint target."""
    n = len(tokens)
    if n == 0:
        return []
    prefix = [0] * (n + 1)
    for i, token_count in enumerate(tokens):
        prefix[i + 1] = prefix[i] + token_count
    if prefix[n] <= max_tokens:
        return [(0, n)]

    target = (overlap_low + overlap_high) / 2
    spans: list[tuple[int, int]] = []
    start = 0
    while start < n:
        end = start
        while end < n and prefix[end + 1] - prefix[start] <= max_tokens:
            end += 1
        if end == start:
            end = start + 1
        spans.append((start, end))
        if end >= n:
            break
        best = start
        best_distance = float("inf")
        for candidate in range(start + 1, end):
            distance = abs(prefix[end] - prefix[candidate] - target)
            if distance < best_distance:
                best_distance = distance
                best = candidate
        start = best
    return spans


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    chunk_text: str
    canonical_url: str
    fund_name: str
    fund_category: str
    section_heading: str
    ingested_at: str
    content_hash: str

    def as_dict(self) -> dict:
        return asdict(self)


def _section_chunks(
    section: Section,
    document: SourceDocument,
    order: int,
) -> list[ChunkRecord]:
    max_tokens = CHUNK_SIZE_TOKENS[1]
    overlap_low, overlap_high = CHUNK_OVERLAP_TOKENS
    line_tokens = [estimate_tokens(line) for line in section.lines]
    spans = _window_spans(line_tokens, max_tokens, overlap_low, overlap_high)

    records: list[ChunkRecord] = []
    for span_index, (start, end) in enumerate(spans):
        lines = section.lines[start:end]
        if span_index > 0:
            lines = [SEGMENT_HEADING_LINE.format(heading=section.heading), *lines]
        chunk_text = "\n".join(lines).strip()
        if not chunk_text:
            continue
        chunk_id = hashlib.sha256(
            f"{document.canonical_url}|{document.content_hash}|{section.heading}|{order}".encode("utf-8")
        ).hexdigest()
        records.append(
            ChunkRecord(
                chunk_id=chunk_id,
                chunk_text=chunk_text,
                canonical_url=document.canonical_url,
                fund_name=document.fund_name,
                fund_category=document.fund_category,
                section_heading=section.heading,
                ingested_at=document.ingested_at,
                content_hash=hashlib.sha256(chunk_text.encode("utf-8")).hexdigest(),
            )
        )
        order += 1
    return records


def chunk_document(document: SourceDocument) -> list[ChunkRecord]:
    """Produce all chunks for one source document."""
    records: list[ChunkRecord] = []
    order = 0
    for section in split_sections(document.source_text):
        section_records = _section_chunks(section, document, order)
        records.extend(section_records)
        order += len(section_records)
    return records


def read_chunks(canonical_url: str) -> list[ChunkRecord]:
    """Load persisted chunk records for one source back into ChunkRecord objects."""
    path = CHUNKS_DIR / f"{source_slug(canonical_url)}.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [ChunkRecord(**item) for item in payload]


def write_chunks(records: list[ChunkRecord]) -> None:
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    if not records:
        return
    source = records[0]
    payload = [
        {
            "chunk_id": record.chunk_id,
            "chunk_text": record.chunk_text,
            "canonical_url": record.canonical_url,
            "fund_name": record.fund_name,
            "fund_category": record.fund_category,
            "section_heading": record.section_heading,
            "ingested_at": record.ingested_at,
            "content_hash": record.content_hash,
        }
        for record in records
    ]
    path = CHUNKS_DIR / f"{source_slug(source.canonical_url)}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def chunk_approved_corpus(load_status_ok: bool = True) -> list[list[ChunkRecord]]:
    """Create and persist chunks for every successfully loaded approved source."""
    all_records: list[list[ChunkRecord]] = []
    for source in APPROVED_SOURCES:
        document = read_existing_document(source["canonical_url"])
        if document is None or not document.source_text:
            continue
        records = chunk_document(document)
        write_chunks(records)
        all_records.append(records)
    return all_records