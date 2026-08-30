"""Chunk size, overlap, metadata, and fact-value integrity tests for Phase 2."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from code.config import CHUNK_OVERLAP_TOKENS, CHUNK_SIZE_TOKENS
from code.ingestion.chunker import (
    ChunkRecord,
    _window_spans,
    chunk_document,
    estimate_tokens,
    split_sections,
    write_chunks,
)
from code.ingestion.cleaner import clean_html_to_source_text
from code.ingestion.pipeline import LOAD_STATUS_SUCCESSFUL, SourceDocument, hash_text

FIXTURE_HTML = (Path(__file__).parent / "fixtures" / "sample_fund_page.html").read_text(encoding="utf-8")
FIXTURE_FUND_NAME = "HDFC Large Cap Fund Direct Growth"
FIXTURE_URL = "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth"

REQUIRED_CHUNK_FIELDS = {
    "chunk_id",
    "chunk_text",
    "canonical_url",
    "fund_name",
    "fund_category",
    "section_heading",
    "ingested_at",
    "content_hash",
}


def make_document(source_text: str, url: str = FIXTURE_URL, fund_name: str = FIXTURE_FUND_NAME) -> SourceDocument:
    return SourceDocument(
        canonical_url=url,
        fund_name=fund_name,
        fund_category="large-cap",
        source_text=source_text,
        ingested_at="2026-08-30T06:24:47Z",
        content_hash=hash_text(source_text),
        load_status=LOAD_STATUS_SUCCESSFUL,
    )


def fixture_document() -> SourceDocument:
    text = clean_html_to_source_text(FIXTURE_HTML, FIXTURE_FUND_NAME, "large-cap")
    return make_document(text)


def test_estimate_tokens_groups_short_lines():
    assert estimate_tokens("Expense ratio: 1.02%") == (len("Expense ratio: 1.02%") + 3) // 4
    assert estimate_tokens("") == 1


def test_split_sections_maps_markdown_headings():
    text = (
        "# HDFC Large Cap Fund Direct Growth\n\n"
        "Scheme name: HDFC Large Cap Fund Direct Growth\n"
        "Expense ratio: 1.02%\n\n"
        "## Historic exit load\n"
        "Exit load as of 2020-01-01: 1% if redeemed within 1 year\n\n"
        "## Fund management\n"
        "Fund manager: Rahul Baijal\n"
    )
    sections = split_sections(text)
    assert [s.heading for s in sections] == ["Overview", "Historic exit load", "Fund management"]
    assert "Scheme name: HDFC Large Cap Fund Direct Growth" in sections[0].lines
    assert "Fund manager: Rahul Baijal" in sections[2].lines


def test_heading_without_space_is_not_a_section():
    text = "# Title\n## Historic exit load\n#2 in India\nLine one\n"
    sections = split_sections(text)
    assert [s.heading for s in sections] == ["Historic exit load"]
    assert "#2 in India" in sections[0].lines
    assert "Line one" in sections[0].lines


def test_blank_lines_are_dropped_from_sections():
    text = "# Title A\n\n\nLine one\n\nLine two\n\n"
    sections = split_sections(text)
    assert sections[0].lines == ["Line one", "Line two"]


def test_window_spans_stay_within_size_and_overlap_bounds():
    line_tokens = [5] * 200
    max_tokens = CHUNK_SIZE_TOKENS[1]
    overlap_low, overlap_high = CHUNK_OVERLAP_TOKENS
    spans = _window_spans(line_tokens, max_tokens, overlap_low, overlap_high)
    assert len(spans) > 2
    for (start, end) in spans:
        assert sum(line_tokens[start:end]) <= max_tokens
    for (_, end), (next_start, _) in zip(spans, spans[1:]):
        overlap = sum(line_tokens[next_start:end])
        assert overlap_low <= overlap <= overlap_high


def test_small_section_is_single_chunk():
    document = fixture_document()
    chunks = [c for c in chunk_document(document) if c.section_heading == "Historic exit load"]
    assert len(chunks) == 1


def test_every_chunk_has_complete_metadata():
    records = chunk_document(fixture_document())
    assert records
    for record in records:
        payload = record.as_dict()
        assert set(payload) == REQUIRED_CHUNK_FIELDS
        assert record.chunk_text.strip()
        assert record.canonical_url == FIXTURE_URL
        assert record.fund_name == FIXTURE_FUND_NAME
        assert record.fund_category == "large-cap"
        assert record.ingested_at
        assert len(record.chunk_id) == 64
        assert len(record.content_hash) == 64


def test_no_duplicate_chunks_within_document():
    records = chunk_document(fixture_document())
    assert len({r.chunk_id for r in records}) == len(records)
    assert len({r.chunk_text for r in records}) == len(records)


def test_fact_value_pairs_are_not_split():
    records = chunk_document(fixture_document())
    facts = ["Expense ratio: 1.02%", "Minimum SIP: 100", "Exit load: Exit load of 1% if redeemed within 1 year"]
    for fact in facts:
        owners = [r for r in records if fact in r.chunk_text]
        assert len(owners) == 1, f"fact should appear in exactly one chunk: {fact}"


def test_chunk_size_cap_enforced_on_large_section():
    long_line = " ".join(["Value"] * 20)
    lines = [f"Fact {i}: {long_line}" for i in range(40)]
    text = f"# Title\n{chr(10).join(lines)}"
    document = make_document(text)
    records = chunk_document(document)
    for record in records:
        assert estimate_tokens(record.chunk_text) <= CHUNK_SIZE_TOKENS[1]


def test_rechunking_is_idempotent():
    document = fixture_document()
    first = [r.as_dict() for r in chunk_document(document)]
    second = [r.as_dict() for r in chunk_document(document)]
    assert first == second


def test_overview_sections_get_more_than_one_window():
    text = "# Title\n" + "\n".join(f"Fact line {i} with padding words" for i in range(120))
    document = make_document(text)
    records = chunk_document(document)
    counts = [r for r in records if r.section_heading == "Overview"]
    assert len(counts) >= 3


def test_write_chunks_to_configured_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("code.ingestion.chunker.CHUNKS_DIR", tmp_path)
    records = chunk_document(fixture_document())
    write_chunks(records)
    target = tmp_path / "hdfc-large-cap-fund-direct-growth.json"
    assert target.exists()
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert len(payload) == len(records)
    ids = [chunk["chunk_id"] for chunk in payload]
    assert len(set(ids)) == len(payload)
    assert all(set(chunk) == REQUIRED_CHUNK_FIELDS for chunk in payload)


def test_chunk_record_is_frozen():
    record = ChunkRecord(
        chunk_id="a",
        chunk_text="b",
        canonical_url=FIXTURE_URL,
        fund_name=FIXTURE_FUND_NAME,
        fund_category="large-cap",
        section_heading="Overview",
        ingested_at="2026-08-30T06:24:47Z",
        content_hash="c",
    )
    with pytest.raises(Exception):
        record.chunk_text = "changed"  # type: ignore[misc]