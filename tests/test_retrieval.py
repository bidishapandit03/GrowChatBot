"""Fund resolution, top-k recall, and relevance-threshold tests."""

from __future__ import annotations

import numpy as np
import pytest

from code.config import APPROVED_SOURCES
from code.ingestion.chunker import ChunkRecord
from code.ingestion.embedder import EmbeddedChunk
from code.ingestion.indexer import ChromaVectorStore
from code.rag.fund_resolver import matched_categories, resolve_fund
from code.rag.policy import QueryClass
from code.rag.retriever import DECISION_BLOCKED, DECISION_CLARIFICATION, DECISION_NOT_FOUND, Retriever

URL_A = "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth"
URL_B = "https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth"
LEGACY_FLEXI_URL = "https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth"
DIM = 4


def unit(*values: float) -> list[float]:
    arr = np.asarray(values, dtype=float)
    return (arr / np.linalg.norm(arr)).tolist()


def make_chunk(text: str, url: str = URL_A, chunk_id: str = "c", heading: str = "Overview") -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        chunk_text=text,
        canonical_url=url,
        fund_name="HDFC Large Cap Fund Direct Growth" if url == URL_A else "HDFC Small Cap Fund Direct Growth",
        fund_category="large-cap" if url == URL_A else "small-cap",
        section_heading=heading,
        ingested_at="2026-08-30T06:24:47Z",
        content_hash=f"hash-{url}",
    )


def embed(chunk: ChunkRecord, vector: list[float]) -> EmbeddedChunk:
    return EmbeddedChunk(chunk=chunk, embedding=vector)


def store_with(items: list[EmbeddedChunk], tmp_path) -> ChromaVectorStore:
    store = ChromaVectorStore(path=tmp_path)
    store.upsert_chunks(items)
    return store


class TestFundResolver:
    def test_legacy_equity_alias_maps_to_flexi_cap(self):
        fund = resolve_fund("What is the lock-in for the HDFC Equity Fund?")
        assert fund is not None
        assert fund.fund_category == "flexi-cap"
        assert fund.canonical_url == LEGACY_FLEXI_URL
        assert fund.fund_name == "HDFC Flexi Cap Fund Direct Growth"

    def test_exact_fund_resolution(self):
        large = resolve_fund("expense ratio of HDFC Large Cap Fund Direct Growth")
        assert large is not None
        assert large.fund_category == "large-cap"
        small = resolve_fund("minimum SIP for hdfc small cap")
        assert small is not None
        assert small.fund_category == "small-cap"
        hybrid = resolve_fund("who runs the hdfc balanced advantage fund?")
        assert hybrid is not None
        assert hybrid.fund_category == "hybrid"

    def test_no_fund_returns_none(self):
        assert resolve_fund("What is the expense ratio?") is None
        assert matched_categories("What is the expense ratio?") == set()

    def test_multiple_funds_returns_none(self):
        assert resolve_fund("Compare the large cap fund and the small cap fund attributes") is None
        assert matched_categories(
            "Compare the large cap fund and the small cap fund attributes"
        ) == {"large-cap", "small-cap"}


class TestRetrieverFlow:
    def test_ambiguous_question_never_embeds(self, tmp_path):
        store = store_with([embed(make_chunk("facts", URL_A, "a1", "Overview"), unit(1, 0, 0, 0))], tmp_path)
        called = {"n": 0}

        def embed_fn(_):
            called["n"] += 1
            return unit(1, 0, 0, 0)

        result = Retriever(store=store, embed_fn=embed_fn).retrieve("What is the expense ratio?")
        assert result.decision == DECISION_CLARIFICATION
        assert result.blocked is True
        assert result.embedded is False
        assert called["n"] == 0

    def test_blocked_intent_never_embeds(self, tmp_path):
        store = store_with([embed(make_chunk("facts", URL_A, "a1", "Overview"), unit(1, 0, 0, 0))], tmp_path)

        def embed_fn(_):
            raise AssertionError("embedding must not run for blocked intent")

        result = Retriever(store=store, embed_fn=embed_fn).retrieve("Should I buy the HDFC small cap fund?")
        assert result.decision == DECISION_BLOCKED
        assert result.query_class == QueryClass.ADVICE
        assert result.embedded is False

    def test_pii_blocks_before_embedding(self, tmp_path):
        store = store_with([embed(make_chunk("facts", URL_A, "a1", "Overview"), unit(1, 0, 0, 0))], tmp_path)

        def embed_fn(_):
            raise AssertionError("embedding must not run for PII")

        result = Retriever(store=store, embed_fn=embed_fn).retrieve("My PAN is ABCDE1234F, check my fund")
        assert result.decision == DECISION_BLOCKED
        assert result.query_class == QueryClass.PII
        assert result.embedded is False

    def test_topk_recall_within_limit_and_sorted(self, tmp_path):
        chunks = [
            embed(make_chunk("benchmark exact", URL_A, "best", "Overview"), unit(0.99, 0.02, 0.02, 0.01)),
            embed(make_chunk("second", URL_A, "c2", "Overview"), unit(0.5, 0.7, 0.2, 0.1)),
            embed(make_chunk("third", URL_A, "c3", "Overview"), unit(0.2, 0.4, 0.9, 0.1)),
            embed(make_chunk("fourth", URL_A, "c4", "Overview"), unit(0.1, 0.2, 0.3, 0.9)),
            embed(make_chunk("fifth", URL_A, "c5", "Overview"), unit(0.05, 0.1, 0.1, 0.1)),
        ]
        store = store_with(chunks, tmp_path)
        result = Retriever(store=store, embed_fn=lambda _: unit(1, 0, 0, 0), top_k=4, threshold=0.99).retrieve(
            "What is the benchmark of the HDFC Large Cap Fund?"
        )
        assert result.decision == "found"
        assert result.fund.fund_category == "large-cap"
        assert result.embedded is True
        assert len(result.evidence) == 4
        assert result.evidence[0].chunk_id == "best"
        distances = [row.distance for row in result.evidence]
        assert distances == sorted(distances)

    def test_metadata_filter_keeps_only_resolved_fund(self, tmp_path):
        alpha = embed(make_chunk("alpha facts", URL_A, "a1", "Overview"), unit(0, 1, 0, 0))
        beta = embed(make_chunk("beta facts", URL_B, "b1", "Overview"), unit(0, 1, 0, 0))
        store = store_with([alpha, beta], tmp_path)
        result = Retriever(store=store, embed_fn=lambda _: unit(0, 1, 0, 0), top_k=4, threshold=0.99).retrieve(
            "facts about hdfc large cap fund"
        )
        assert result.fund.fund_category == "large-cap"
        assert {row.canonical_url for row in result.evidence} == {URL_A}
        assert "b1" not in {row.chunk_id for row in result.evidence}

    def test_weak_evidence_is_rejected(self, tmp_path):
        store = store_with([embed(make_chunk("far chunk", URL_A, "far", "Overview"), unit(0, 0, 1, 0))], tmp_path)
        result = Retriever(store=store, embed_fn=lambda _: unit(1, 0, 0, 0), threshold=0.3).retrieve(
            "who is every ceo under hdfc large cap fund?"
        )
        assert result.decision == DECISION_NOT_FOUND
        assert result.embedded is True
        assert result.evidence == []
        assert "couldn't find" in result.message

    def test_legacy_alias_routed_to_flexi_cap_store(self, tmp_path):
        flexi = embed(make_chunk("Exit load: 1% if redeemed within 1 year", LEGACY_FLEXI_URL, "f1", "Overview"), unit(1, 0, 0, 0))
        store = store_with([flexi], tmp_path)
        result = Retriever(store=store, embed_fn=lambda _: unit(1, 0, 0, 0), threshold=0.99).retrieve(
            "what is the exit load for hdfc equity fund?"
        )
        assert result.decision == "found"
        assert result.fund.fund_category == "flexi-cap"
        assert result.fund.canonical_url == LEGACY_FLEXI_URL
        assert result.evidence[0].chunk_id == "f1"

    def test_conflict_detected_in_structured_sections(self, tmp_path):
        c1 = embed(make_chunk("Expense ratio: 1.02%\nExit load: 1%", URL_A, "c1", "Overview"), unit(1, 0, 0, 0))
        c2 = embed(make_chunk("Expense ratio: 1.05%\nAUM: 100 Cr", URL_A, "c2", "Overview"), unit(0.9, 0.1, 0, 0))
        store = store_with([c1, c2], tmp_path)
        result = Retriever(store=store, embed_fn=lambda _: unit(1, 0, 0, 0), threshold=0.99).retrieve(
            "expense ratio of hdfc large cap fund"
        )
        assert result.conflicts
        assert any("expense ratio" in conflict for conflict in result.conflicts)

    def test_visible_page_text_does_not_create_false_conflicts(self, tmp_path):
        overview = embed(make_chunk("Expense ratio: 1.02%\nNAV: 161.226", URL_A, "o", "Overview"), unit(1, 0, 0, 0))
        visible = embed(
            make_chunk("## Visible page text\nNAV: 28 Aug '26\nExpense ratio", URL_A, "v", "Visible page text"),
            unit(0.9, 0.1, 0, 0),
        )
        store = store_with([overview, visible], tmp_path)
        result = Retriever(store=store, embed_fn=lambda _: unit(1, 0, 0, 0), threshold=0.99).retrieve(
            "expense ratio of hdfc large cap fund"
        )
        assert result.conflicts == []


def test_approved_sources_have_five_categories():
    assert {s["fund_category"] for s in APPROVED_SOURCES} == {"large-cap", "flexi-cap", "elss", "small-cap", "hybrid"}