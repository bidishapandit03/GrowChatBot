"""Persistent ChromaDB store: idempotency, source replacement, and query tests."""

from __future__ import annotations

import pytest

from code.config import CHROMA_COLLECTION_NAME
from code.ingestion.embedder import EmbeddedChunk
from code.ingestion.indexer import ChromaVectorStore, ChromaVectorStoreError
from code.ingestion.chunker import ChunkRecord

URL_A = "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth"
URL_B = "https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth"


def make_chunk(text: str, url: str = URL_A, chunk_id: str | None = None, heading: str = "Overview") -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id or f"id-{hash(text)}-{url}",
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


def test_upsert_persists_and_metadata_is_complete(tmp_path):
    store = ChromaVectorStore(path=tmp_path)
    promoted = embed(make_chunk("Expense ratio: 1.02%", URL_A, chunk_id="a1"), [1.0, 0.0, 0.0, 0.0])
    assert store.upsert_chunks([promoted]) == 1
    assert store.count() == 1
    rows = store.query([1.0, 0.0, 0.0, 0.0], n_results=1)
    assert rows[0]["chunk_id"] == "a1"
    assert rows[0]["metadata"]["canonical_url"] == URL_A
    assert rows[0]["metadata"]["fund_name"]
    assert rows[0]["metadata"]["fund_category"]
    assert rows[0]["metadata"]["section_heading"]
    assert rows[0]["metadata"]["ingested_at"]
    assert rows[0]["metadata"]["content_hash"]


def test_rerun_unchanged_ingestion_creates_no_duplicates(tmp_path):
    store = ChromaVectorStore(path=tmp_path)
    first = [embed(make_chunk("SIP 100", URL_A, chunk_id="a"), [1.0, 0.0, 0.0, 0.0])]
    second = [embed(make_chunk("SIP 100", URL_A, chunk_id="a"), [1.0, 0.0, 0.0, 0.0])]
    store.sync_source(first)
    store.sync_source(second)
    assert store.count() == 1


def test_sync_replaces_obsolete_chunks_for_source_only(tmp_path):
    store = ChromaVectorStore(path=tmp_path)
    a_old = embed(make_chunk("old exit load", URL_A, chunk_id="a1"), [1.0, 0.0, 0.0, 0.0])
    a_keep = embed(make_chunk("min sip", URL_A, chunk_id="a2"), [0.0, 1.0, 0.0, 0.0])
    b_chunk = embed(make_chunk("small cap facts", URL_B, chunk_id="b1"), [0.0, 0.0, 1.0, 0.0])

    store.sync_source([a_old, a_keep])
    store.sync_source([b_chunk])
    refreshed = [a_keep, embed(make_chunk("new expense ratio", URL_A, chunk_id="a3"), [0.0, 0.0, 0.0, 1.0])]
    store.sync_source(refreshed)

    ids = {r["chunk_id"] for r in store.query([1.0, 1.0, 1.0, 1.0], n_results=10)}
    assert ids == {"a2", "a3", "b1"}
    assert store.count() == 3


def test_collection_persists_across_restarts(tmp_path):
    ChromaVectorStore(path=tmp_path).sync_source(
        [embed(make_chunk("facts", URL_A, chunk_id="a1"), [1.0, 0.0, 0.0, 0.0])]
    )
    restarted = ChromaVectorStore(path=tmp_path, collection_name=CHROMA_COLLECTION_NAME)
    assert restarted.count() == 1
    rows = restarted.query([1.0, 0.0, 0.0, 0.0], n_results=1)
    assert rows[0]["chunk_id"] == "a1"


def test_query_topk_and_metadata_filter(tmp_path):
    store = ChromaVectorStore(path=tmp_path)
    target = embed(make_chunk("riskometer moderately high", URL_A, chunk_id="t"), [0.99, 0.01, 0.0, 0.0])
    other_a = embed(make_chunk("aum large", URL_A, chunk_id="oa"), [0.1, 0.9, 0.0, 0.0])
    other_b = embed(make_chunk("small cap aum", URL_B, chunk_id="ob"), [0.9, 0.1, 0.0, 0.0])
    store.upsert_chunks([target, other_a, other_b])

    rows = store.query([1.0, 0.0, 0.0, 0.0], n_results=2)
    assert rows[0]["chunk_id"] == "t"
    assert len(rows) == 2

    filtered = store.query([1.0, 0.0, 0.0, 0.0], n_results=5, where={"canonical_url": URL_A})
    assert {r["chunk_id"] for r in filtered} == {"t", "oa"}


def test_query_rejects_zero_results(tmp_path):
    store = ChromaVectorStore(path=tmp_path)
    with pytest.raises(ChromaVectorStoreError):
        store.query([1.0, 0.0, 0.0, 0.0], n_results=0)


def test_clear_source_removes_only_that_source(tmp_path):
    store = ChromaVectorStore(path=tmp_path)
    store.sync_source([embed(make_chunk("a facts", URL_A, chunk_id="a1"), [1.0, 0.0, 0.0, 0.0])])
    store.sync_source([embed(make_chunk("b facts", URL_B, chunk_id="b1"), [1.0, 0.0, 0.0, 0.0])])
    store.clear_source(URL_A)
    remaining = {r["chunk_id"] for r in store.query([1.0, 0.0, 0.0, 0.0], n_results=5)}
    assert remaining == {"b1"}


def test_upsert_failure_keeps_previous_valid_index(tmp_path):
    store = ChromaVectorStore(path=tmp_path)
    store.sync_source([embed(make_chunk("previous facts", URL_A, chunk_id="a1"), [1.0, 0.0, 0.0, 0.0])])
    assert store.count() == 1

    def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    store._collection.upsert = boom
    with pytest.raises(ChromaVectorStoreError, match="upsert failed"):
        store.sync_source([embed(make_chunk("never persisted", URL_A, chunk_id="a2"), [0.0, 1.0, 0.0, 0.0])])
    assert store.count() == 1
    rows = store.query([1.0, 0.0, 0.0, 0.0], n_results=1)
    assert rows[0]["chunk_id"] == "a1"