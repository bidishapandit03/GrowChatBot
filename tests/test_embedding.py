"""Embedding model loading, preprocessing, batching, and vector-export tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from code.config import EMBEDDING_MODEL
from code.ingestion.chunker import ChunkRecord
from code.ingestion.embedder import (
    EMBEDDINGS_DIR,
    EmbeddedChunk,
    EmbeddingError,
    embed_chunks,
    embed_query,
    encode_texts,
    export_embeddings,
    preprocess_text,
)

URL_A = "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth"
URL_B = "https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth"
DIMS = 384


class FakeEmbedder:
    """Stands in for the SentenceTransformer without network access."""

    def __init__(self, fail: bool = False):
        self.calls: list[tuple[list[str], dict]] = []
        self._fail = fail

    def encode(self, texts: list[str], **kwargs) -> np.ndarray:
        self.calls.append((list(texts), kwargs))
        if self._fail:
            raise RuntimeError("synthetic encode failure")
        rows = np.arange(len(texts) * DIMS, dtype=np.float64).reshape(len(texts), DIMS)
        rows = rows / np.linalg.norm(rows, axis=1, keepdims=True)
        return rows


def make_chunk(text: str, url: str = URL_A, heading: str = "Overview") -> ChunkRecord:
    return ChunkRecord(
        chunk_id=f"id-{hash(text)}-{url}",
        chunk_text=text,
        canonical_url=url,
        fund_name="HDFC Large Cap Fund Direct Growth",
        fund_category="large-cap",
        section_heading=heading,
        ingested_at="2026-08-30T06:24:47Z",
        content_hash=f"hash-{url}",
    )


def test_preprocess_text_matches_document_normalization():
    assert preprocess_text("  What   is the   expense ratio?  ") == "What is the expense ratio?"
    assert preprocess_text("a\n\na\na") == "a\n\na"
    assert preprocess_text("\n\n  \n") == ""


def test_encode_texts_uses_shared_preprocessing(monkeypatch):
    fake = FakeEmbedder()
    monkeypatch.setattr("code.ingestion.embedder.get_embedder", lambda: fake)
    encode_texts(["  What   is   the   min SIP?  "])
    sent, kwargs = fake.calls[0]
    assert sent == ["What is the min SIP?"]
    assert kwargs["normalize_embeddings"] is True
    assert kwargs["convert_to_numpy"] is True


def test_encode_texts_returns_aligned_vectors(monkeypatch):
    fake = FakeEmbedder()
    monkeypatch.setattr("code.ingestion.embedder.get_embedder", lambda: fake)
    vectors = encode_texts(["a", "b", "c"])
    assert len(vectors) == 3
    assert all(isinstance(v, list) and len(v) == DIMS for v in vectors)
    assert vectors[0] != vectors[1]


def test_empty_texts_skip_model_download(monkeypatch):
    monkeypatch.setattr(
        "code.ingestion.embedder.get_embedder",
        lambda: (_ for _ in ()).throw(EmbeddingError("should not load")),
    )
    assert encode_texts([]) == []


def test_encode_failure_raises_embedding_error(monkeypatch):
    fake = FakeEmbedder(fail=True)
    monkeypatch.setattr("code.ingestion.embedder.get_embedder", lambda: fake)
    with pytest.raises(EmbeddingError):
        encode_texts(["any"])


def test_embed_query_uses_document_path(monkeypatch):
    fake = FakeEmbedder()
    monkeypatch.setattr("code.ingestion.embedder.get_embedder", lambda: fake)
    vector = embed_query("  What   is   the   benchmark?  ")
    sent, _ = fake.calls[0]
    assert sent == ["What is the benchmark?"]
    assert len(vector) == DIMS


def test_embed_chunks_atomic_on_mismatch(monkeypatch):
    monkeypatch.setattr(
        "code.ingestion.embedder.encode_texts",
        lambda texts, batch_size=16: [[0.0]] * 1,
    )
    records = [make_chunk("one"), make_chunk("two")]
    with pytest.raises(EmbeddingError, match="does not match"):
        embed_chunks(records)


def test_embed_chunks_empty():
    assert embed_chunks([]) == []


def test_export_groups_embeddings_by_source(tmp_path, monkeypatch):
    monkeypatch.setattr("code.ingestion.embedder.EMBEDDINGS_DIR", tmp_path)
    a = make_chunk("text a", URL_A)
    b = make_chunk("text b", URL_B)
    embedded = [
        EmbeddedChunk(a, [0.1] * DIMS),
        EmbeddedChunk(b, [0.2] * DIMS),
        EmbeddedChunk(make_chunk("text a2", URL_A), [0.3] * DIMS),
    ]
    written = export_embeddings(embedded)
    assert set(written) == {URL_A, URL_B}

    payload_a = json.loads((tmp_path / "hdfc-large-cap-fund-direct-growth.json").read_text(encoding="utf-8"))
    payload_b = json.loads((tmp_path / "hdfc-small-cap-fund-direct-growth.json").read_text(encoding="utf-8"))
    assert payload_a["model"] == EMBEDDING_MODEL
    assert payload_a["canonical_url"] == URL_A
    assert len(payload_a["embeddings"]) == 2
    assert len(payload_b["embeddings"]) == 1
    assert all(len(e["embedding"]) == DIMS for e in payload_a["embeddings"] + payload_b["embeddings"])
    assert len({e["chunk_id"] for e in payload_a["embeddings"]}) == 2