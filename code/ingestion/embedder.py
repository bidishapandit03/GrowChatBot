"""Phase 3 embedding service shared by the document-indexing and query paths."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from code.config import EMBEDDING_MODEL, EMBEDDINGS_DIR, HF_CACHE_DIR, source_slug
from code.ingestion.chunker import ChunkRecord
from code.ingestion.cleaner import normalize_whitespace

# Keep the model cache inside the project so it is part of the deployed build and the
# app never tries to download it at runtime. Must be set before the model is created.
os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(HF_CACHE_DIR / "hub")

# Keep memory low on small containers (e.g. Render free tier) and avoid tokenizer
# thread races. These must be set before the model is first created.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

EMBED_BATCH_SIZE = 16

_model = None
_model_lock = threading.Lock()


class EmbeddingError(RuntimeError):
    """Raised when the embedding model cannot be loaded, run, or validated."""


def preprocess_text(text: str) -> str:
    """Shared preprocessing so document chunks and user questions embed identically."""
    return normalize_whitespace(text)


def get_embedder():
    """Return the process-wide sentence-transformers model, loading it once."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                try:
                    import socket

                    from sentence_transformers import SentenceTransformer

                    try:
                        import torch

                        torch.set_num_threads(1)
                    except Exception:
                        pass
                    # Bound any network stall during model download/load so a missing
                    # cache surfaces as an error instead of an infinite spinner.
                    previous_timeout = socket.getdefaulttimeout()
                    socket.setdefaulttimeout(60)
                    try:
                        _model = SentenceTransformer(EMBEDDING_MODEL)
                    finally:
                        socket.setdefaulttimeout(previous_timeout)
                except Exception as exc:
                    raise EmbeddingError(
                        f"Failed to load embedding model {EMBEDDING_MODEL}: {exc}"
                    ) from exc
    return _model


def encode_texts(texts: list[str], batch_size: int = EMBED_BATCH_SIZE) -> list[list[float]]:
    """Embed texts in batches; vectors align with the input order."""
    if not texts:
        return []
    model = get_embedder()
    normalized = [preprocess_text(text) for text in texts]
    try:
        vectors = model.encode(
            normalized,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    except Exception as exc:
        raise EmbeddingError(f"Embedding failed: {exc}") from exc
    return [np.asarray(vector).tolist() for vector in vectors]


def embed_query(text: str) -> list[float]:
    """Embed a single user question after it has passed the PII and policy gate."""
    vector = encode_texts([text])
    return vector[0] if vector else []


@dataclass(frozen=True)
class EmbeddedChunk:
    chunk: ChunkRecord
    embedding: list[float]

    def as_dict(self) -> dict:
        return {
            "chunk_id": self.chunk.chunk_id,
            "canonical_url": self.chunk.canonical_url,
            "fund_name": self.chunk.fund_name,
            "fund_category": self.chunk.fund_category,
            "section_heading": self.chunk.section_heading,
            "embedding": self.embedding,
        }


def embed_chunks(records: list[ChunkRecord]) -> list[EmbeddedChunk]:
    """Embed every chunk and fail atomically if the model output cannot be aligned."""
    if not records:
        return []
    vectors = encode_texts([record.chunk_text for record in records])
    if len(vectors) != len(records):
        raise EmbeddingError("Embedding output count does not match input chunk count.")
    return [EmbeddedChunk(record, vector) for record, vector in zip(records, vectors)]


def export_embeddings(embedded: list[EmbeddedChunk], output_dir: Path | None = None) -> dict[str, Path]:
    """Write one JSON export per source under data/embeddings/. Returns url -> path."""
    output_dir = output_dir or EMBEDDINGS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    by_source: dict[str, list[EmbeddedChunk]] = {}
    for item in embedded:
        by_source.setdefault(item.chunk.canonical_url, []).append(item)

    written: dict[str, Path] = {}
    for canonical_url, items in by_source.items():
        payload = {
            "model": EMBEDDING_MODEL,
            "canonical_url": canonical_url,
            "fund_name": items[0].chunk.fund_name,
            "fund_category": items[0].chunk.fund_category,
            "embeddings": [item.as_dict() for item in items],
        }
        path = output_dir / f"{source_slug(canonical_url)}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        written[canonical_url] = path
    return written


def embed_approved_chunks() -> list[EmbeddedChunk]:
    """Embed every persisted chunk for the approved corpus and export the vectors."""
    from code.config import APPROVED_SOURCES
    from code.ingestion.chunker import read_chunks

    all_embedded: list[EmbeddedChunk] = []
    for source in APPROVED_SOURCES:
        records = read_chunks(source["canonical_url"])
        if records:
            all_embedded.extend(embed_chunks(records))
    export_embeddings(all_embedded)
    return all_embedded