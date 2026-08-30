"""Phase 4 vector store: embed persisted chunks and upsert into persistent ChromaDB."""

from __future__ import annotations

from pathlib import Path

from code.config import CHROMA_COLLECTION_NAME, CHROMA_DIR
from code.ingestion.embedder import EmbeddedChunk

METADATA_FIELDS = (
    "canonical_url",
    "fund_name",
    "fund_category",
    "section_heading",
    "ingested_at",
    "content_hash",
)


class ChromaVectorStoreError(RuntimeError):
    """Raised when a ChromaDB operation cannot complete successfully."""


def _metadata(chunk) -> dict[str, str]:
    return {field: getattr(chunk, field) for field in METADATA_FIELDS}


class ChromaVectorStore:
    """One persistent local ChromaDB collection. Chunks upsert by stable chunk_id.

    Re-running unchanged ingestion is idempotent because chunk_id embeds the
    document content hash. A source refresh adds/updates its new chunks before
    deleting obsolete entries from that same source, so a failed refresh never
    removes the previously valid index.
    """

    def __init__(self, path: Path | None = None, collection_name: str | None = None):
        import chromadb

        self._path = path or CHROMA_DIR
        self._client = chromadb.PersistentClient(path=str(self._path))
        self._collection = self._client.get_or_create_collection(
            name=collection_name or CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )

    def upsert_chunks(self, embedded: list[EmbeddedChunk]) -> int:
        if not embedded:
            return 0
        try:
            self._collection.upsert(
                ids=[item.chunk.chunk_id for item in embedded],
                documents=[item.chunk.chunk_text for item in embedded],
                embeddings=[item.embedding for item in embedded],
                metadatas=[_metadata(item.chunk) for item in embedded],
            )
        except Exception as exc:
            raise ChromaVectorStoreError(f"ChromaDB upsert failed: {exc}") from exc
        return len(embedded)

    def _ids_for_url(self, canonical_url: str) -> set[str]:
        try:
            result = self._collection.get(where={"canonical_url": canonical_url}, include=[])
        except Exception as exc:
            raise ChromaVectorStoreError(f"ChromaDB read failed: {exc}") from exc
        return set(result.get("ids", []))

    def sync_source(self, embedded: list[EmbeddedChunk]) -> int:
        """Replace one source's stored index: upsert new chunks, then drop only obsolete ones."""
        upserted = self.upsert_chunks(embedded)
        if not embedded:
            return 0
        canonical_url = embedded[0].chunk.canonical_url
        new_ids = {item.chunk.chunk_id for item in embedded}
        obsolete = self._ids_for_url(canonical_url) - new_ids
        if obsolete:
            try:
                self._collection.delete(ids=sorted(obsolete))
            except Exception as exc:
                raise ChromaVectorStoreError(f"ChromaDB delete failed: {exc}") from exc
        return upserted

    def clear_source(self, canonical_url: str) -> int:
        """Remove every stored entry for one source."""
        try:
            self._collection.delete(where={"canonical_url": canonical_url})
        except Exception as exc:
            raise ChromaVectorStoreError(f"ChromaDB delete failed: {exc}") from exc
        return 0

    def query(
        self,
        embedding: list[float],
        n_results: int,
        where: dict | None = None,
    ) -> list[dict]:
        if n_results < 1:
            raise ChromaVectorStoreError("n_results must be at least 1.")
        kwargs: dict = {"where": where} if where is not None else {}
        try:
            result = self._collection.query(
                query_embeddings=[embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
                **kwargs,
            )
        except Exception as exc:
            raise ChromaVectorStoreError(f"ChromaDB query failed: {exc}") from exc

        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        rows: list[dict] = []
        for index, chunk_id in enumerate(ids):
            rows.append(
                {
                    "chunk_id": chunk_id,
                    "chunk_text": documents[index],
                    "metadata": metadatas[index],
                    "distance": distances[index],
                }
            )
        return rows

    def count(self) -> int:
        try:
            return self._collection.count()
        except Exception as exc:
            raise ChromaVectorStoreError(f"ChromaDB count failed: {exc}") from exc


def index_approved_chunks() -> int:
    """Embed every persisted chunk and sync it into data/chroma/ for the five sources."""
    from code.config import APPROVED_SOURCES
    from code.ingestion.chunker import read_chunks
    from code.ingestion.embedder import embed_chunks

    store = ChromaVectorStore()
    total = 0
    for source in APPROVED_SOURCES:
        records = read_chunks(source["canonical_url"])
        if not records:
            store.clear_source(source["canonical_url"])
            continue
        embedded = embed_chunks(records)
        total += store.sync_source(embedded)
    return total