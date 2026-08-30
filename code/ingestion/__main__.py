"""Run ingestion phases:
    python -m code.ingestion            Phase 1 data loading
    python -m code.ingestion --chunk    Phase 2 chunking of existing documents
    python -m code.ingestion --embed    Phase 3 embedding and vector export
    python -m code.ingestion --index    Phase 4 embed and upsert into ChromaDB
"""

from __future__ import annotations

import argparse

from code.ingestion.chunker import chunk_approved_corpus
from code.ingestion.embedder import embed_approved_chunks
from code.ingestion.indexer import ChromaVectorStore, index_approved_chunks
from code.ingestion.pipeline import load_approved_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="HDFC fund corpus ingestion")
    parser.add_argument("--chunk", action="store_true", help="Chunk existing source documents (Phase 2)")
    parser.add_argument("--embed", action="store_true", help="Embed persisted chunks and export vectors (Phase 3)")
    parser.add_argument("--index", action="store_true", help="Embed and upsert chunks into ChromaDB (Phase 4)")
    args = parser.parse_args()

    if args.index:
        indexed = index_approved_chunks()
        count = ChromaVectorStore().count()
        print(f"indexed {indexed} chunks across the approved corpus; collection size = {count}")
        return

    if args.embed:
        embedded = embed_approved_chunks()
        by_source: dict[str, int] = {}
        for item in embedded:
            by_source[item.chunk.fund_category] = by_source.get(item.chunk.fund_category, 0) + 1
        for category, count in by_source.items():
            print(f"embedded {count:3d} chunks  {category:12} dims={len(embedded[0].embedding) if embedded else 0}")
        return

    if args.chunk:
        per_source = chunk_approved_corpus()
        for records in per_source:
            source = records[0] if records else None
            if source is None:
                continue
            total_tokens = sum(len(chunk.chunk_text.split()) for chunk in records)
            print(
                f"chunked {len(records):3d} chunks  {source.fund_category:12} "
                f"{source.canonical_url} words={total_tokens}"
            )
        return

    documents = load_approved_corpus()
    for document in documents:
        preview = document.source_text[:80].replace("\n", " ") if document.source_text else ""
        print(
            f"{document.load_status:20} {document.fund_category:12} "
            f"{document.canonical_url} hash={document.content_hash[:12]} {preview}"
        )


if __name__ == "__main__":
    main()