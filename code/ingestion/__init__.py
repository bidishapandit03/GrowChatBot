"""Offline indexing: load, clean, chunk, embed, and persist approved corpus pages."""

from code.ingestion.pipeline import load_approved_corpus

__all__ = ["load_approved_corpus"]

