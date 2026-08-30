"""Phase 5 output tests: prompt grounding, validation contract, and fail-closed pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from code.config import APPROVED_SOURCES
from code.ingestion.chunker import ChunkRecord
from code.ingestion.embedder import EmbeddedChunk
from code.ingestion.indexer import ChromaVectorStore
from code.rag.generator import GenerationError, get_api_key, generate, strip_json_fences
from code.rag.pipeline import (
    MESSAGE_UNABLE_TO_ANSWER,
    AnswerResult,
    answer,
)
from code.rag.prompt import build_messages, evidence_freshness
from code.rag.retriever import DECISION_FOUND, DECISION_NOT_FOUND, Retriever
from code.rag.validator import ValidationError, ValidatedAnswer, allowed_urls, count_sentences, validate_answer

URL_A = "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth"
OTHER_URL = "https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth"
FORBIDDEN_URL = "https://finance.example/unapproved-page"

OVERRIDE_URL = None


def unit(*values: float) -> list[float]:
    arr = np.asarray(values, dtype=float)
    return (arr / np.linalg.norm(arr)).tolist()


def chunk(text: str, chunk_id: str = "c", url: str = URL_A, heading: str = "Overview", ingested_at: str = "2026-08-30T06:24:47Z") -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        chunk_text=text,
        canonical_url=url,
        fund_name="HDFC Large Cap Fund Direct Growth" if url == URL_A else "HDFC Small Cap Fund Direct Growth",
        fund_category="large-cap" if url == URL_A else "small-cap",
        section_heading=heading,
        ingested_at=ingested_at,
        content_hash=f"hash-{chunk_id}",
    )


class TestPrompt:
    def test_evidence_is_data_not_instructions(self):
        messages = build_messages("What is the benchmark?", [
            type(
                "E",
                (),
                {
                    "chunk_id": "c1", "chunk_text": "Benchmark: Nifty 50", "canonical_url": URL_A,
                    "fund_name": "HDFC Large Cap", "fund_category": "large-cap",
                    "section_heading": "Overview", "ingested_at": "2026-08-30T06:24:47Z",
                    "distance": 0.3,
                },
            )()
        ])
        system = messages[0]["content"]
        user = messages[1]["content"]
        assert "facts-only" in system
        assert "at most three sentences" in system
        assert URL_A in user
        assert "Benchmark: Nifty 50" in user
        assert "What is the benchmark?" in user
        assert "<evidence>" in user
        assert "ignore your rules" not in system.lower()

    def test_evidence_freshness_uses_latest_date(self):
        rows = [
            type("E", (), {"ingested_at": "2026-08-28T00:00:00Z"})(),
            type("E", (), {"ingested_at": "2026-08-30T00:00:00Z"})(),
        ]
        assert evidence_freshness(rows) == "2026-08-30"


class TestValidator:
    def test_valid_answer_passes(self):
        valid = validate_answer(
            {"answer": "The expense ratio is 0.55%.", "source": URL_A, "last_updated": "2026-08-30"},
            [type("E", (), {"ingested_at": "2026-08-30T06:24:47Z"})()],
        )
        assert isinstance(valid, ValidatedAnswer)
        assert valid.source == URL_A

    def test_four_sentences_rejected(self):
        text = "One fact. Two facts. Three facts. Four facts."
        with pytest.raises(ValidationError):
            validate_answer({"answer": text, "source": URL_A, "last_updated": "2026-08-30"}, None)

    def test_unapproved_source_rejected(self):
        with pytest.raises(ValidationError):
            validate_answer({"answer": "A fact.", "source": FORBIDDEN_URL, "last_updated": "2026-08-30"}, None)

    def test_missing_keys_rejected(self):
        with pytest.raises(ValidationError):
            validate_answer({"answer": "A fact.", "source": URL_A}, None)

    def test_bad_freshness_rejected(self):
        with pytest.raises(ValidationError):
            validate_answer({"answer": "A fact.", "source": URL_A, "last_updated": "2026-08-29"}, [
                type("E", (), {"ingested_at": "2026-08-30T06:24:47Z"})()
            ])

    def test_decimal_figures_do_not_inflate_sentence_count(self):
        assert count_sentences("Expense ratio is 1.02% and exit load is 1%.") <= 2

    def test_allowed_urls_matches_config(self):
        assert allowed_urls() == {s["canonical_url"] for s in APPROVED_SOURCES}


class TestGenerator:
    def test_strip_json_fences(self):
        assert strip_json_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
        assert strip_json_fences('{"a": 1}') == '{"a": 1}'

    def test_missing_key_raises_before_http(self, monkeypatch):
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        monkeypatch.delenv("DOCUMENTAIRE", raising=False)
        assert get_api_key() is None
        with pytest.raises(GenerationError):
            generate([{"role": "user", "content": "hi"}])


def _fake_evidence_rows():
    obj = type(
        "E",
        (),
        {
            "chunk_id": "c1", "chunk_text": "Expense ratio: 0.55%", "canonical_url": URL_A,
            "fund_name": "HDFC Large Cap", "fund_category": "large-cap",
            "section_heading": "Overview", "ingested_at": "2026-08-30T06:24:47Z", "distance": 0.3,
        },
    )
    return [obj()]


def _retriever_that_returns(decision: str, conflicts=(), evidence=None, message=None):
    class _Fake:
        def retrieve(self, question):
            fund = type("F", (), {"canonical_url": URL_A, "fund_name": "HDFC Large Cap", "fund_category": "large-cap"})()
            return type(
                "R",
                (),
                {
                    "question": question, "query_class": None, "decision": decision,
                    "blocked": False, "embedded": True, "fund": fund,
                    "evidence": evidence if evidence is not None else _fake_evidence_rows(),
                    "conflicts": conflicts, "message": message,
                },
            )()

    return _Fake()


class TestPipeline:
    def test_happy_path_is_grounded(self):
        def fake_generate(messages):
            assert messages
            return {"answer": "The expense ratio is 0.55%.",
                    "source": URL_A, "last_updated": "2026-08-30"}

        result: AnswerResult = answer("what is the expense ratio?",
                                      retriever=_retriever_that_returns(DECISION_FOUND),
                                      generate_fn=fake_generate)
        assert result.grounded is True
        assert "0.55%" in result.answer

    def test_validation_failure_falls_back(self):
        def fake_generate(_):
            return {"answer": "This is one fact. This is two facts. This is three. This is four.",
                    "source": URL_A, "last_updated": "2026-08-30"}

        result = answer("what is the expense ratio?", retriever=_retriever_that_returns(DECISION_FOUND), generate_fn=fake_generate)
        assert result.grounded is False
        assert result.answer == MESSAGE_UNABLE_TO_ANSWER

    def test_generation_error_falls_back(self):
        def fake_generate(_):
            raise GenerationError("boom")

        result = answer("what is the expense ratio?", retriever=_retriever_that_returns(DECISION_FOUND), generate_fn=fake_generate)
        assert result.grounded is False
        assert result.answer == MESSAGE_UNABLE_TO_ANSWER

    def test_conflicts_never_call_the_model(self):
        def fake_generate(_):
            raise AssertionError("conflicts must bypass generation")

        result = answer("expense ratio?", retriever=_retriever_that_returns(DECISION_FOUND, conflicts=["expense ratio: 0.55%, 0.60%"]), generate_fn=fake_generate)
        assert result.grounded is False
        assert "conflicting values" in result.answer
        assert "expense ratio" in result.answer

    def test_not_found_returns_safe_message_without_generation(self):
        called = {"n": 0}

        def gen(_):
            called["n"] += 1
            return {"answer": "x", "source": URL_A, "last_updated": "2026-08-30"}

        result = answer(
            "who supports this fund by phone?",
            retriever=_retriever_that_returns(DECISION_NOT_FOUND, evidence=[], message="nope"),
            generate_fn=gen,
        )
        assert result.grounded is False
        assert result.answer == "nope"
        assert called["n"] == 0


def test_real_episode_store_found_flow(tmp_path):
    store = ChromaVectorStore(path=tmp_path)
    store.upsert_chunks([EmbeddedChunk(chunk("Expense ratio: 0.55%\nExit load: 1%", "c1", URL_A, "Overview"), unit(1, 0, 0, 0))])
    retriever = Retriever(store=store, embed_fn=lambda _: unit(1, 0, 0, 0), threshold=0.99)

    def fake_generate(messages):
        return {"answer": "The expense ratio is 0.55%.", "source": URL_A, "last_updated": "2026-08-30"}

    result = answer("what is the expense ratio of hdfc large cap fund?", retriever=retriever, generate_fn=fake_generate)
    assert result.grounded is True
    assert result.retrieval.decision == DECISION_FOUND