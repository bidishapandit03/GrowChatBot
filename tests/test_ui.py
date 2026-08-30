"""Frontend test: layout renders and example questions produce an assistant answer."""

from __future__ import annotations

import pytest

from code.rag.fund_resolver import ResolvedFund
from code.rag.pipeline import AnswerResult
from code.rag.policy import QueryClass
from code.rag.retriever import EvidenceRow, RetrievalResult

URL_A = "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth"


def _row() -> EvidenceRow:
    return EvidenceRow(
        chunk_id="c1",
        chunk_text="Expense ratio: 0.55%\nExit load: 1%",
        canonical_url=URL_A,
        fund_name="HDFC Large Cap Fund Direct Growth",
        fund_category="large-cap",
        section_heading="Overview",
        ingested_at="2026-08-30T06:24:47Z",
        distance=0.3,
    )


def test_link_sources_turns_approved_source_line_into_markdown():
    from code.app import _link_sources

    text = f"An answer.\nSource: {URL_A}\nLast updated from sources: 2026-08-30"
    rendered = _link_sources(text)
    assert f"Source: [{URL_A}]({URL_A})" in rendered


def test_link_sources_leaves_unapproved_url_untouched():
    from code.app import _link_sources

    text = "An answer.\nSource: https://evil.example/payload"
    rendered = _link_sources(text)
    assert "https://evil.example/payload" in rendered
    assert "](" not in rendered


def test_fake_answer_result_builds_grounded_entry():
    fund = ResolvedFund(URL_A, "HDFC Large Cap Fund Direct Growth", "large-cap")
    retrieval = RetrievalResult(
        question="What is the expense ratio?",
        query_class=QueryClass.FACTUAL,
        decision="found",
        blocked=False,
        embedded=True,
        fund=fund,
        evidence=[_row()],
    )
    result = AnswerResult(
        question="What is the expense ratio?",
        retrieval=retrieval,
        answer="The expense ratio is 0.55%.",
        grounded=True,
    )
    assert result.grounded is True
    assert result.retrieval.evidence[0].chunk_id == "c1"


def test_dashboard_module_imports():
    import code.app  # noqa: F401


def test_app_renders_and_answers_via_example_button(monkeypatch):
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        pytest.skip("Streamlit AppTest not available")

    from code.app import EXAMPLE_QUESTIONS
    from code.rag import pipeline

    fund = ResolvedFund(URL_A, "HDFC Large Cap Fund Direct Growth", "large-cap")

    def fake_answer(q):
        retrieval = RetrievalResult(
            question=q,
            query_class=QueryClass.FACTUAL,
            decision="found",
            blocked=False,
            embedded=True,
            fund=fund,
            evidence=[_row()],
        )
        return AnswerResult(
            question=q,
            retrieval=retrieval,
            answer=f"The expense ratio is 0.55%.\nSource: {URL_A}\nLast updated from sources: 2026-08-30",
            grounded=True,
        )

    monkeypatch.setattr(pipeline, "answer", fake_answer)
    at = AppTest.from_file("code/app.py", default_timeout=60)
    at.run()
    assert not at.exception
    assert at.title[0].value == "HDFC Mutual Fund Facts Assistant"
    assert len(at.button) == 3
    assert len(at.info) == 1 and "facts-only" in str(at.info[0].value).lower()
    assert len(at.warning) == 1 and "PAN" in str(at.warning[0].value)
    assert len(at.caption) >= 2

    at.button[0].click().run()
    assert not at.exception
    assert EXAMPLE_QUESTIONS[0] in at.session_state["messages"][0]["text"]
    rendered = " ".join(str(m.value) for m in at.markdown)
    assert "expense ratio" in rendered.lower()
    assert f"[{URL_A}]({URL_A})" in rendered