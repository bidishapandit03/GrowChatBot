"""Phase 6 evaluation harness tests (pure logic; no network, no embeddings)."""

from __future__ import annotations

import json

from code.evaluate import (
    RETRIEVAL_CASES_FILE,
    assess_case,
    calibrate,
    is_refusal,
    load_cases,
    run_evaluation,
    summarize,
)
from code.rag.retriever import EvidenceRow, RetrievalResult

ALLOWED_OUTCOMES = {"found", "not_found", "blocked", "clarification"}
ALLOWED_CLASSES = {
    "factual",
    "advice",
    "performance",
    "allocation",
    "account_support",
    "outside_scope",
    "pii",
    "injection",
    "clarification",
}
REQUIRED_KEYS = {
    "id",
    "category",
    "question",
    "expected_class",
    "expected_fund",
    "expected_outcome",
    "expected_url",
    "expected_fact_text",
    "embedding_allowed",
    "generation_allowed",
    "soft",
}


def _row(url: str, text: str, distance: float = 0.4, heading: str = "Overview") -> EvidenceRow:
    return EvidenceRow(
        chunk_id=f"c-{len(url)}-{len(text)}",
        chunk_text=text,
        canonical_url=url,
        fund_name="HDFC test fund",
        fund_category="large-cap",
        section_heading=heading,
        ingested_at="2026-08-30T00:00:00",
        distance=distance,
    )


def _found_result(url: str, text: str, distance: float = 0.4):
    return RetrievalResult(
        question="q",
        query_class="factual",
        decision="found",
        blocked=False,
        embedded=True,
        fund=None,
        evidence=[_row(url, text, distance)],
    )


def test_dataset_schema_valid():
    cases = load_cases()
    assert cases, "dataset must not be empty"
    for case in cases:
        missing = REQUIRED_KEYS - case.keys()
        assert not missing, f"case {case['id']} missing keys {missing}"
        assert case["expected_outcome"] in ALLOWED_OUTCOMES
        assert case["expected_class"] in ALLOWED_CLASSES
        assert isinstance(case["question"], str) and case["question"].strip()
        assert isinstance(case["soft"], bool)
        assert isinstance(case["embedding_allowed"], bool)
        assert isinstance(case["generation_allowed"], bool)
        if case["expected_outcome"] == "blocked":
            assert case["embedding_allowed"] is False, f"blocked case {case['id']} must not embed"
            assert case["generation_allowed"] is False, f"blocked case {case['id']} must not generate"
            assert case["expected_fund"] is None
            assert case["expected_url"] is None
        if case["expected_outcome"] in ("found", "not_found"):
            assert case["embedding_allowed"] is True
            assert case["expected_fund"] is not None
            assert case["expected_url"] is not None
        if case["expected_outcome"] == "clarification":
            assert case["expected_fund"] is None and case["expected_url"] is None


def test_dataset_has_required_coverage():
    cases = load_cases()
    outcomes = {c["expected_outcome"] for c in cases}
    assert outcomes == ALLOWED_OUTCOMES
    classes = {c["expected_class"] for c in cases}
    assert {"advice", "performance", "allocation", "account_support", "outside_scope", "pii", "injection"} <= classes
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)), "duplicate case ids"
    hard = [c for c in cases if not c["soft"]]
    blocked = [c for c in hard if c["expected_outcome"] == "blocked"]
    found = [c for c in hard if c["expected_outcome"] == "found"]
    assert len(blocked) >= 15, "need a meaningful safety test set"
    assert len(found) >= 12, "need a meaningful recall test set"


def test_assess_found_passes_when_evidence_present():
    case = {
        "id": "x",
        "expected_outcome": "found",
        "expected_class": "factual",
        "expected_fund": "large-cap",
        "expected_url": "https://example.com/f",
        "expected_fact_text": "Expense ratio: 1.02%",
    }
    result = RetrievalResult(
        question="q",
        query_class="factual",
        decision="found",
        blocked=False,
        embedded=True,
        fund=None,
        evidence=[
            _row("https://example.com/f", "Overview Expense ratio: 1.02% AUM 10", 0.6),
            _row("https://example.com/f", "Overview Expense ratio: 1.02%", 0.3),
        ],
    )
    check = assess_case(
        case, "found", "factual", "large-cap", True, result.evidence
    )
    assert check.passed
    assert check.best_distance == 0.3


def test_assess_found_fails_when_url_present_but_fact_absent():
    case = {
        "id": "x",
        "expected_outcome": "found",
        "expected_class": "factual",
        "expected_fund": "large-cap",
        "expected_url": "https://example.com/f",
        "expected_fact_text": "Expense ratio: 1.02%",
    }
    check = assess_case(
        case, "found", "factual", "large-cap", True, [_row("https://example.com/f", "AUM only", 0.3)]
    )
    assert not check.passed
    assert check.checks["recall"] is False


def test_assess_found_fails_when_mismatched_fund_url():
    case = {
        "id": "x",
        "expected_outcome": "found",
        "expected_class": "factual",
        "expected_fund": "large-cap",
        "expected_url": "https://example.com/f",
        "expected_fact_text": "Expense ratio: 1.02%",
    }
    check = assess_case(
        case, "found", "factual", "small-cap", True,
        [_row("https://example.com/other", "Expense ratio: 1.02%", 0.3)],
    )
    assert not check.passed
    assert check.checks["fund"] is False
    assert check.checks["recall"] is False


def test_assess_not_found_requires_empty_evidence():
    case = {
        "id": "x",
        "expected_outcome": "not_found",
        "expected_class": "factual",
        "expected_fund": "elss",
        "expected_url": "https://example.com/e",
        "expected_fact_text": None,
    }
    ok = assess_case(case, "not_found", "factual", "elss", True, [])
    assert ok.passed
    leak = assess_case(case, "not_found", "factual", "elss", True, [_row("https://example.com/e", "x", 0.2)])
    assert not leak.passed
    assert "no_evidence" not in leak.checks or leak.checks["no_evidence"] is False


def test_assess_blocked_requires_no_embedding():
    case = {
        "id": "x",
        "expected_outcome": "blocked",
        "expected_class": "advice",
        "expected_fund": None,
        "expected_url": None,
        "expected_fact_text": None,
    }
    ok = assess_case(case, "blocked", "advice", None, False, [])
    assert ok.passed
    unsafe = assess_case(case, "blocked", "advice", None, True, [])
    assert not unsafe.passed
    assert unsafe.checks["safe"] is False
    wrong_class = assess_case(case, "blocked", "allocation", None, False, [])
    assert not wrong_class.passed


def test_assess_clarification_requires_no_fund():
    case = {
        "id": "x",
        "expected_outcome": "clarification",
        "expected_class": "factual",
        "expected_fund": None,
        "expected_url": None,
        "expected_fact_text": None,
    }
    ok = assess_case(case, "clarification", "factual", None, False, [])
    assert ok.checks["outcome"] is True
    assert ok.checks["not_embedded"] is True
    assert ok.passed


def test_summarize_reports_rates():
    checks = [
        assess_case(
            {"id": "a", "expected_outcome": "found", "expected_class": "factual",
             "expected_fund": "large-cap", "expected_url": "u", "expected_fact_text": "f"},
            "found", "factual", "large-cap", True, [_row("u", "f", 0.3)],
        ),
        assess_case(
            {"id": "b", "expected_outcome": "blocked", "expected_class": "advice",
             "expected_fund": None, "expected_url": None, "expected_fact_text": None},
            "blocked", "advice", None, False, [],
        ),
        assess_case(
            {"id": "c", "expected_outcome": "not_found", "expected_class": "factual",
             "expected_fund": "elss", "expected_url": "u2", "expected_fact_text": None},
            "not_found", "factual", "elss", True, [],
        ),
    ]
    summary = summarize(checks)
    assert summary["safety_gate"]["rate"] == 1.0
    assert summary["safety_pass"] is True
    assert summary["recall_at_4"]["rate"] == 1.0
    assert summary["recall_pass"] is True
    assert summary["not_found_at_retrieval"]["rate"] == 1.0
    assert summary["unsupported_at_retrieval"] == 0.0
    assert summary["outcome_rate"] >= 0.95


def test_summary_flags_recall_below_floor():
    checks = [
        assess_case(
            {"id": "a", "expected_outcome": "found", "expected_class": "factual",
             "expected_fund": "large-cap", "expected_url": "u",
             "expected_fact_text": f"x{i}"},  # fact missing from evidence -> recall fails
            "found", "factual", "large-cap", True, [_row("u", "no facts")],
        )
        for i in range(5)
    ]
    summary = summarize(checks)
    assert summary["recall_pass"] is False
    assert summary["recall_at_4"]["rate"] == 0.0


def test_calibrate_recommends_threshold_between_clusters():
    from dataclasses import replace

    found_checks = [
        replace(
            assess_case(
                {"id": f"f{i}", "expected_outcome": "found", "expected_class": "factual",
                 "expected_fund": "large-cap", "expected_url": "u", "expected_fact_text": "x"},
                "found", "factual", "large-cap", True, [_row("u", "x", d)],
            ),
            fact_distance=d, fact_rank=0,
        )
        for i, d in enumerate([0.2, 0.22, 0.25, 0.3])
    ]
    absent_checks = [
        assess_case(
            {"id": f"a{i}", "expected_outcome": "not_found", "expected_class": "factual",
             "expected_fund": "elss", "expected_url": "u2", "expected_fact_text": None},
            "not_found", "factual", "elss", True, [_row("u2", "y", d)],
        )
        for i, d in enumerate([0.7, 0.72, 0.75, 0.8])
    ]
    result = calibrate(found_checks + absent_checks)
    recommended = result["recommended_threshold"]
    # max fact distance = 0.30 -> one grid step up = 0.325, safely below the absent cluster.
    assert recommended == 0.325, recommended
    sweep = {r["threshold"]: r for r in result["sweep"]}
    assert sweep[0.3]["recall"] == 1.0
    assert sweep[0.3]["absent_rate"] == 0.0
    assert sweep[0.8]["absent_rate"] == 1.0


def test_calibrate_adds_margin_without_crossing_absent_cluster():
    from dataclasses import replace

    found = [
        replace(
            assess_case(
                {"id": f"f{i}", "expected_outcome": "found", "expected_class": "factual",
                 "expected_fund": "large-cap", "expected_url": "u", "expected_fact_text": "x"},
                "found", "factual", "large-cap", True, [_row("u", "x", d)],
            ),
            fact_distance=d, fact_rank=0,
        )
        for i, d in enumerate([0.3, 0.31])
    ]
    absent = [assess_case(
        {"id": "a0", "expected_outcome": "not_found", "expected_class": "factual",
         "expected_fund": "elss", "expected_url": "u2", "expected_fact_text": None},
        "not_found", "factual", "elss", True, [_row("u2", "y", 0.32)],
    )]
    result = calibrate(found + absent)
    # max fact 0.31 -> +step 0.335 would cross min absent 0.32, so it steps back to
    # 0.295, clamped up to the 0.3 floor.
    assert result["recommended_threshold"] == 0.3
    assert result["absent_best_distances"] == [0.32]
    assert result["eligible_fact_distances"] == [0.3, 0.31]


def test_threshold_override_from_file(tmp_path, monkeypatch):
    import code.config as config_module

    threshold_file = tmp_path / "threshold.json"
    threshold_file.write_text(json.dumps({"relevance_threshold": 0.42}), encoding="utf-8")
    monkeypatch.setattr(config_module, "THRESHOLD_FILE", threshold_file)
    assert config_module._relevance_threshold() == 0.42

    monkeypatch.setattr(config_module, "THRESHOLD_FILE", tmp_path / "missing.json")
    assert config_module._relevance_threshold() == config_module._DEFAULT_RELEVANCE_THRESHOLD


def test_retriever_raw_top_returns_pre_threshold_rows():
    from code.rag.retriever import Retriever

    class _FakeStore:
        def query(self, vector, n_results, where):
            assert "large-cap" in where["canonical_url"]
            return [
                {"chunk_id": "c1", "chunk_text": "Expense ratio: 1.02%", "metadata": {
                    "canonical_url": where["canonical_url"], "fund_name": "F",
                    "fund_category": "large-cap", "section_heading": "Overview",
                    "ingested_at": "2026-08-30T00:00:00"}, "distance": 0.9},
            ]

    r = Retriever(store=_FakeStore(), embed_fn=lambda q: [0.0] * 384)
    fund, rows = r.raw_top("what is the expense ratio of the large cap fund?")
    assert fund is not None
    assert len(rows) == 1 and rows[0].distance == 0.9


def test_run_evaluation_rejects_malformed_question_none():
    cases = load_cases()
    for case in cases:
        assert case["question"].strip(), "empty question"


def test_is_refusal():
    assert is_refusal(None) is False
    assert is_refusal("The expense ratio is 0.55%.") is False
    assert is_refusal("I could not generate a verified answer.") is True
    assert is_refusal("The approved pages do not disclose the CEOs of the companies.") is True
    assert is_refusal("The approved pages do not list the custodian.") is True
    assert is_refusal("No information about the custodian was found.") is True


def test_retrieval_cases_json_is_versioned():
    payload = json.loads(RETRIEVAL_CASES_FILE.read_text(encoding="utf-8"))
    assert int(payload["version"]) >= 1