"""Loader and cleaner tests for the five approved Groww URLs."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from code.config import APPROVED_SOURCES, LOAD_STATUS_SUCCESSFUL, LOAD_STATUS_UNAVAILABLE, LOAD_STATUS_VALIDATION_FAILED
from code.ingestion.cleaner import clean_html_to_source_text, flatten_fund_facts
from code.ingestion.loader import ALLOWED_FETCH_URLS, FetchResult, fetch_public_html
from code.ingestion.pipeline import REQUIRED_DOCUMENT_FIELDS, SourceDocument, load_approved_corpus, load_one_source

FIXTURE_HTML = (Path(__file__).parent / "fixtures" / "sample_fund_page.html").read_text(encoding="utf-8")
LARGE_CAP = APPROVED_SOURCES[0]


@pytest.fixture
def data_dirs(tmp_path, monkeypatch):
    html_dir = tmp_path / "raw" / "html"
    docs_dir = tmp_path / "raw" / "documents"
    html_dir.mkdir(parents=True)
    docs_dir.mkdir(parents=True)
    monkeypatch.setattr("code.ingestion.loader.RAW_HTML_DIR", html_dir)
    monkeypatch.setattr("code.ingestion.pipeline.RAW_DOCUMENTS_DIR", docs_dir)
    return html_dir, docs_dir


def test_allowlist_contains_exactly_five_sources():
    assert len(APPROVED_SOURCES) == 5
    assert len(ALLOWED_FETCH_URLS) == 5


def test_fetch_rejects_user_supplied_url():
    with pytest.raises(ValueError, match="allowlisted"):
        fetch_public_html("https://example.com/not-approved")


def test_fetch_rejects_unapproved_groww_page():
    with pytest.raises(ValueError, match="allowlisted"):
        fetch_public_html("https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth")


def test_redirect_to_other_host_is_validation_failed(data_dirs):
    html_dir, docs_dir = data_dirs
    with patch("code.ingestion.loader.fetch_public_html", return_value=("https://evil.example/phish", FIXTURE_HTML)):
        document = load_one_source(LARGE_CAP)
    assert document.load_status == LOAD_STATUS_VALIDATION_FAILED
    assert document.source_text == ""
    assert not (html_dir / "hdfc-large-cap-fund-direct-growth.html").exists()


def test_failed_fetch_does_not_overwrite_last_valid_document(data_dirs):
    _, docs_dir = data_dirs
    previous = SourceDocument(
        canonical_url=LARGE_CAP["canonical_url"],
        fund_name=LARGE_CAP["fund_name"],
        fund_category=LARGE_CAP["fund_category"],
        source_text="Expense ratio: 1.02%",
        ingested_at="2026-01-01T00:00:00Z",
        content_hash="abc",
        load_status=LOAD_STATUS_SUCCESSFUL,
    )
    (docs_dir / "hdfc-large-cap-fund-direct-growth.json").write_text(
        json.dumps(previous.as_dict()), encoding="utf-8"
    )

    failed = FetchResult(
        canonical_url=LARGE_CAP["canonical_url"],
        fund_name=LARGE_CAP["fund_name"],
        fund_category=LARGE_CAP["fund_category"],
        final_url=None,
        html=None,
        ok=False,
        load_status=LOAD_STATUS_UNAVAILABLE,
        error="timeout",
    )
    document = load_one_source(LARGE_CAP, fetch_result=failed)
    assert document.source_text == "Expense ratio: 1.02%"
    assert document.ingested_at == "2026-01-01T00:00:00Z"
    saved = json.loads((docs_dir / "hdfc-large-cap-fund-direct-growth.json").read_text(encoding="utf-8"))
    assert saved["source_text"] == "Expense ratio: 1.02%"
    attempt = json.loads((docs_dir / "hdfc-large-cap-fund-direct-growth.attempt.json").read_text(encoding="utf-8"))
    assert attempt["load_status"] == LOAD_STATUS_UNAVAILABLE
    assert attempt["retained_previous_successful"] is True


def test_first_failure_writes_unavailable_status_without_invented_text(data_dirs):
    document = load_one_source(
        LARGE_CAP,
        fetch_result=FetchResult(
            canonical_url=LARGE_CAP["canonical_url"],
            fund_name=LARGE_CAP["fund_name"],
            fund_category=LARGE_CAP["fund_category"],
            final_url=None,
            html=None,
            ok=False,
            load_status=LOAD_STATUS_UNAVAILABLE,
            error="404",
        ),
    )
    assert document.load_status == LOAD_STATUS_UNAVAILABLE
    assert document.source_text == ""
    assert all(getattr(document, field) is not None for field in REQUIRED_DOCUMENT_FIELDS)


def test_successful_load_writes_html_and_complete_metadata(data_dirs):
    html_dir, docs_dir = data_dirs
    result = FetchResult(
        canonical_url=LARGE_CAP["canonical_url"],
        fund_name=LARGE_CAP["fund_name"],
        fund_category=LARGE_CAP["fund_category"],
        final_url=LARGE_CAP["canonical_url"],
        html=FIXTURE_HTML,
        ok=True,
        load_status=LOAD_STATUS_SUCCESSFUL,
        error=None,
    )
    document = load_one_source(LARGE_CAP, fetch_result=result)
    assert document.load_status == LOAD_STATUS_SUCCESSFUL
    assert document.canonical_url == LARGE_CAP["canonical_url"]
    assert document.fund_name == LARGE_CAP["fund_name"]
    assert document.fund_category == LARGE_CAP["fund_category"]
    assert document.ingested_at
    assert len(document.content_hash) == 64
    assert "Expense ratio: 1.02%" in document.source_text
    assert "Minimum SIP: 100" in document.source_text
    assert (html_dir / "hdfc-large-cap-fund-direct-growth.html").exists()
    saved = json.loads((docs_dir / "hdfc-large-cap-fund-direct-growth.json").read_text(encoding="utf-8"))
    assert set(REQUIRED_DOCUMENT_FIELDS) <= set(saved)


def test_cleaner_keeps_fact_value_pairs_and_drops_chrome():
    text = clean_html_to_source_text(FIXTURE_HTML, LARGE_CAP["fund_name"], LARGE_CAP["fund_category"])
    assert "Expense ratio: 1.02%" in text
    assert "Exit load: Exit load of 1% if redeemed within 1 year" in text
    assert "window.tracking" not in text
    assert "Invest in stocks" not in text
    assert "Download the app" not in text


def test_flatten_does_not_invent_empty_lock_in():
    text = flatten_fund_facts(
        {"scheme_name": "Test", "lock_in": {"years": None, "months": None, "days": None}},
        "Test",
        "large-cap",
    )
    assert "Lock-in:" not in text


def test_pipeline_attempts_exactly_the_five_allowlisted_sources(data_dirs):
    def fake_fetch(source):
        return FetchResult(
            canonical_url=source["canonical_url"],
            fund_name=source["fund_name"],
            fund_category=source["fund_category"],
            final_url=source["canonical_url"],
            html=FIXTURE_HTML,
            ok=True,
            load_status=LOAD_STATUS_SUCCESSFUL,
            error=None,
        )

    with patch("code.ingestion.pipeline.fetch_approved_source", side_effect=fake_fetch) as mocked:
        documents = load_approved_corpus()

    assert mocked.call_count == 5
    assert [doc.canonical_url for doc in documents] == [s["canonical_url"] for s in APPROVED_SOURCES]
    assert all(doc.load_status == LOAD_STATUS_SUCCESSFUL for doc in documents)
    _, docs_dir = data_dirs
    summary = json.loads((docs_dir / "_run_summary.json").read_text(encoding="utf-8"))
    assert summary["attempt_count"] == 5


def test_fetch_public_html_uses_allowlisted_url_only():
    response = MagicMock()
    response.url = LARGE_CAP["canonical_url"]
    response.text = FIXTURE_HTML
    response.raise_for_status = MagicMock()
    with patch("code.ingestion.loader.requests.get", return_value=response) as get:
        final_url, html = fetch_public_html(LARGE_CAP["canonical_url"])
    assert final_url == LARGE_CAP["canonical_url"]
    assert html == FIXTURE_HTML
    called_url = get.call_args.args[0]
    assert called_url == LARGE_CAP["canonical_url"]


def test_http_error_is_unavailable(data_dirs):
    with patch(
        "code.ingestion.loader.requests.get",
        side_effect=requests.ConnectionError("blocked"),
    ):
        from code.ingestion.loader import fetch_approved_source

        result = fetch_approved_source(LARGE_CAP)
    assert result.ok is False
    assert result.load_status == LOAD_STATUS_UNAVAILABLE
    assert result.html is None
