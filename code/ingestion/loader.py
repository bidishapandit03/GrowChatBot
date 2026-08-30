"""Fetch public HTML for allowlisted Groww URLs and write files under data/raw/html/."""

from __future__ import annotations

from dataclasses import dataclass

import requests

from code.config import (
    APPROVED_SOURCES,
    HTTP_MAX_RETRIES,
    HTTP_TIMEOUT_SECONDS,
    HTTP_USER_AGENT,
    LOAD_STATUS_SUCCESSFUL,
    LOAD_STATUS_UNAVAILABLE,
    LOAD_STATUS_VALIDATION_FAILED,
    RAW_HTML_DIR,
    is_corresponding_approved_url,
    source_slug,
)

ALLOWED_FETCH_URLS = {source["canonical_url"] for source in APPROVED_SOURCES}


@dataclass(frozen=True)
class FetchResult:
    canonical_url: str
    fund_name: str
    fund_category: str
    final_url: str | None
    html: str | None
    ok: bool
    load_status: str
    error: str | None


def fetch_public_html(canonical_url: str) -> tuple[str, str]:
    """GET an allowlisted public page. Raises ValueError if the URL is not on the allowlist."""
    if canonical_url not in ALLOWED_FETCH_URLS:
        raise ValueError("Only allowlisted corpus URLs may be fetched.")

    last_error: Exception | None = None
    for attempt in range(HTTP_MAX_RETRIES + 1):
        try:
            response = requests.get(
                canonical_url,
                headers={"User-Agent": HTTP_USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
                timeout=HTTP_TIMEOUT_SECONDS,
                allow_redirects=True,
            )
            response.raise_for_status()
            return response.url, response.text
        except requests.RequestException as exc:
            last_error = exc
            if attempt >= HTTP_MAX_RETRIES:
                break
    raise ConnectionError(f"Failed to fetch {canonical_url}: {last_error}") from last_error


def fetch_approved_source(source: dict) -> FetchResult:
    canonical_url = source["canonical_url"]
    fund_name = source["fund_name"]
    fund_category = source["fund_category"]

    if canonical_url not in ALLOWED_FETCH_URLS:
        return FetchResult(
            canonical_url=canonical_url,
            fund_name=fund_name,
            fund_category=fund_category,
            final_url=None,
            html=None,
            ok=False,
            load_status="validation_failed",
            error="Source is not on the fixed allowlist.",
        )

    try:
        final_url, html = fetch_public_html(canonical_url)
    except ConnectionError as exc:
        return FetchResult(
            canonical_url=canonical_url,
            fund_name=fund_name,
            fund_category=fund_category,
            final_url=None,
            html=None,
            ok=False,
            load_status="unavailable",
            error=str(exc),
        )
    except ValueError as exc:
        return FetchResult(
            canonical_url=canonical_url,
            fund_name=fund_name,
            fund_category=fund_category,
            final_url=None,
            html=None,
            ok=False,
            load_status="validation_failed",
            error=str(exc),
        )

    if not is_corresponding_approved_url(final_url, canonical_url):
        return FetchResult(
            canonical_url=canonical_url,
            fund_name=fund_name,
            fund_category=fund_category,
            final_url=final_url,
            html=None,
            ok=False,
            load_status="validation_failed",
            error=f"Redirect left the approved page: {final_url}",
        )

    if not html or not html.strip():
        return FetchResult(
            canonical_url=canonical_url,
            fund_name=fund_name,
            fund_category=fund_category,
            final_url=final_url,
            html=None,
            ok=False,
            load_status="unavailable",
            error="Empty HTML body.",
        )

    return FetchResult(
        canonical_url=canonical_url,
        fund_name=fund_name,
        fund_category=fund_category,
        final_url=final_url,
        html=html,
        ok=True,
            load_status=LOAD_STATUS_SUCCESSFUL,
        error=None,
    )


def write_raw_html(canonical_url: str, html: str) -> None:
    RAW_HTML_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_HTML_DIR / f"{source_slug(canonical_url)}.html"
    path.write_text(html, encoding="utf-8")
