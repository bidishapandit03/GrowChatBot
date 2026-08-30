"""Extract visible fund content and write cleaned source documents under data/raw/documents/."""

from __future__ import annotations

import json
import re
from html import unescape

from bs4 import BeautifulSoup

DROP_TAGS = ("script", "style", "noscript", "svg", "iframe", "img", "button", "nav", "footer", "header")
NAV_CLASS_HINTS = (
    "header2025",
    "loggedout_nav",
    "dropdownui",
    "cookie",
    "footer",
    "bottombar",
)
NOISE_LINE_PREFIXES = (
    "invest in stocks",
    "trade in futures",
    "download the app",
    "© 2016",
    "products",
    "share market",
    "see all",
    "view details",
    "compare similar",
    "return calculator",
)


def _strip_tags(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", value)
    text = unescape(text)
    return normalize_whitespace(text)


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    deduped: list[str] = []
    for line in lines:
        if not line:
            if deduped and deduped[-1] != "":
                deduped.append("")
            continue
        if deduped and deduped[-1] == line:
            continue
        deduped.append(line)
    return "\n".join(deduped).strip()


def extract_mf_server_side_data(html: str) -> dict | None:
    soup = BeautifulSoup(html, "html.parser")
    tag = soup.find("script", id="__NEXT_DATA__")
    if not tag or not tag.string:
        return None
    try:
        payload = json.loads(tag.string)
    except json.JSONDecodeError:
        return None
    data = payload.get("props", {}).get("pageProps", {}).get("mfServerSideData")
    return data if isinstance(data, dict) else None


def _format_money_cr(value: object) -> str | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return f"₹{number:,.2f} Cr"


def _format_lock_in(lock_in: object) -> str | None:
    if not isinstance(lock_in, dict):
        return None
    parts = []
    for unit in ("years", "months", "days"):
        amount = lock_in.get(unit)
        if amount not in (None, "", 0):
            parts.append(f"{amount} {unit}")
    if not parts:
        return None
    return ", ".join(parts)


def flatten_fund_facts(data: dict, fund_name: str, fund_category: str) -> str:
    """Turn public page JSON into labeled fact-value lines (no linked pages are fetched)."""
    lines: list[str] = [f"# {fund_name}", "", f"Fund category: {fund_category}"]

    def add(label: str, value: object) -> None:
        if value in (None, "", [], {}):
            return
        if isinstance(value, str):
            cleaned = _strip_tags(value)
            if not cleaned:
                return
            lines.append(f"{label}: {cleaned}")
            return
        if isinstance(value, bool):
            lines.append(f"{label}: {'yes' if value else 'no'}")
            return
        lines.append(f"{label}: {value}")

    add("Scheme name", data.get("scheme_name") or fund_name)
    add("Fund house", data.get("fund_house"))
    add("Category", data.get("category"))
    add("Sub-category", data.get("sub_category"))
    add("Scheme type", data.get("scheme_type"))
    add("Plan type", data.get("plan_type"))
    add("Investment objective", data.get("description"))
    add("Riskometer", data.get("nfo_risk"))
    add("Benchmark", data.get("benchmark_name") or data.get("benchmark"))
    add("Expense ratio", f"{data['expense_ratio']}%" if data.get("expense_ratio") not in (None, "") else None)
    add("Minimum SIP", data.get("min_sip_investment"))
    add("Minimum lumpsum / 1st investment", data.get("min_investment_amount"))
    add("Minimum additional investment", data.get("mini_additional_investment"))
    add("SIP allowed", data.get("sip_allowed"))
    add("Lumpsum allowed", data.get("lumpsum_allowed"))
    add("Exit load", data.get("exit_load"))
    add("Lock-in", _format_lock_in(data.get("lock_in")))
    add("Stamp duty", data.get("stamp_duty"))
    add("AUM", _format_money_cr(data.get("aum")))
    add("NAV", data.get("nav"))
    add("NAV date", data.get("nav_date"))
    add("Launch date", data.get("launch_date"))
    add("Allotment date", data.get("allotment_date"))
    add("Registrar", data.get("registrar_agent"))

    category_info = data.get("category_info") if isinstance(data.get("category_info"), dict) else {}
    add("Tax implication", category_info.get("tax_impact"))
    add("Category description", category_info.get("description") or category_info.get("category_helper_text"))

    historic = data.get("historic_exit_loads")
    if isinstance(historic, list) and historic:
        lines.append("")
        lines.append("## Historic exit load")
        for item in historic:
            if not isinstance(item, dict):
                continue
            note = (item.get("note") or "").strip()
            as_on = (item.get("as_on_date") or "")[:10]
            if note:
                lines.append(f"Exit load as of {as_on}: {normalize_whitespace(note)}" if as_on else f"Exit load: {note}")

    managers = data.get("fund_manager_details")
    if isinstance(managers, list) and managers:
        lines.append("")
        lines.append("## Fund management")
        for manager in managers:
            if not isinstance(manager, dict):
                continue
            name = manager.get("person_name")
            if not name:
                continue
            add("Fund manager", name)
            add("Fund manager education", manager.get("education"))
            add("Fund manager from", (manager.get("date_from") or "")[:10] or None)

    holdings = data.get("holdings")
    if isinstance(holdings, list) and holdings:
        lines.append("")
        lines.append("## Holdings")
        for holding in holdings[:15]:
            if not isinstance(holding, dict):
                continue
            name = holding.get("stock_name") or holding.get("name") or holding.get("company_name")
            assets = holding.get("corpus_per") or holding.get("assets") or holding.get("percentage")
            sector = holding.get("sector_name") or holding.get("sector")
            if not name:
                continue
            detail = name
            if sector:
                detail += f" | Sector: {sector}"
            if assets not in (None, ""):
                detail += f" | Assets: {assets}%"
            lines.append(f"Holding: {detail}")

    return normalize_whitespace("\n".join(lines))


def extract_visible_page_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(DROP_TAGS):
        tag.decompose()
    for element in list(soup.find_all(True)):
        attrs = element.attrs or {}
        class_value = attrs.get("class") or []
        if isinstance(class_value, str):
            class_value = [class_value]
        class_text = " ".join(class_value).lower()
        element_id = str(attrs.get("id") or "").lower()
        if any(hint in class_text or hint in element_id for hint in NAV_CLASS_HINTS):
            element.decompose()

    root = soup.select_one(".pw14ContentWrapper") or soup.select_one(".layout-main") or soup.find("body") or soup
    raw = root.get_text("\n", strip=True)
    kept: list[str] = []
    for line in raw.split("\n"):
        compact = normalize_whitespace(line)
        if not compact or len(compact) <= 1:
            continue
        if compact.lower().startswith(NOISE_LINE_PREFIXES):
            continue
        kept.append(compact)
    return normalize_whitespace("\n".join(kept))


def clean_html_to_source_text(html: str, fund_name: str, fund_category: str) -> str:
    data = extract_mf_server_side_data(html)
    parts: list[str] = []
    if data:
        parts.append(flatten_fund_facts(data, fund_name, fund_category))
    visible = extract_visible_page_text(html)
    if visible:
        parts.append("")
        parts.append("## Visible page text")
        parts.append(visible)
    return normalize_whitespace("\n".join(parts))
