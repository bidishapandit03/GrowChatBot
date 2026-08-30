"""Map fund names and the legacy HDFC Equity Fund alias to one approved source."""

from __future__ import annotations

import re
from dataclasses import dataclass

from code.config import APPROVED_SOURCES

SOURCES_BY_CATEGORY = {source["fund_category"]: source for source in APPROVED_SOURCES}

# Order matters: more specific aliases must come before general ones for the *same* fund,
# and phrases that only fit one fund do not collide across funds.
ALIASES = [
    ("large-cap", ("hdfc large cap fund", "hdfc large cap", "large cap fund")),
    ("flexi-cap", ("hdfc flexi cap fund", "hdfc flexi cap", "flexi cap fund", "hdfc equity fund", "hdfc equity")),
    ("elss", ("hdfc elss", "hdfc tax saver", "elss tax saver", "tax saver fund")),
    ("small-cap", ("hdfc small cap fund", "hdfc small cap", "small cap fund")),
    ("hybrid", ("hdfc balanced advantage fund", "hdfc balanced advantage", "balanced advantage fund")),
]


@dataclass(frozen=True)
class ResolvedFund:
    canonical_url: str
    fund_name: str
    fund_category: str


def _normalize(question: str) -> str:
    lowered = question.lower()
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def matched_categories(question: str) -> set[str]:
    """Return every approved fund category that the question names (via name or alias)."""
    text = _normalize(question)
    matched: set[str] = set()
    for category, phrases in ALIASES:
        if any(phrase in text for phrase in phrases):
            matched.add(category)
    return matched


def resolve_fund(question: str) -> ResolvedFund | None:
    """Resolve to exactly one approved fund, or None when ambiguous or absent."""
    matched = matched_categories(question)
    if len(matched) != 1:
        return None
    category = next(iter(matched))
    source = SOURCES_BY_CATEGORY[category]
    return ResolvedFund(
        canonical_url=source["canonical_url"],
        fund_name=source["fund_name"],
        fund_category=source["fund_category"],
    )


def clarification_categories() -> list[str]:
    return ["large-cap", "flexi-cap", "elss", "small-cap", "hybrid"]