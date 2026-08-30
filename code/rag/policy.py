"""Deterministic PII, advice, performance, and corpus-scope gates for the facts-only assistant."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class QueryClass(str, Enum):
    FACTUAL = "factual"
    PII = "pii"
    ADVICE = "advice"
    PERFORMANCE = "performance"
    ALLOCATION = "allocation"
    ACCOUNT_SUPPORT = "account_support"
    OUTSIDE_SCOPE = "outside_scope"
    INJECTION = "injection"


PII_PATTERNS = {
    "pan": re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", re.IGNORECASE),
    "aadhaar": re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b"),
    "otp": re.compile(r"\b(?:otp|one\s?time\s?password)[^\d]{0,12}\d{4,8}\b", re.IGNORECASE),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"),
    "account": re.compile(r"\b(?:account|a\/c|acc\.?|folio)[\s#:.-]{0,4}\d{6,}\b", re.IGNORECASE),
    "phone": re.compile(r"\b(?:\+?91[ -]?)?[6-9]\d{9}\b"),
}


@dataclass(frozen=True)
class PIICheck:
    flagged: bool
    categories: tuple[str, ...]
    redacted: str

    @classmethod
    def clean(cls, text: str) -> "PIICheck":
        redacted = text
        found: list[str] = []
        for name, pattern in PII_PATTERNS.items():
            if pattern.search(redacted):
                found.append(name)
                redacted = pattern.sub("[redacted]", redacted)
        return cls(flagged=bool(found), categories=tuple(sorted(set(found))), redacted=redacted)


ADVICE_PHRASES = (
    "should i buy",
    "should i sell",
    "should i invest",
    "should i continue",
    "should i hold",
    "should i switch",
    "is it a good",
    "is this fund good",
    "good investment",
    "worth investing",
    "recommend",
    "suggest",
    "suitable",
    "suit me",
    "best for me",
    "for my portfolio",
    "in my portfolio",
    "good for me",
    "my goal",
    "my situation",
    "my age",
    "my risk",
    "retirement",
    "advice",
    "tips",
    "buy the fund",
    "sell the fund",
    "switch to",
    "which fund should",
    "is it better",
    "is better",
    "better than",
    "which is best",
    "best among",
    "rank",
    "invest more",
)

PERFORMANCE_PHRASES = (
    "returns",
    "return",
    "cagr",
    "xirr",
    "annualised",
    "annualized",
    "performance",
    "predict",
    "forecast",
    "will perform",
    "future",
    "outperform",
    "outperformed",
    "compound",
    "will be worth",
    "how much will",
    "grow my",
    "grow to",
    "yield",
    "3 year return",
    "3-year return",
    "1 year return",
    "1-year return",
    "5 year return",
    "5-year return",
    "10 year return",
    "10-year return",
    "last year",
    "rate of return",
    "past return",
    "next year",
)

ALLOCATION_PHRASES = (
    "allocate",
    "allocation",
    "diversif",
    "combination",
    "how much should i invest in each",
    "asset mix",
    "which of these",
    "split between",
    "weightage",
)

ACCOUNT_PHRASES = (
    "my account",
    "my portfolio",
    "login",
    "log in",
    "kyc",
    "statement",
    "folio no",
    "transaction",
    "password",
    "redemption status",
    "purchase status",
    "unlock",
    "verify mobile",
)

OUTSIDE_SCOPE_PATTERNS = (
    re.compile(r"\bsbi\b", re.IGNORECASE),
    re.compile(r"\baxis\b", re.IGNORECASE),
    re.compile(r"\bicici\b", re.IGNORECASE),
    re.compile(r"\bnippon\b", re.IGNORECASE),
    re.compile(r"\bkotak\b", re.IGNORECASE),
    re.compile(r"\bfranklin\b", re.IGNORECASE),
    re.compile(r"\bmirae\b", re.IGNORECASE),
    re.compile(r"\buti\b", re.IGNORECASE),
    re.compile(r"\bparag parikh\b", re.IGNORECASE),
    re.compile(r"\bquant\b", re.IGNORECASE),
    re.compile(r"\btata\b", re.IGNORECASE),
    re.compile(r"\bcanara\b", re.IGNORECASE),
    re.compile(r"\bidfc\b", re.IGNORECASE),
    re.compile(r"\bdsp\b", re.IGNORECASE),
    re.compile(r"\bsundaram\b", re.IGNORECASE),
    re.compile(r"\bmotilal\b", re.IGNORECASE),
    re.compile(r"\bindex fund\b", re.IGNORECASE),
    re.compile(r"\bmid cap fund\b", re.IGNORECASE),
    re.compile(r"\bmid-cap fund\b", re.IGNORECASE),
    re.compile(r"\bhd\w*\.?\s*mid\s*-?\s*cap\b", re.IGNORECASE),
    re.compile(r"\bsensex fund\b", re.IGNORECASE),
    re.compile(r"\bgold etf\b", re.IGNORECASE),
    re.compile(r"\barbitrage\b", re.IGNORECASE),
)

INJECTION_PATTERNS = (
    re.compile(r"\bignore\b(?: your| the| previous| all)?\s*(?:rule|instruction|prompt|system|above)", re.IGNORECASE),
    re.compile(r"\bdisregard\b", re.IGNORECASE),
    re.compile(r"\byou are now\b", re.IGNORECASE),
    re.compile(r"\bact as\b", re.IGNORECASE),
    re.compile(r"\bsystem prompt\b", re.IGNORECASE),
    re.compile(r"\bjailbreak\b", re.IGNORECASE),
    re.compile(r"\bsearch the web\b", re.IGNORECASE),
    re.compile(r"\bdo a web search\b", re.IGNORECASE),
    re.compile(r"\bforget your rules\b", re.IGNORECASE),
    re.compile(r"\bas an ai\b", re.IGNORECASE),
)


@dataclass(frozen=True)
class PolicyDecision:
    query_class: QueryClass
    blocked: bool
    message: str | None = None


def classify_intent(question: str) -> PolicyDecision:
    """Classify advice, performance, allocation, account, injection, and scope intents. PII is handled separately."""
    lowered = question.lower()

    for pattern in INJECTION_PATTERNS:
        if pattern.search(lowered):
            return PolicyDecision(QueryClass.INJECTION, True, REFUSAL_MESSAGES[QueryClass.INJECTION])

    for pattern in OUTSIDE_SCOPE_PATTERNS:
        if pattern.search(lowered):
            return PolicyDecision(QueryClass.OUTSIDE_SCOPE, True, REFUSAL_MESSAGES[QueryClass.OUTSIDE_SCOPE])

    for phrase in PERFORMANCE_PHRASES:
        if phrase in lowered:
            return PolicyDecision(QueryClass.PERFORMANCE, True, REFUSAL_MESSAGES[QueryClass.PERFORMANCE])

    for phrase in ALLOCATION_PHRASES:
        if phrase in lowered:
            return PolicyDecision(QueryClass.ALLOCATION, True, REFUSAL_MESSAGES[QueryClass.ALLOCATION])

    for phrase in ADVICE_PHRASES:
        if phrase in lowered:
            return PolicyDecision(QueryClass.ADVICE, True, REFUSAL_MESSAGES[QueryClass.ADVICE])

    for phrase in ACCOUNT_PHRASES:
        if phrase in lowered:
            return PolicyDecision(QueryClass.ACCOUNT_SUPPORT, True, REFUSAL_MESSAGES[QueryClass.ACCOUNT_SUPPORT])

    return PolicyDecision(QueryClass.FACTUAL, False)


REFUSAL_MESSAGES = {
    QueryClass.PII: (
        "Please do not share PAN, Aadhaar, account numbers, OTPs, email addresses, "
        "or phone numbers here. No sensitive value was processed or stored."
    ),
    QueryClass.ADVICE: (
        "This assistant provides facts only and cannot give investment, buy, sell, hold, "
        "suitability, or allocation advice. I can share factual attributes such as expense "
        "ratio, exit load, or minimum SIP from the approved pages."
    ),
    QueryClass.PERFORMANCE: (
        "Return, yield, and performance analysis is outside the scope of this prototype. "
        "I can share static facts such as expense ratio, exit load, benchmark, or minimum SIP "
        "from the five approved pages."
    ),
    QueryClass.ALLOCATION: (
        "I cannot recommend portfolio allocation or compare funds. I can provide individual "
        "factual attributes for any single one of the five approved funds."
    ),
    QueryClass.ACCOUNT_SUPPORT: (
        "I cannot help with account or transaction issues. Please contact Groww's official "
        "support through their own channels; I do not have that contact information."
    ),
    QueryClass.OUTSIDE_SCOPE: (
        "This prototype covers only five HDFC funds on Groww: Large Cap, Flexi Cap, "
        "ELSS Tax Saver, Small Cap, and Balanced Advantage (all Direct Growth). "
        "It cannot answer about other funds, AMCs, or markets."
    ),
    QueryClass.INJECTION: (
        "I only answer from the five approved Groww pages and cannot be instructed to "
        "change, ignore, or bypass that behavior."
    ),
}