"""Streamlit UI for the HDFC mutual fund facts assistant.

Runs the full retrieval pipeline (``code.rag.pipeline.answer``) behind a small
Groww-inspired chat interface: welcome line, three example buttons, chat input,
citation, freshness label, and safety notices.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

try:
    from code.config import APPROVED_SOURCES
    from code.rag.pipeline import answer
except ModuleNotFoundError:
    # ``streamlit run`` (unlike ``python -m``) can resolve the ``code`` package to
    # the app's own folder instead of the project root; re-insert the root and retry.
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from code.config import APPROVED_SOURCES
    from code.rag.pipeline import answer

PAGE_TITLE = "HDFC Mutual Fund Facts Assistant"
WELCOME = "Ask about any of the five HDFC funds (Large Cap, Flexi Cap, ELSS Tax Saver, Small Cap, Balanced Advantage)."
NOTICE = "Facts-only. No investment advice."
DISCLAIMER = "Independent prototype using public Groww pages."
PII_WARNING = "Do not enter PAN, Aadhaar, account numbers, OTPs, email addresses, or phone numbers."

EXAMPLE_QUESTIONS = [
    "What is the expense ratio of HDFC Large Cap Fund Direct Growth?",
    "What is the lock-in period for the HDFC ELSS fund?",
    "What is the minimum SIP for HDFC Small Cap Fund Direct Growth?",
]

_SOURCE_LINE = re.compile(r"^Source: (https?://\S+)$", re.MULTILINE)

_GREEN = "#00d09c"
_APPROVED_URLS = {source["canonical_url"] for source in APPROVED_SOURCES}


def _link_sources(text: str) -> str:
    """Convert approved ``Source: <url>`` lines into clickable markdown links.

    URLs not on the allowlist stay as plain text so unapproved citations are never
    rendered as navigable links.
    """

    def _replace(match: re.Match) -> str:
        url = match.group(1)
        if url not in _APPROVED_URLS:
            return match.group(0)
        return f"Source: [{url}]({url})"

    return _SOURCE_LINE.sub(_replace, text)


def _evidence_text(result) -> str:
    lines = []
    for row in result.evidence:
        lines.append(
            f"[{row.fund_category}/{row.section_heading}] "
            f"(distance {row.distance:.3f})\n{row.canonical_url}\n{row.chunk_text}"
        )
    return "\n\n---\n\n".join(lines)


def _render_message(entry: dict) -> None:
    if entry["role"] == "user":
        with st.chat_message("user"):
            st.markdown(entry["text"])
        return

    with st.chat_message("assistant"):
        st.markdown(_link_sources(entry["answer"]))
        if entry.get("evidence"):
            with st.expander("Evidence (retrieved chunks)"):
                st.markdown(_evidence_text(entry["result"]))
        if entry.get("conflicts"):
            st.error("The approved pages contain conflicting values; no single figure was chosen.")


_MESSAGE_OPERATIONAL_ERROR = (
    "Sorry, something went wrong while answering. Please try again in a moment."
)


def _process(question: str) -> dict:
    try:
        result = answer(question)
    except Exception:
        logging.getLogger("ui").exception("pipeline failed for question")
        return {"role": "assistant", "question": question, "answer": _MESSAGE_OPERATIONAL_ERROR,
                "grounded": False, "evidence": False, "conflicts": False, "result": None}
    return {
        "role": "assistant",
        "question": question,
        "answer": result.answer or "",
        "grounded": result.grounded,
        "evidence": bool(result.retrieval.evidence),
        "conflicts": bool(result.retrieval.conflicts),
        "result": result.retrieval,
    }


def _init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending" not in st.session_state:
        st.session_state.pending = None
    if "busy" not in st.session_state:
        st.session_state.busy = False


def _inject_styles() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{ background-color: #000000; }}
        .stMarkdown p, .stMarkdown li {{ color: #e5e7eb; }}
        .stCaption, .stMarkdown caption, [data-testid="stCaptionContainer"] {{ color: #9ca3af; }}
        .stTitle, h1, h2, h3 {{ color: #ffffff; }}
        .stButton > button {{
            background-color: {_GREEN}; color: #ffffff; border: none;
            border-radius: 8px; font-weight: 600;
        }}
        .stButton > button:hover {{ background-color: #00b98c; color: #ffffff; }}
        div[data-testid="stChatInput"] button {{
            background-color: {_GREEN}; color: #ffffff; border: none;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title=PAGE_TITLE,
        layout="centered",
    )
    _inject_styles()
    _init_state()

    st.title(PAGE_TITLE)
    st.caption(WELCOME)
    st.info(NOTICE)
    st.warning(PII_WARNING)
    st.caption(DISCLAIMER)

    buttons = st.columns(len(EXAMPLE_QUESTIONS))
    for column, example in zip(buttons, EXAMPLE_QUESTIONS):
        if column.button(example, width="stretch"):
            st.session_state.pending = example

    for entry in st.session_state.messages:
        _render_message(entry)

    prompt = st.chat_input("Ask about one of the five HDFC funds\u2026", disabled=st.session_state.busy)

    if prompt:
        st.session_state.pending = prompt

    if st.session_state.pending and not st.session_state.busy:
        question = st.session_state.pending
        st.session_state.pending = None
        st.session_state.busy = True

        st.session_state.messages.append({"role": "user", "text": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching approved Groww pages\u2026"):
                entry = _process(question)

        st.session_state.messages.append(entry)
        st.session_state.busy = False
        _render_message(entry)


if __name__ == "__main__":
    main()