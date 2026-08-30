"""Run Phase 5 retrieval and generation from the command line:

    python -m code.rag "what is the expense ratio of hdfc large cap fund?"
    python -m code.rag "should i buy the small cap fund?" --json
    python -m code.rag "what is the expense ratio of hdfc large cap fund?" --answer
"""

from __future__ import annotations

import argparse
import json

from code.rag.pipeline import answer
from code.rag.retriever import Retriever

FUND_HINT = {
    "large-cap": "large-cap",
    "flexi-cap": "flexi-cap",
    "elss": "elss",
    "small-cap": "small-cap",
    "hybrid": "hybrid",
}


def _category_label(category: str | None) -> str:
    return category if category else "-"


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m code.rag", description="HDFC facts retrieval")
    parser.add_argument("questions", nargs="+", help="One or more questions to run through retrieval")
    parser.add_argument("--json", action="store_true", help="Print full JSON results")
    parser.add_argument("--answer", action="store_true", help="Run grounded Mistral generation and validation")
    args = parser.parse_args()

    if args.answer:
        for question in args.questions:
            result = answer(question)
            if args.json:
                print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
                continue
            print(f"\nQ: {question}")
            print(f"  class    : {result.retrieval.query_class.value:<14} "
                  f"decision={result.retrieval.decision:<14} grounded={result.grounded}")
            print(f"  fund     : {_category_label(result.retrieval.fund.fund_category if result.retrieval.fund else None)}")
            if result.retrieval.evidence:
                for row in result.retrieval.evidence:
                    print(f"    {row.distance:>5.3f}  [{row.fund_category}/{row.section_heading}] "
                          f"{row.chunk_text.splitlines()[0][:60]}")
            print(f"  answer   : {result.answer}")
        return

    retriever = Retriever()
    for question in args.questions:
        result = retriever.retrieve(question)
        if args.json:
            print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
            continue
        print(f"\nQ: {question}")
        print(f"  class    : {result.query_class.value:<14} decision={result.decision:<14} "
              f"blocked={result.blocked} embedded={result.embedded}")
        print(f"  fund     : {_category_label(result.fund.fund_category if result.fund else None)}")
        if result.evidence:
            for row in result.evidence:
                print(f"    {row.distance:>5.3f}  [{row.fund_category}/{row.section_heading}] "
                      f"{row.chunk_text.splitlines()[0][:60]}")
        if result.conflicts:
            print(f"  conflicts: {result.conflicts}")
        if result.message:
            print(f"  message  : {result.message}")


if __name__ == "__main__":
    main()