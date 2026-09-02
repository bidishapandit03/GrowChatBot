"""Phase 6 retrieval evaluation and threshold calibration.

Runs the manually labelled dataset (tests/retrieval_cases.json) through the real
retriever and reports:

- Safety gate rate (blocked queries must never embed)
- Precision of intent and fund classification
- Recall@4 / grounded-evidence rate for eligible factual questions
- Absent-fact behaviour at retrieval

Metric split: cosine distance cannot separate in-scope absent-fact questions from
genuine facts (the embedder weights fund-name tokens over topic), so the spec's
"<5% unsupported claims" is a generation-layer property. It is therefore measured
end-to-end with --live (retrieval -> grounded generation -> validation), while
the deterministic retrieval run gates safety (100%) and recall@4 (>=90%). The
--calibrate sweep picks a threshold that keeps every true fact and rejects the
clearly-foreign absent queries, and persists it to data/threshold.json.

Usage:
    python -m code.evaluate                 Run the full labelled evaluation
    python -m code.evaluate --calibrate     Report and pick a relevance threshold
    python -m code.evaluate --live          End-to-end unsupported-claim check
    python -m code.evaluate --json          Machine-readable result of the run
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

from code.config import THRESHOLD_FILE
from code.rag.retriever import EvidenceRow, Retriever

RETRIEVAL_CASES_FILE = Path(__file__).resolve().parent.parent / "tests" / "retrieval_cases.json"

EXPECTED_OUTCOMES = {"found", "not_found", "blocked", "clarification"}


@dataclass(frozen=True)
class CaseCheck:
    id: str
    category: str
    question: str
    expected_outcome: str
    expected_class: str
    expected_fund: str | None
    decision: str
    query_class: str
    fund: str | None
    embedded: bool
    checks: dict[str, bool] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    best_distance: float | None = None
    fact_distance: float | None = None
    fact_rank: int | None = None
    soft: bool = False

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(self.checks.values())


def assess_case(case: dict, decision: str, query_class: str, fund: str | None,
                embedded: bool, evidence: list[EvidenceRow]) -> CaseCheck:
    """Pure per-case evaluation against the labelled expectations (no I/O)."""
    checks: dict[str, bool] = {}
    reasons: list[str] = []
    expected_outcome = case["expected_outcome"]

    checks["outcome"] = decision == expected_outcome
    if not checks["outcome"]:
        reasons.append(f"outcome expected={expected_outcome} got={decision}")

    if expected_outcome == "blocked":
        checks["class"] = query_class == case["expected_class"]
        if not checks["class"]:
            reasons.append(f"class expected={case['expected_class']} got={query_class}")
        checks["safe"] = embedded is False
        if not checks["safe"]:
            reasons.append("blocked query reached embedding")
    elif expected_outcome == "clarification":
        checks["not_embedded"] = embedded is False
        if not checks["not_embedded"]:
            reasons.append("clarification query should not embed")
    else:
        checks["embedded"] = embedded is True
        if not checks["embedded"]:
            reasons.append("eligible query did not embed")

    if case.get("expected_fund"):
        checks["fund"] = fund == case["expected_fund"]
        if not checks["fund"]:
            reasons.append(f"fund expected={case['expected_fund']} got={fund}")

    if expected_outcome == "found":
        expected_url = case["expected_url"]
        fact = case.get("expected_fact_text")
        evidence_for_url = [row for row in evidence if row.canonical_url == expected_url]
        checks["recall"] = any(fact in row.chunk_text for row in evidence_for_url)
        if not checks["recall"]:
            reasons.append(
                f"fact {fact!r} not in retrieved evidence for {expected_url}"
                + (f" (present urls: {sorted({r.canonical_url for r in evidence})})" if evidence else " (empty)"
                )
            )
        if evidence:
            checks["distance_ok"] = min(row.distance for row in evidence) >= 0.0
    elif expected_outcome == "not_found":
        checks["no_evidence"] = not evidence
        if not checks["no_evidence"]:
            reasons.append(f"expected no evidence, got {len(evidence)} rows")

    best_distance = min((row.distance for row in evidence), default=None)
    return CaseCheck(
        id=case["id"],
        category=case.get("category", ""),
        question=case.get("question", ""),
        expected_outcome=expected_outcome,
        expected_class=case["expected_class"],
        expected_fund=case.get("expected_fund"),
        decision=decision,
        query_class=query_class,
        fund=fund,
        embedded=embedded,
        checks=checks,
        reasons=reasons,
        best_distance=best_distance,
        soft=bool(case.get("soft", False)),
    )


def load_cases(path: Path | None = None) -> list[dict]:
    path = path or RETRIEVAL_CASES_FILE
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload["cases"]
    for case in cases:
        if case["expected_outcome"] not in EXPECTED_OUTCOMES:
            raise ValueError(f"bad expected_outcome in case {case['id']}")
        if "expected_class" not in case or "question" not in case or "expected_outcome" not in case:
            raise ValueError(f"case {case['id']} is missing required fields")
    return cases


def run_evaluation(cases: list[dict], retriever: Retriever | None = None) -> list[CaseCheck]:
    retriever = retriever or Retriever()
    checks: list[CaseCheck] = []
    for case in cases:
        result = retriever.retrieve(case["question"])
        check = assess_case(
            case,
            decision=result.decision,
            query_class=result.query_class.value,
            fund=result.fund.fund_category if result.fund else None,
            embedded=result.embedded,
            evidence=result.evidence,
        )
        if check.soft:
            checks.append(check)
            continue
        # Calibration needs pre-threshold distances that the decision path discards.
        # For found cases, record the distance/rank of the row that actually carries
        # the verified fact (it is not always the closest row) so the gate is sized
        # to recover every true fact.
        fund, raw = retriever.raw_top(case["question"])
        if check.expected_outcome == "found":
            fact = case.get("expected_fact_text")
            hits = [i for i, row in enumerate(raw) if row.canonical_url == case["expected_url"] and fact in row.chunk_text]
            if hits:
                idx = min(hits)
                check = replace(check, fact_distance=raw[idx].distance, fact_rank=idx)
        elif check.expected_outcome == "not_found" and raw:
            check = replace(check, best_distance=min(row.distance for row in raw))
        checks.append(check)
    return checks


def summarize(checks: list[CaseCheck]) -> dict:
    hard = [c for c in checks if not c.soft]
    blocked = [c for c in hard if c.expected_outcome == "blocked"]
    found = [c for c in hard if c.expected_outcome == "found"]
    not_found = [c for c in hard if c.expected_outcome == "not_found"]
    clarified = [c for c in hard if c.expected_outcome == "clarification"]
    soft = [c for c in checks if c.soft]

    def rate(items: list, key: str) -> tuple[int, int]:
        total = len(items)
        passed = sum(1 for c in items if c.checks.get(key))
        return int(total), int(passed)

    t_blocked, p_safe = rate(blocked, "safe")
    t_found, p_recall = rate(found, "recall")
    t_notfound, p_notfound = rate(not_found, "no_evidence")
    t_outcome, p_outcome = rate(hard, "outcome")

    # Absent-fact questions that surfaced non-empty evidence are NOT counted as
    # unsupported claims here: for in-scope fund questions the MiniLM embedder
    # cannot separate them by cosine distance (see --calibrate), and the
    # grounded-answer guard is what refuses them. That is measured end-to-end
    # with --live.
    return {
        "recall_floor": 0.9,
        "safety_gate": {"tested": t_blocked, "passed": p_safe, "rate": round(p_safe / t_blocked, 3) if t_blocked else None},
        "safety_pass": t_blocked > 0 and p_safe == t_blocked,
        "recall_at_4": {"tested": t_found, "passed": p_recall, "rate": round(p_recall / t_found, 3) if t_found else None},
        "recall_pass": t_found > 0 and p_recall / t_found >= 0.9,
        "not_found_at_retrieval": {"tested": t_notfound, "passed": p_notfound,
                                   "rate": round(p_notfound / t_notfound, 3) if t_notfound else None},
        "unsupported_at_retrieval": round((t_notfound - p_notfound) / t_notfound, 3) if t_notfound else None,
        "outcome_rate": round(p_outcome / t_outcome, 3) if t_outcome else None,
        "clarified": len(clarified),
        "soft_probes": len(soft),
        "generated_pass": p_outcome / t_outcome >= 0.95 if t_outcome else True,
    }


def _best_distance_for(case: dict) -> float | None:
    """Closest pre-threshold distance for calibration purposes."""
    fund, rows = Retriever().raw_top(case["question"])
    if fund is None or not rows:
        return None
    return min(row.distance for row in rows)


def calibrate(checks: list[CaseCheck]) -> dict:
    """Score candidate thresholds on found/not-found separation.

    The retained-distance for a found case is the distance of the evidence row
    that actually carries the verified fact (``fact_distance``), not the global
    minimum over the top-4, so the gate is sized to recover every true fact with
    one grid-step of margin. In-scope absent-fact questions can still overlap the
    found cluster in cosine distance (the MiniLM embedder weights fund-name
    tokens over topic); the grounded-answer guard handles those, measured --live.
    """
    recall_floor = 0.9
    found = [c for c in checks if not c.soft and c.expected_outcome == "found"]
    absent = [c for c in checks if not c.soft and c.expected_outcome == "not_found"]

    found_dists = sorted(c.fact_distance for c in found if c.fact_distance is not None)
    not_recovered = [c.id for c in found if c.fact_distance is None]
    absent_dists = sorted(c.best_distance for c in absent if c.best_distance is not None)

    grid_step = 0.025
    candidates = [round(0.30 + grid_step * i, 3) for i in range(25)]
    rows = []
    for threshold in candidates:
        retained = sum(1 for d in found_dists if d <= threshold)
        accepted = sum(1 for d in absent_dists if d <= threshold)
        recall = retained / len(found) if found else 0.0
        rows.append({
            "threshold": threshold,
            "found_retained": retained,
            "recall": round(recall, 3),
            "absent_accepted": accepted,
            "absent_rate": round(accepted / len(absent_dists), 3) if absent_dists else None,
            "found_rejected_rate": round(1.0 - recall, 3),
        })

    if not found_dists:
        return {"recommended_threshold": None, "sweep": rows, "found_eligible": len(found),
                "absent_eligible": len(absent), "eligible_fact_distances": [],
                "absent_best_distances": absent_dists, "not_recovered_at_top_k": not_recovered}

    max_fact = found_dists[-1]
    min_absent = absent_dists[0] if absent_dists else None
    recommended = round(max_fact + grid_step, 3)
    if min_absent is not None and recommended >= min_absent:
        recommended = round(min_absent - grid_step, 3)
    if recommended < 0.3:
        recommended = 0.3

    return {
        "found_eligible": len(found),
        "absent_eligible": len(absent),
        "eligible_fact_distances": [round(d, 3) for d in found_dists],
        "absent_best_distances": [round(d, 3) for d in absent_dists],
        "not_recovered_at_top_k": not_recovered,
        "recall_floor": recall_floor,
        "recommended_threshold": recommended,
        "sweep": rows,
    }


def _print_checks(checks: list[CaseCheck]) -> None:
    print(f"\n{'id':<30} {'class':<14} {'expected':<12} {'decision':<12} pass  best_d")
    for c in checks:
        mark = "PASS" if c.passed else "FAIL"
        print(
            f"{c.id:<30} {c.query_class:<14} {c.expected_outcome:<12} {c.decision:<12} "
            f"{mark:<4} {c.best_distance if c.best_distance is not None else '-'}"
        )
        for reason in c.reasons:
            print(f"  ! {reason}")


def _ensure_api_key() -> str:
    import os

    key = os.environ.get("MISTRAL_API_KEY")
    if key:
        return key
    from dotenv import load_dotenv

    load_dotenv()
    key = os.environ.get("MISTRAL_API_KEY")
    if not key:
        raise SystemExit("--live requires MISTRAL_API_KEY (create .env and restart)")
    return key


_REFUSAL_HEDGES = (
    "does not provide", "do not provide", "does not mention", "do not mention",
    "does not say", "do not say", "does not list", "do not list",
    "does not disclose", "do not disclose", "does not name", "do not name",
    "does not include", "do not include", "not available", "not specified",
    "not disclosed", "not listed", "not found", "not mentioned", "not named",
    "no information", "no mention", "no record", "no listing", "cannot",
    "could not", "absent",
)


def is_refusal(answer: str | None) -> bool:
    """True when the answer explicitly declines rather than asserting a fact."""
    if not answer:
        return False
    lower = answer.lower()
    return any(hedge in lower for hedge in _REFUSAL_HEDGES)


def run_live_unsupported(cases: list[dict]) -> dict:
    """End-to-end unsupported-claim check: absent-fact questions must never ground.

    Exercises the full answer() path (retrieval -> generation -> validation) for
    every not_found-labelled question, including the soft distance-overlap probes
    that retrieval cannot separate. A grounded answer that explicitly states the
    evidence lacks the fact (hedged refusal) is safe; a grounded substantive claim
    on an absent fact counts as an unsupported claim.
    """
    from code.rag.pipeline import answer

    _ensure_api_key()
    absent = [c for c in cases if c["expected_outcome"] == "not_found"]
    results = []
    for case in absent:
        outcome = answer(case["question"])
        refused = not outcome.grounded or is_refusal(outcome.answer)
        results.append({
            "id": case["id"],
            "question": case["question"],
            "grounded": outcome.grounded,
            "decision": outcome.retrieval.decision,
            "refused": refused,
            "unsupported": not refused,
        })
    accepted = sum(1 for r in results if r["unsupported"])
    return {
        "absent_tested": len(results),
        "unsupported_claims": accepted,
        "unsupported_claim_acceptance": round(accepted / len(results), 3) if results else None,
        "cases": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6 retrieval evaluation")
    parser.add_argument("--calibrate", action="store_true", help="Pick and persist a relevance threshold")
    parser.add_argument("--live", action="store_true", help="Run the end-to-end unsupported-claim check (needs MISTRAL_API_KEY)")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()

    cases = load_cases()
    checks = run_evaluation(cases)
    summary = summarize(checks)

    live: dict | None = None
    if args.live:
        live = run_live_unsupported(cases)

    calibration: dict | None = None
    if args.calibrate:
        calibration = calibrate(checks)
        best = calibration["recommended_threshold"]
        if best is not None:
            THRESHOLD_FILE.write_text(json.dumps({"relevance_threshold": best}, indent=2) + "\n", encoding="utf-8")

    if args.json:
        payload: dict = {"summary": summary}
        if calibration is not None:
            payload["calibration"] = calibration
        if live is not None:
            payload["live_unsupported"] = live
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    print("=== Phase 6 evaluation ===")
    _print_checks(checks)
    print("\n--- metrics ---")
    print(json.dumps(summary, indent=2))
    if live is not None:
        print("\n--- end-to-end unsupported-claim check ---")
        for row in live["cases"]:
            mark = "UNSUPPORTED" if row["unsupported"] else "ok"
            print(f"{row['id']:<34} decision={row['decision']:<11} grounded={row['grounded']} {mark}")
        print(f"unsupported_claim_acceptance: {live['unsupported_claim_acceptance']} ({live['unsupported_claims']}/{live['absent_tested']})")
    if calibration is not None:
        print("\n--- calibration ---")
        print(f"eligible(found) fact-row distances: {calibration['eligible_fact_distances']}")
        print(f"absent      best distances: {calibration['absent_best_distances']}")
        if calibration.get("not_recovered_at_top_k"):
            print(f"fact NOT recovered in top-k (recall failure, not a threshold effect): {calibration['not_recovered_at_top_k']}")
        print(f"recommended threshold: {calibration['recommended_threshold']} -> data/threshold.json")
        if calibration["recommended_threshold"] is None:
            print("WARNING: no threshold meets the recall floor; review evidence.")


if __name__ == "__main__":
    main()
    sys.exit(0)