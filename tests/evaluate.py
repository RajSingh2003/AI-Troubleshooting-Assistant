"""Evaluate retrieval + unknown-issue handling on tests/test_queries.json.

Usage:
    python -m tests.evaluate            # uses GEMINI_API_KEY if set (hybrid + LLM)
    python -m tests.evaluate --offline  # keyword retrieval only, no API calls
    python -m tests.evaluate --retrieval-only   # skip LLM calls (cheaper)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.factory import build_assistant

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--retrieval-only", action="store_true")
    args = ap.parse_args()

    bot = build_assistant(offline=args.offline)
    cases = json.loads((HERE / "test_queries.json").read_text())
    print(f"Mode: {bot.retriever.mode} | threshold={bot.threshold:.2f}\n")

    passed = 0
    for c in cases:
        q, exp = c["query"], c["expected"]
        if args.retrieval_only:
            hits = bot.retriever.search(q, top_k=3)
            top = hits[0]
            got = top.entry["id"] if top.score >= bot.threshold else "UNKNOWN"
            detail = f"score={top.score:.2f} top={top.entry['id']}"
        else:
            ans = bot.ask(q)
            if ans.status == "answered" and ans.sources:
                got = ans.sources[0] if exp == "UNKNOWN" else (exp if exp in ans.sources else ans.sources[0])
            else:
                got = "UNKNOWN"
            top = ans.hits[0] if ans.hits else None
            detail = f"status={ans.status} score={top.score:.2f} top={top.entry['id']}" if top else f"status={ans.status}"
        ok = got == exp
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  expected={exp:8} got={got:8} {detail}  | {q}")

    print(f"\n{passed}/{len(cases)} passed ({passed / len(cases):.0%})")


if __name__ == "__main__":
    main()
