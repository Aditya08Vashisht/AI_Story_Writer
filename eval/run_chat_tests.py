"""Run the 53 chatbot test questions against a live server and check behaviour.

  python eval/run_chat_tests.py                       # needs the app on :5050
  python eval/run_chat_tests.py --only grounded_hindi
  python eval/run_chat_tests.py --url http://127.0.0.1:5050

This checks BEHAVIOUR, not exact wording: did it answer from sources when it
should, refuse when it should, reply in the right script, avoid inventing a
citation, block injection, redact a phone number. Exact answer text varies run
to run and is not something a test can pin.

Questions tagged `known_gap` are expected to fail today (follow-ups need
history-aware retrieval, plan.md Phase C). They are reported separately so a
known limitation is never mistaken for a regression -- and so the day they
start passing is visible.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

DEVA = re.compile(r"[ऀ-ॿ]")
HERE = Path(__file__).resolve().parent


def ask(url: str, item: dict) -> dict:
    payload = {
        "question": item["question"],
        "language": item.get("language", "english"),
        "class_level": item.get("class_level", ""),
        "subject": item.get("subject", ""),
        "history": item.get("history", []),
    }
    req = urllib.request.Request(
        f"{url.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=240) as r:
        return json.loads(r.read().decode("utf-8"))


def check(item: dict, resp: dict) -> list:
    """Return a list of failures. Empty means the behaviour was right."""
    exp = item.get("expect", {})
    answer = resp.get("answer") or ""
    fails = []

    if exp.get("blocked") is True:
        if resp.get("intent") != "blocked":
            fails.append("expected the input guardrail to block this")
        return fails
    if exp.get("blocked") is False and resp.get("intent") == "blocked":
        fails.append("blocked a legitimate question")

    if "pii_redacted" in exp and exp["pii_redacted"] in (resp.get("question") or ""):
        fails.append("personal data reached the pipeline unredacted")

    if exp.get("grounded") is True and not resp.get("grounded"):
        fails.append("should have answered from the textbooks but found no sources")
    if exp.get("grounded") is False:
        if resp.get("grounded"):
            fails.append("answered an out-of-syllabus question as if grounded")
        if resp.get("citations_used"):
            fails.append("cited sources for an out-of-syllabus question")

    if exp.get("conversational"):
        if resp.get("citations_used"):
            fails.append("cited textbook sources in a greeting")
        if not answer.strip():
            fails.append("empty reply to a greeting")

    if resp.get("invented_citations"):
        fails.append(f"model invented citations {resp['invented_citations']} (stripped by guardrail)")

    lang = item.get("language", "english")
    if answer.strip() and lang in {"hindi", "marathi"} and not DEVA.search(answer):
        fails.append(f"asked for {lang} but the reply has no Devanagari")

    if not answer.strip():
        fails.append("empty answer")
    return fails


def main() -> int:
    ap = argparse.ArgumentParser(description="Behavioural tests for the RAG chatbot.")
    ap.add_argument("--url", default="http://127.0.0.1:5050")
    ap.add_argument("--questions", default=str(HERE / "chat_questions.json"))
    ap.add_argument("--only", help="run a single category")
    ap.add_argument("--report", default=str(HERE / "reports" / "chat_test_report.json"))
    args = ap.parse_args()

    items = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    if args.only:
        items = [i for i in items if i["category"] == args.only]
    print(f"running {len(items)} questions against {args.url}\n")

    rows, by_cat = [], defaultdict(lambda: {"pass": 0, "fail": 0, "gap": 0})
    for item in items:
        t0 = time.time()
        try:
            resp = ask(args.url, item)
            fails = check(item, resp)
        except Exception as exc:  # noqa: BLE001
            resp, fails = {}, [f"request failed: {exc}"]
        secs = time.time() - t0

        gap = bool(item.get("known_gap"))
        status = "PASS" if not fails else ("GAP " if gap else "FAIL")
        by_cat[item["category"]]["pass" if not fails else ("gap" if gap else "fail")] += 1

        print(f"  {status} {item['id']:<5} {secs:5.1f}s  {item['question'][:58]}")
        for f in fails:
            print(f"         - {f}")

        rows.append({
            "id": item["id"], "category": item["category"], "question": item["question"],
            "status": status.strip(), "failures": fails, "seconds": round(secs, 2),
            "provider": resp.get("model_provider"), "grounded": resp.get("grounded"),
            "sources": len(resp.get("sources") or []),
            "answer": (resp.get("answer") or "")[:500],
        })

    print("\n" + "=" * 62)
    print(f"{'category':<20} {'pass':>5} {'fail':>5} {'known gap':>10}")
    print("-" * 62)
    tp = tf = tg = 0
    for cat, c in by_cat.items():
        print(f"{cat:<20} {c['pass']:>5} {c['fail']:>5} {c['gap']:>10}")
        tp, tf, tg = tp + c["pass"], tf + c["fail"], tg + c["gap"]
    print("-" * 62)
    print(f"{'TOTAL':<20} {tp:>5} {tf:>5} {tg:>10}")

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nfull answers written to {args.report}")
    return 0 if tf == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
