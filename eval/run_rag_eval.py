"""End-to-end RAG eval: retrieval + generation + guardrails, per language.

Where run_retrieval_eval.py asks "did we find the right page", this asks
"was the answer any good" -- and it needs a live LLM, so it is the heavier of
the two.

Metrics, all computed locally with no paid judge:

  citation_rate    fraction of answers carrying at least one valid [Sn] marker
  invented_rate    fraction that cited a source number that does not exist --
                   the worst failure mode here, because a fake citation looks
                   checkable to a student who then finds nothing
  groundedness     lexical overlap between the answer and its retrieved
                   sources; a cheap proxy that catches wholesale drift, not
                   subtle factual inversion (plan.md Sec.E schedules HHEM for that)
  language_match   did a Hindi question get a Hindi answer
  refusal_rate     how often the evidence gate declined to answer
  latency          seconds per question

Read citation_rate and groundedness together: a high citation rate with low
groundedness means the model is decorating invented prose with markers.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from story_mvp.guardrails import LOW_GROUNDEDNESS  # noqa: E402
from story_mvp.rag_chat import answer_question  # noqa: E402

DEVA = re.compile(r"[ऀ-ॿ]")


def language_matches(answer: str, language: str) -> bool:
    """Devanagari is the only script signal available without a language ID model."""
    has_deva = bool(DEVA.search(answer or ""))
    return has_deva if language in {"hindi", "marathi"} else not has_deva


def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-end RAG eval against the golden set.")
    parser.add_argument("--golden", default="eval/golden/golden_set.jsonl")
    parser.add_argument("--dataset-dir", default="knowledge_base")
    parser.add_argument("--limit", type=int, default=60,
                        help="Questions per run. Generation is the slow part.")
    parser.add_argument("--report", default="eval/reports/rag_eval.json")
    parser.add_argument("--no-llm", action="store_true",
                        help="Retrieval-only pass, to separate retrieval failures from generation ones.")
    args = parser.parse_args()

    records = [json.loads(l) for l in Path(args.golden).read_text(encoding="utf-8").splitlines() if l.strip()]
    if not records:
        print("golden set is empty")
        return 1

    # Balance across languages rather than taking the first N, which would be
    # dominated by whichever language sorts first.
    by_lang = defaultdict(list)
    for r in records:
        by_lang[r.get("language", "unknown")].append(r)
    per_lang = max(1, args.limit // max(len(by_lang), 1))
    sample = [r for rows in by_lang.values() for r in rows[:per_lang]]

    print(f"evaluating {len(sample)} questions across {len(by_lang)} languages")

    from story_mvp.rag_engine import RetrievalService

    engine = RetrievalService(dataset_dir=args.dataset_dir)

    llm = None
    if not args.no_llm:
        from story_mvp.llm_client import StoryLLM

        llm = StoryLLM()

    stats = defaultdict(lambda: defaultdict(list))
    rows = []

    for i, rec in enumerate(sample, 1):
        lang = rec.get("language", "unknown")
        start = time.time()
        out = answer_question(
            {
                "question": rec["question"],
                "language": lang,
                "class_level": rec.get("class_level") or "",
                "subject": rec.get("subject") or "",
            },
            engine,
            llm,
        )
        elapsed = time.time() - start

        answer = out.get("answer", "")
        refused = not out.get("grounded", False)
        cited = bool(out.get("citations_used"))
        invented = bool(out.get("invented_citations"))
        ground = out.get("groundedness")

        stats[lang]["latency"].append(elapsed)
        stats[lang]["refused"].append(1 if refused else 0)
        stats[lang]["cited"].append(1 if cited else 0)
        stats[lang]["invented"].append(1 if invented else 0)
        stats[lang]["lang_ok"].append(1 if language_matches(answer, lang) else 0)
        if ground is not None:
            stats[lang]["ground"].append(ground)
        stats[lang]["provider"].append(out.get("model_provider", "?"))

        rows.append({
            "question": rec["question"][:160], "language": lang,
            "provider": out.get("model_provider"), "grounded": out.get("grounded"),
            "groundedness": ground, "citations": out.get("citations_used"),
            "invented": out.get("invented_citations"), "answer": answer[:400],
        })

        if i % 10 == 0:
            print(f"  {i}/{len(sample)}")

    def pct(values):
        return round(100 * sum(values) / len(values), 1) if values else 0.0

    print("\n" + "=" * 84)
    print(f"{'lang':<10} {'n':>4} {'cited%':>8} {'invented%':>10} {'ground':>8} "
          f"{'lang_ok%':>9} {'refused%':>9} {'sec':>6}")
    print("-" * 84)
    summary = {}
    for lang in sorted(stats):
        s = stats[lang]
        n = len(s["latency"])
        row = {
            "n": n,
            "citation_rate": pct(s["cited"]),
            "invented_rate": pct(s["invented"]),
            "groundedness": round(statistics.mean(s["ground"]), 3) if s["ground"] else None,
            "language_match": pct(s["lang_ok"]),
            "refusal_rate": pct(s["refused"]),
            "mean_latency_s": round(statistics.mean(s["latency"]), 2),
            "providers": dict((p, s["provider"].count(p)) for p in set(s["provider"])),
        }
        summary[lang] = row
        print(f"{lang:<10} {n:>4} {row['citation_rate']:>8} {row['invented_rate']:>10} "
              f"{str(row['groundedness']):>8} {row['language_match']:>9} "
              f"{row['refusal_rate']:>9} {row['mean_latency_s']:>6}")
    print("=" * 84)

    providers = {p for s in stats.values() for p in s["provider"]}
    if providers <= {"passages_only", "none", "conversational"}:
        print("\nNOTE: no LLM was reachable, so generation metrics reflect the")
        print("passage fallback, not a real model. Start Ollama or vLLM first.")

    out_path = Path(args.report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\nwrote {out_path}")
    print(f"(groundedness below {LOW_GROUNDEDNESS} means the answer barely shares "
          f"vocabulary with its sources)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
