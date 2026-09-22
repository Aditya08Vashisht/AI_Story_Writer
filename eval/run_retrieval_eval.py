"""Measure retrieval quality per language against the golden set.

Needs no LLM -- this evaluates the retriever alone, so it runs while the
generator is still queued.

METHODOLOGICAL CAVEAT, read before trusting the numbers:
`gold_chunk_ids` is the chunk each question was *extracted from* -- i.e. the
chunk holding the exercise list -- not necessarily the chunk holding the
answer. So exact-chunk Recall@k measures "can retrieval find the page this
question came from", which is a proxy for, not the same as, "can retrieval
find what's needed to answer it".

That is why chapter_hit@k is reported alongside: it asks the softer and
arguably more meaningful question of whether retrieval landed in the right
chapter at all. Read both together.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from story_mvp.rag_engine import RetrievalService


def load_golden(path: Path, only_verified: bool) -> list:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if only_verified:
        records = [r for r in records if r.get("status") in {"keep_clean", "keep_noisy"}]
    return records


def evaluate(engine, records, top_k: int) -> dict:
    per_lang = defaultdict(lambda: defaultdict(float))
    per_lang_n = defaultdict(int)
    latencies = []

    for rec in records:
        lang = rec.get("language") or "unknown"
        gold_ids = set(rec.get("gold_chunk_ids") or [])
        gold_chapter = rec.get("gold_chapter")

        filters = {
            "class_level": rec.get("class_level") or "",
            "subject": rec.get("subject") or "",
            "language": lang if lang in {"english", "hindi", "marathi"} else "",
        }

        start = time.time()
        results = engine.retrieve(query=rec["question"], filters=filters, top_k=top_k)
        latencies.append(time.time() - start)

        retrieved_ids = [r.get("id") for r in results]
        retrieved_chapters = [r.get("chapter") for r in results]

        per_lang_n[lang] += 1
        for k in (1, 5, 10):
            if any(rid in gold_ids for rid in retrieved_ids[:k]):
                per_lang[lang][f"recall@{k}"] += 1
            if gold_chapter and gold_chapter in retrieved_chapters[:k]:
                per_lang[lang][f"chapter_hit@{k}"] += 1

        # MRR over the exact-chunk match
        for rank, rid in enumerate(retrieved_ids, 1):
            if rid in gold_ids:
                per_lang[lang]["mrr"] += 1.0 / rank
                break

        if results:
            per_lang[lang]["_score_sum"] += float(results[0].get("retrieval_score") or 0.0)
            per_lang[lang]["_empty"] += 0
        else:
            per_lang[lang]["_empty"] += 1

    summary = {}
    for lang, n in per_lang_n.items():
        row = {"n": n}
        for metric, total in per_lang[lang].items():
            if metric.startswith("_"):
                continue
            row[metric] = round(total / n, 3)
        row["empty_results"] = int(per_lang[lang]["_empty"])
        row["mean_top1_score"] = round(per_lang[lang]["_score_sum"] / max(n - per_lang[lang]["_empty"], 1), 3)
        summary[lang] = row

    summary["_meta"] = {
        "questions": sum(per_lang_n.values()),
        "top_k": top_k,
        "mean_latency_s": round(sum(latencies) / max(len(latencies), 1), 3),
        "rerank_enabled": os.environ.get("STORYTUTOR_ENABLE_RERANK", "") or "0",
        "embedding_model": engine.model_name,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Per-language retrieval eval against the golden set.")
    parser.add_argument("--golden", default="eval/golden/golden_set_verified.jsonl")
    parser.add_argument("--dataset-dir", default="knowledge_base")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--only-verified", action="store_true",
                        help="Use only the 50 human-reviewed keeps. Smaller but trustworthy.")
    parser.add_argument("--report", default="eval/reports/retrieval_eval.json")
    args = parser.parse_args()

    records = load_golden(Path(args.golden), args.only_verified)
    if not records:
        print("No records to evaluate. Did you mean to drop --only-verified?")
        return 1
    print(f"evaluating {len(records)} questions (top_k={args.top_k})")

    engine = RetrievalService(dataset_dir=args.dataset_dir)
    summary = evaluate(engine, records, args.top_k)

    print("\n" + "=" * 78)
    header = f"{'lang':<10} {'n':>4} {'R@1':>6} {'R@5':>6} {'R@10':>6} {'ch@1':>6} {'ch@5':>6} {'MRR':>6} {'empty':>6}"
    print(header)
    print("-" * 78)
    for lang in sorted(k for k in summary if not k.startswith("_")):
        r = summary[lang]
        print(f"{lang:<10} {r['n']:>4} {r.get('recall@1',0):>6} {r.get('recall@5',0):>6} "
              f"{r.get('recall@10',0):>6} {r.get('chapter_hit@1',0):>6} {r.get('chapter_hit@5',0):>6} "
              f"{r.get('mrr',0):>6} {r['empty_results']:>6}")
    print("=" * 78)
    print(f"mean latency: {summary['_meta']['mean_latency_s']}s   "
          f"rerank: {summary['_meta']['rerank_enabled']}   "
          f"embeddings: {summary['_meta']['embedding_model']}")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {report_path}")

    print("\nReminder: recall@k uses the chunk each question was EXTRACTED from,")
    print("not necessarily the chunk containing its answer. Read chapter_hit@k")
    print("alongside it -- a low recall@k with a high chapter_hit@k means")
    print("retrieval is landing in the right chapter but on a different page.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
