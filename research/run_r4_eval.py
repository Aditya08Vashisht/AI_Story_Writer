"""R4: retrieval robustness by query style -- native vs romanized vs code-switched.

The chat tests already show the phenomenon: "photosynthesis kya hota hai?"
finds nothing while "पौधे अपना भोजन कैसे बनाते हैं?" answers fine
(eval/reports chat_test_report.json, rb02). This measures it per variant
instead of anecdotally.

Gold answers itself from the corpus, keeping the eval self-contained:

    For each (concept, language) the NATIVE-script query of the same language
    defines the gold top-1 chunk; if a language has no native row, the
    concept's English native query is used and the comparison is marked
    cross_lingual (chapter-level agreement only -- chunk ids cannot match
    across languages).

    Native rows are scored against their own gold, so they appear at 100% --
    they are the reference row, not a result.

Metrics per variant: top1_match% (same-language gold), chapter_hit% (gold
chapter in top-k, the language-agnostic signal), mean top-1 score, empty
rate, latency. Retrieval only -- no LLM, so it runs in ~a minute.

Usage (on Sol, storytutor env):
    python research/run_r4_eval.py
    python research/run_r4_eval.py --top-k 5 --report research/reports/r4_retrieval.json
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


def load_queries(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    bad = [r.get("id") for r in rows if not {"id", "concept", "variant", "language", "query"} <= set(r)]
    if bad:
        raise SystemExit(f"malformed rows (missing keys): {bad}")
    return rows


def filters_for(q: dict) -> dict:
    """The same metadata filters the product path sends (class/subject/language)."""
    return {"class_level": str(q.get("class_level", "")),
            "subject": q.get("subject", ""),
            "language": q.get("language", "")}


def build_gold(queries: list[dict], engine, top_k: int) -> dict:
    """(concept, language) -> gold chunk, taken from the native-script query."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for q in queries:
        groups[q["concept"]].append(q)

    gold: dict[tuple, dict] = {}
    for concept, group in groups.items():
        natives = [q for q in group if q["variant"] == "native"]
        if not natives:
            continue
        for lang in {q["language"] for q in group}:
            same = [n for n in natives if n["language"] == lang]
            if same:
                pick, cross = same[0], False
            else:
                en = [n for n in natives if n["language"] == "english"]
                if not en:
                    continue
                pick, cross = en[0], True
            docs = engine.retrieve(pick["query"], top_k=1, filters=filters_for(pick))
            if docs:
                gold[(concept, lang)] = {
                    "chunk_id": str(docs[0].get("id")),
                    "chapter": str(docs[0].get("chapter")),
                    "by": pick["id"],
                    "cross_lingual": cross,
                }
    return gold


def evaluate(queries: list[dict], gold: dict, engine, top_k: int) -> list[dict]:
    rows = []
    for q in queries:
        g = gold.get((q["concept"], q["language"]))
        t0 = time.time()
        docs = engine.retrieve(q["query"], top_k=top_k, filters=filters_for(q))
        ids = [str(d.get("id")) for d in docs]
        chapters = [str(d.get("chapter")) for d in docs]
        rows.append({
            "id": q["id"], "concept": q["concept"], "variant": q["variant"],
            "language": q["language"], "query": q["query"],
            "gold_chunk": g["chunk_id"] if g else None,
            "gold_by": g["by"] if g else None,
            "cross_lingual_gold": bool(g and g["cross_lingual"]),
            "top1_match": bool(g and ids and ids[0] == g["chunk_id"]),
            "chapter_hit": bool(g and chapters and g["chapter"] in chapters),
            "top1_score": round(float(docs[0].get("retrieval_score", 0.0)), 4) if docs else None,
            "n_results": len(docs),
            "seconds": round(time.time() - t0, 3),
        })
    return rows


def summarize(rows: list[dict]) -> dict:
    by_variant: dict[str, dict] = {}
    for variant in sorted({r["variant"] for r in rows}):
        v = [r for r in rows if r["variant"] == variant]
        same = [r for r in v if not r["cross_lingual_gold"] and r["gold_chunk"]]
        scored = [r["top1_score"] for r in v if r["top1_score"] is not None]
        by_variant[variant] = {
            "n": len(v),
            "top1_match_pct": round(100 * sum(r["top1_match"] for r in same) / len(same), 1) if same else None,
            "chapter_hit_pct": round(100 * sum(r["chapter_hit"] for r in v if r["gold_chunk"]) /
                                     max(1, len([r for r in v if r["gold_chunk"]])), 1),
            "empty_pct": round(100 * sum(1 for r in v if not r["n_results"]) / len(v), 1),
            "mean_top1_score": round(sum(scored) / len(scored), 4) if scored else None,
            "mean_latency_s": round(sum(r["seconds"] for r in v) / len(v), 3),
            "cross_lingual_n": sum(1 for r in v if r["cross_lingual_gold"]),
        }
    return by_variant


def print_table(summary: dict) -> None:
    print("\n" + "=" * 86)
    print(f"{'variant':<18} {'n':>4} {'top1%':>7} {'ch@k%':>7} {'empty%':>8} "
          f"{'score':>7} {'sec':>6} {'xlang':>6}")
    print("-" * 86)
    for variant, s in summary.items():
        print(f"{variant:<18} {s['n']:>4} {str(s['top1_match_pct']):>7} "
              f"{str(s['chapter_hit_pct']):>7} {s['empty_pct']:>8} "
              f"{str(s['mean_top1_score']):>7} {s['mean_latency_s']:>6} {s['cross_lingual_n']:>6}")
    print("=" * 86)
    print("top1% is measured only against same-language gold (native = reference row).")
    print("ch@k% = gold chapter found in top-k -- the language-agnostic signal.")


def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="R4: retrieval by query style (native/romanized/code-switched).")
    ap.add_argument("--queries", default=str(here / "queries" / "romanized_queries.jsonl"))
    ap.add_argument("--dataset-dir", default=str(here.parent / "knowledge_base"))
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--report", default=str(here / "reports" / "r4_retrieval.json"))
    args = ap.parse_args()

    queries = load_queries(Path(args.queries))
    print(f"{len(queries)} queries across {len({q['concept'] for q in queries})} concepts")

    # Heavy imports only when actually running -- the local logic test above
    # must not need torch.
    from story_mvp.rag_engine import RetrievalService

    engine = RetrievalService(dataset_dir=args.dataset_dir)
    print("building gold from native-script queries ...")
    gold = build_gold(queries, engine, args.top_k)
    print(f"gold established for {len(gold)} (concept, language) pairs")

    rows = evaluate(queries, gold, engine, args.top_k)
    summary = summarize(rows)
    print_table(summary)

    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
