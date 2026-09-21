"""Standalone FAISS index build for StoryTutor-MM.

Run this as a batch job before starting the app. Building at Flask import time
blocks the server for the whole embedding pass, which is unusable on a cluster.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from story_mvp.rag_engine import RetrievalService


SMOKE_QUERIES = [
    ("energy transfer when a spoon warms in hot water", {"class_level": "6", "subject": "science", "language": "english"}),
    ("वनस्पतींचे भाग", {"subject": "science", "language": "marathi"}),
    ("भारत के संसाधन", {"subject": "social_science", "language": "hindi"}),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the StoryTutor-MM FAISS index.")
    parser.add_argument("--dataset-dir", default="knowledge_base")
    parser.add_argument("--index-dir", default=None)
    parser.add_argument("--skip-smoke", action="store_true", help="Skip the post-build retrieval check.")
    args = parser.parse_args()

    start = time.time()
    engine = RetrievalService(dataset_dir=args.dataset_dir, index_dir=args.index_dir)
    elapsed = time.time() - start

    if engine.index is None:
        print("FAILED: no index was built. Is the corpus present?")
        return 1

    print("-" * 60)
    print(f"device       : {engine.device}")
    print(f"model        : {engine.model_name}")
    print(f"max_seq_len  : {engine.model.max_seq_length}")
    print(f"documents    : {len(engine.documents)}")
    print(f"vectors      : {engine.index.ntotal}")
    print(f"dimension    : {engine.index.d}")
    print(f"build seconds: {elapsed:.1f}")
    print("-" * 60)

    if engine.index.ntotal != len(engine.documents):
        print(f"FAILED: vector count {engine.index.ntotal} != document count {len(engine.documents)}")
        return 1

    if args.skip_smoke:
        return 0

    for query, filters in SMOKE_QUERIES:
        results = engine.retrieve(query=query, filters=filters, top_k=3)
        print(f"\nquery: {query[:60]}  filters={filters}")
        if not results:
            print("  (no results)")
            continue
        for doc in results:
            print(
                f"  {doc.get('retrieval_score', 0):.3f}  "
                f"class={doc.get('class_level')} {doc.get('subject')} {doc.get('language')} "
                f"ch={doc.get('chapter')} p{doc.get('page_start')}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
