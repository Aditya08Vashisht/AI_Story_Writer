"""Extract a golden evaluation set from NCERT chapter-end exercises.

This is a heuristic v1 extractor (regex-based), not an NLP pipeline. It finds
chunks the ingester's exercise markers flagged, splits them on numbered-item
boundaries, and treats each resulting segment as one candidate question.

Per plan.md Sec.8.1: human-verify at least 50 items (review_sample_50.jsonl)
before trusting any eval number this set produces. `reference_answer` and
`question_type` are intentionally left for that review pass to fill in.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

EXERCISE_MARKERS = {
    "english": [r"\bExercises?\b", r"\bQuestions?\b", r"Let us enhance our learning", r"\bAnswer the following"],
    "hindi": [r"अभ्यास", r"प्रश्न", r"आइए.*करें"],
    "marathi": [r"स्वाध्याय", r"प्रश्न", r"अभ्यास"],
}

# Numbered-item boundary: "1. ", "2) ", etc.
_ITEM_SPLIT = re.compile(r"(?:(?<=[\s.])|^)([1-9][0-9]?)[.\)]\s+")
_LEADING_NUMBER = re.compile(r"^[1-9][0-9]?[.\)]\s+")

_MIN_LEN = 20
_MAX_LEN = 400
_MIN_ALPHA_RUN = re.compile(r"[A-Za-zऀ-ॿ]{3,}")


def has_exercise_marker(text: str, language: str) -> bool:
    patterns = EXERCISE_MARKERS.get(language, [])
    return any(re.search(p, text) for p in patterns)


def split_candidates(text: str) -> List[str]:
    """Split an exercise-flagged chunk into candidate question segments."""
    positions = [m.start(1) for m in _ITEM_SPLIT.finditer(text)]
    if len(positions) < 2:
        # No clean numbered list in this chunk; fall back to '?'-terminated
        # sentence splitting for prose-style question chunks.
        return [s.strip() + "?" for s in text.split("?") if s.strip()]

    segments = []
    for i, start in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(text)
        seg = _LEADING_NUMBER.sub("", text[start:end])
        segments.append(seg.strip())
    return segments


def clean_candidate(seg: str) -> Optional[str]:
    seg = re.sub(r"\s+", " ", seg).strip()
    if not (_MIN_LEN <= len(seg) <= _MAX_LEN):
        return None
    if not _MIN_ALPHA_RUN.search(seg):
        return None
    return seg


def extract_from_chunk(chunk: Dict) -> List[str]:
    language = chunk.get("language", "")
    text = chunk.get("text", "")
    if not has_exercise_marker(text, language):
        return []
    out = []
    for seg in split_candidates(text):
        cleaned = clean_candidate(seg)
        if cleaned:
            out.append(cleaned)
    return out


def build_records(chunks: List[Dict]) -> List[Dict]:
    records = []
    seen = set()
    for chunk in chunks:
        for q in extract_from_chunk(chunk):
            key = (chunk.get("language"), q.lower())
            if key in seen:
                continue
            seen.add(key)
            records.append(
                {
                    "question": q,
                    "language": chunk.get("language"),
                    "class_level": chunk.get("class_level"),
                    "subject": chunk.get("subject"),
                    "gold_chapter": chunk.get("chapter"),
                    "gold_chunk_ids": [chunk.get("id")],
                    "source_file": chunk.get("source_file"),
                    "page_start": chunk.get("page_start"),
                    "reference_answer": None,
                    "question_type": "unclassified",
                    "needs_review": True,
                }
            )
    return records


def stratified_sample(records: List[Dict], per_cell: int, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    cells: Dict[Tuple, List[Dict]] = defaultdict(list)
    for r in records:
        cells[(r["class_level"], r["subject"], r["language"])].append(r)

    sampled = []
    for cell in sorted(cells):
        items = cells[cell]
        rng.shuffle(items)
        sampled.extend(items[:per_cell])
    return sampled


def write_jsonl(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a golden eval set from NCERT exercise chunks.")
    parser.add_argument("--chunks", default="knowledge_base/processed/curriculum_chunks.json")
    parser.add_argument("--output", default="eval/golden/golden_set.jsonl")
    parser.add_argument("--review-sample", default="eval/golden/review_sample_50.jsonl")
    parser.add_argument("--per-cell", type=int, default=17)
    parser.add_argument("--review-sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    chunks = json.loads(Path(args.chunks).read_text(encoding="utf-8"))
    all_records = build_records(chunks)
    print(f"candidate questions extracted: {len(all_records)}")

    sampled = stratified_sample(all_records, args.per_cell, args.seed)
    n_cells = len({(r["class_level"], r["subject"], r["language"]) for r in sampled})
    print(f"stratified sample: {len(sampled)} across {n_cells} cells")

    for i, r in enumerate(sampled, 1):
        r["id"] = f"golden_{i:04d}"

    out_path = Path(args.output)
    write_jsonl(out_path, sampled)
    print(f"wrote {len(sampled)} records -> {out_path}")

    rng = random.Random(args.seed)
    review_pool = sampled[:]
    rng.shuffle(review_pool)
    review_sample = review_pool[: args.review_sample_size]
    review_path = Path(args.review_sample)
    write_jsonl(review_path, review_sample)
    print(f"wrote {len(review_sample)} review-sample records -> {review_path}")

    print("\nper-cell counts (class_level, subject, language):")
    cells = defaultdict(int)
    for r in sampled:
        cells[(r["class_level"], r["subject"], r["language"])] += 1
    for cell in sorted(cells):
        print(f"  {cell}: {cells[cell]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
