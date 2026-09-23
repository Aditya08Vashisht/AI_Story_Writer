"""Measure Devanagari extraction damage per source PDF.

Why this exists: "the data is bad" is not actionable. The NCERT files are real
text-layer PDFs, not scans, so extraction *should* work. This quantifies what
is actually broken, per file, so the fix can be targeted rather than guessed.

How Devanagari extraction breaks in a text-layer PDF:
  A PDF stores glyph indices, not characters. A ToUnicode CMap maps them back
  to Unicode. For Latin that mapping is trivially 1:1. For Devanagari it often
  is not, because the renderer reorders matras (the vowel sign ि is drawn
  BEFORE its consonant but encoded AFTER it) and fuses consonants into
  conjunct ligatures (क + ् + ष -> क्ष as ONE glyph). When the CMap is
  incomplete or wrong, extraction returns glyphs in visual order with broken
  clusters, and inter-glyph kerning gets misread as word spaces.

Detection signals (all are structurally impossible in valid Devanagari):
  1. orphan matra   - a vowel sign with no consonant before it
  2. orphan virama  - a halant not followed by a consonant
  3. singleton runs - one-character Devanagari tokens, which real words rarely are

A high score means the text layer is unusable and the file needs a different
extractor or an OCR fallback.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from story_mvp.deva_quality import score_text  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Quantify Devanagari extraction damage per PDF.")
    parser.add_argument("--chunks", default="knowledge_base/processed/curriculum_chunks.json")
    parser.add_argument("--report", default="eval/reports/corpus_damage.json")
    parser.add_argument("--worst", type=int, default=15)
    args = parser.parse_args()

    chunks = json.loads(Path(args.chunks).read_text(encoding="utf-8"))

    by_file = defaultdict(lambda: {"text": [], "language": "", "class_level": "", "subject": ""})
    for c in chunks:
        if c.get("language") not in {"hindi", "marathi"}:
            continue
        key = Path(str(c.get("source_file", ""))).name
        by_file[key]["text"].append(c.get("text", ""))
        by_file[key]["language"] = c.get("language", "")
        by_file[key]["class_level"] = c.get("class_level", "")
        by_file[key]["subject"] = c.get("subject", "")

    rows = []
    for fname, data in by_file.items():
        s = score_text(" ".join(data["text"]))
        if not s.get("scored"):
            continue
        rows.append({
            "file": fname,
            "language": data["language"],
            "class_level": data["class_level"],
            "subject": data["subject"],
            **s,
        })

    rows.sort(key=lambda r: r["damage"], reverse=True)

    by_lang = defaultdict(list)
    for r in rows:
        by_lang[r["language"]].append(r["damage"])

    print(f"scored {len(rows)} Devanagari PDFs\n")
    print("damage by language (lower is cleaner):")
    print(f"{'lang':<10} {'files':>6} {'median':>8} {'mean':>8} {'worst':>8} {'clean<10':>9}")
    print("-" * 56)
    for lang, scores in sorted(by_lang.items()):
        scores_sorted = sorted(scores)
        median = scores_sorted[len(scores_sorted) // 2]
        clean = sum(1 for s in scores if s < 10)
        print(f"{lang:<10} {len(scores):>6} {median:>8.1f} {sum(scores)/len(scores):>8.1f} "
              f"{max(scores):>8.1f} {clean:>9}")

    print(f"\nworst {args.worst} files:")
    print(f"{'damage':>8}  {'lang':<9} {'cls':<4} {'file'}")
    print("-" * 66)
    for r in rows[: args.worst]:
        print(f"{r['damage']:>8.1f}  {r['language']:<9} {r['class_level']:<4} {r['file']}")

    print(f"\nbest 5 (these extract fine - proof it is per-file, not per-language):")
    for r in rows[-5:]:
        print(f"{r['damage']:>8.1f}  {r['language']:<9} {r['class_level']:<4} {r['file']}")

    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"files": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
