"""R1: build the (claim, evidence, verdict) dataset from video sidecars.

Every video that StoryTutor renders writes a sidecar JSON next to the MP4
(`story_mvp/video/render.py::write_sidecar`) containing the scenes, the
diagram edges, the grounded answer it was checked against, and the
`grounding` report produced by `story_mvp/video/script.py::grounding_problems`.

Those sidecars ARE the research dataset for R1 (claim-level verification of
generated explainer videos). This script flattens them into one JSONL row per
claim:

    {"video", "question", "language", "claim_id", "kind", "scene", "text",
     "verdict", "why", "check_method", "verified", "evidence", "sources"}

Verdicts per claim:
    supported    claim id not in grounding.unsupported and grounding.verified
    unsupported  claim id in grounding.unsupported (with the reason `why`)
    unchecked    sidecar has no grounding report (older video)

The claim extraction intentionally mirrors `_claims()` in
`story_mvp/video/script.py` with stdlib only, so this runs anywhere (no torch,
no model server) — including a laptop reading sidecars copied off Sol.

Usage:
    python research/aggregate_video_claims.py
    python research/aggregate_video_claims.py --videos-dir outputs/videos \
        --out research/reports/video_claims.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Must match story_mvp/video/script.py: title is a label, check is a question.
FACT_SCENES = ("idea", "diagram")


def claims_of(side: dict) -> list[dict]:
    """Flatten one sidecar into claim rows (mirrors script.py::_claims)."""
    out: list[dict] = []
    for scene in side.get("scenes") or []:
        if scene.get("key") in FACT_SCENES and (scene.get("narration") or "").strip():
            out.append({"id": f"C{len(out) + 1}", "kind": "scene",
                        "scene": scene.get("key"), "text": scene["narration"].strip()})
    for edge in side.get("diagram_edges") or []:
        if len(edge) == 2:
            a, b = str(edge[0]), str(edge[1])
            out.append({"id": f"D{len(out) + 1}", "kind": "arrow", "scene": "diagram",
                        "text": f"{a} -> {b}", "edge": f"{a} -> {b}"})
    return out


def rows_from_sidecar(side: dict, video_name: str) -> list[dict]:
    grounding = side.get("grounding") or {}
    flagged = {p.get("id"): p.get("why", "") for p in grounding.get("unsupported") or []}
    method = grounding.get("method")
    verified = grounding.get("verified")
    rows = []
    for claim in claims_of(side):
        if claim["id"] in flagged:
            verdict, why = "unsupported", flagged[claim["id"]]
        elif verified is None or method in (None, "skipped"):
            verdict, why = "unchecked", ""
        else:
            verdict, why = "supported", ""
        rows.append({
            "video": video_name,
            "question": side.get("question"),
            "language": side.get("language"),
            "class_level": side.get("class_level"),
            "subject": side.get("subject"),
            "claim_id": claim["id"],
            "kind": claim["kind"],
            "scene": claim.get("scene"),
            "text": claim["text"],
            "verdict": verdict,
            "why": why,
            "check_method": method,
            "verified": verified,
            "evidence": side.get("rag_answer"),
            "sources": [{"page": s.get("page"), "chapter": s.get("chapter"),
                         "language": s.get("language")} for s in side.get("sources") or []],
        })
    return rows


def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--videos-dir", default=str(here.parent / "outputs" / "videos"),
                    help="directory holding rendered *.mp4 + sidecar *.json")
    ap.add_argument("--out", default=str(here / "reports" / "video_claims.jsonl"))
    args = ap.parse_args()

    videos_dir = Path(args.videos_dir)
    # batch_report.json lives here too and is not a sidecar; debug/ is a subdir.
    sidecars = sorted(p for p in videos_dir.glob("*.json")
                      if p.name != "batch_report.json")
    if not sidecars:
        print(f"no sidecars in {videos_dir} -- make a video first "
              f"(Make video in the chat, or scripts/make_video.py)", file=sys.stderr)
        return 1

    rows: list[dict] = []
    videos = Counter()
    for path in sidecars:
        try:
            side = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  skip {path.name}: {exc}", file=sys.stderr)
            continue
        got = rows_from_sidecar(side, path.stem)
        if got:
            rows.extend(got)
            videos[side.get("language") or "?"] += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    verdicts = Counter(r["verdict"] for r in rows)
    kinds = Counter(r["kind"] for r in rows)
    methods = Counter(str(r["check_method"]) for r in rows)
    print(f"{sum(videos.values())} videos ({', '.join(f'{k}:{v}' for k, v in sorted(videos.items()))})")
    print(f"{len(rows)} claims -> {out}")
    print(f"  verdicts : {dict(verdicts)}")
    print(f"  kinds    : {dict(kinds)}  (scene = narration line, arrow = diagram edge)")
    print(f"  methods  : {dict(methods)}")
    unchecked = verdicts.get("unchecked", 0)
    if unchecked:
        print(f"  note     : {unchecked} claims unchecked -- rerun those videos on a "
              f"stack with a working model to grow the verified set")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
