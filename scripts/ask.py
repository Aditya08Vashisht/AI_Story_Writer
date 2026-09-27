"""Ask the running chatbot from the command line.

  python scripts/ask.py "Why does a metal spoon get hot in hot water?"
  python scripts/ask.py "पौधे अपना भोजन कैसे बनाते हैं?" --language hindi --class-level 7
  python scripts/ask.py "What is a food chain?" --speak      # also save the answer as speech

Talks to the web app (bash scripts/sol.sh up), so answers come back in
seconds instead of reloading every model per question.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEVA = re.compile(r"[ऀ-ॿ]")


def post(url: str, payload: dict, timeout: float = 300):
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def main() -> int:
    ap = argparse.ArgumentParser(description="Ask the StoryTutor chatbot.")
    ap.add_argument("question")
    ap.add_argument("--language", choices=["english", "hindi", "marathi"],
                    help="default: hindi/marathi if the question is in Devanagari, else english")
    ap.add_argument("--class-level", default="")
    ap.add_argument("--subject", default="", choices=["", "science", "social_science"])
    ap.add_argument("--speak", action="store_true", help="also save the answer as a WAV")
    ap.add_argument("--url", default=os.environ.get("STORYTUTOR_URL", "http://127.0.0.1:5050"))
    args = ap.parse_args()

    language = args.language or ("hindi" if DEVA.search(args.question) else "english")
    payload = {"question": args.question, "language": language,
               "class_level": args.class_level, "subject": args.subject}

    t0 = time.time()
    try:
        with post(f"{args.url}/api/chat", payload) as r:
            d = json.loads(r.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        print(f"No chatbot at {args.url} ({exc}). Start it with:  bash scripts/sol.sh up")
        return 2

    print(f"\n{d.get('answer', '').strip()}\n")
    print(f"-- {len(d.get('sources') or [])} source(s) · grounded={d.get('grounded')} · "
          f"{d.get('model_provider')} · {time.time() - t0:.1f}s")
    for s in (d.get("sources") or [])[:5]:
        where = " ".join(x for x in [f"Class {s.get('class_level')}" if s.get("class_level") else "",
                                     s.get("subject") or "",
                                     f"ch {s.get('chapter')}" if s.get("chapter") else "",
                                     f"p{s.get('page')}" if s.get("page") else ""] if x)
        print(f"   [{s.get('marker', '?')}] {where}  {s.get('source_file') or ''}")

    if args.speak and d.get("answer"):
        out = Path("outputs/ask_audio") / f"answer_{int(time.time())}.wav"
        try:
            t1 = time.time()
            with post(f"{args.url}/api/tts", {"text": d["answer"], "language": language}) as r:
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(r.read())
            print(f"\nspeech: {out}  ({time.time() - t1:.1f}s to synthesise)")
        except urllib.error.HTTPError as exc:
            print(f"\nspeech unavailable ({exc.code}): is the voice server up?  bash scripts/sol.sh status")
    return 0


if __name__ == "__main__":
    sys.exit(main())
