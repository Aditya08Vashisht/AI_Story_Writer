"""Voice server for StoryTutor: text in, WAV out.

Runs in its OWN Python environment (see scripts/sol.sh setup-tts), because
parler-tts pins an old transformers that would break the main environment.

  GET  /health                      -> {"ok": true, "engine": "parler", ...}
  POST /tts  {"text", "language"}   -> audio/wav

Both the chat page (via the app's /api/tts) and make_video.py use it.
Standard library HTTP only, so the voice environment needs nothing extra.
Every clip is cached on disk, so a repeated sentence costs nothing.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from story_mvp.video.tts import cache_key, local_engine  # noqa: E402

LANGS = {"english", "hindi", "marathi"}
MAX_CHARS = 4000

ENGINE = None
LOCK = threading.Lock()          # one GPU model: one synthesis at a time
CACHE = Path(os.environ.get("STORYTUTOR_TTS_CACHE", "outputs/tts_cache"))


class Handler(BaseHTTPRequestHandler):
    server_version = "StoryTutorTTS/1.0"

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path.rstrip("/") == "/health":
            ok = ENGINE is not None and ENGINE.failure is None
            self._json(200 if ok else 503, {
                "ok": ok, "engine": getattr(ENGINE, "name", None),
                "model": getattr(ENGINE, "model_id", None),
                "error": getattr(ENGINE, "failure", None)})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        if self.path.rstrip("/") != "/tts":
            return self._json(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        except Exception:  # noqa: BLE001
            return self._json(400, {"error": "body must be JSON"})

        text = str(req.get("text", "")).strip()[:MAX_CHARS]
        language = req.get("language", "english")
        language = language if language in LANGS else "english"
        if not text:
            return self._json(400, {"error": "empty text"})

        path = CACHE / f"{cache_key(text, language, ENGINE.name)}.wav"
        if not path.exists():
            t0 = time.time()
            try:
                import soundfile as sf

                with LOCK:
                    audio = ENGINE.speak_array(text, language)
                buf = io.BytesIO()
                sf.write(buf, audio, ENGINE.sample_rate, format="WAV")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(buf.getvalue())
            except Exception as exc:  # noqa: BLE001
                return self._json(500, {"error": str(exc)})
            print(f"tts {language:<8} {len(text):>4} chars  {time.time() - t0:5.1f}s")

        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):  # the per-request log above is enough
        pass


def main() -> int:
    global ENGINE
    ap = argparse.ArgumentParser(description="StoryTutor voice server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("STORYTUTOR_TTS_PORT", 5060)))
    args = ap.parse_args()

    ENGINE = local_engine()
    if not ENGINE.available:
        print(f"no voice could load: {ENGINE.failure}")
        return 1
    # Warm up once so the first real request is not the slow one.
    try:
        ENGINE.speak_array("Hello.", "english")
    except Exception as exc:  # noqa: BLE001
        print(f"warm-up failed: {exc}")
    print(f"voice server ready on http://{args.host}:{args.port} · engine={ENGINE.name}")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
