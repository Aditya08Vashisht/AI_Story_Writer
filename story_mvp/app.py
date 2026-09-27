"""Flask app for the StoryTutor-MM MVP."""

from __future__ import annotations

import os
import sys
from datetime import datetime

from dotenv import load_dotenv

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

from flask import Flask, Response, jsonify, render_template, request

from story_mvp.generator import generate_story_piece, generate_story_piece_ai
from story_mvp.rag_chat import answer_question, stream_answer

RAG_ENGINE = None
LLM_CLIENT = None

# Building the index at import time makes the module unimportable in under a
# minute, which blocks route tests and UI iteration. This escape hatch lets
# either run without loading an embedding model.
if os.environ.get("STORYTUTOR_SKIP_AI_INIT", "").lower() in {"1", "true", "yes"}:
    print("STORYTUTOR_SKIP_AI_INIT set - serving UI only, retrieval disabled.")
    AI_ENABLED = False
else:
    try:
        from story_mvp.rag_engine import RetrievalService
        from story_mvp.llm_client import StoryLLM

        DATASET_DIR = os.path.join(_PROJECT_ROOT, "knowledge_base")
        RAG_ENGINE = RetrievalService(dataset_dir=DATASET_DIR)
        LLM_CLIENT = StoryLLM()
        AI_ENABLED = True
    except Exception as e:
        print(f"Warning: Failed to initialize AI components - {e}")
        AI_ENABLED = False


app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)
app.config["SECRET_KEY"] = "story-tutor-mm-mvp-2026"


DEMO_PROMPTS = [
    {
        "label": "Energy mystery",
        "mode": "hook",
        "genre": "thriller",
        "tone": "suspenseful",
        "language": "hinglish",
        "idea": "Help a student understand how energy transfers when a spoon warms in hot water and a speaker vibrates rice grains.",
        "characters": "Asha, Kabir",
        "length": "medium",
        "class_level": "6",
        "subject": "science",
        "chapter": "energy transfer",
        "output_type": "video_demo_plan",
        "difficulty": "medium",
    },
    {
        "label": "Pond ecosystem",
        "mode": "expand",
        "genre": "family drama",
        "tone": "emotional",
        "language": "hindi",
        "idea": "Teach how sunlight, algae, insects, fish, oxygen, and decomposers connect in a pond ecosystem.",
        "characters": "Anaya, Dev",
        "length": "medium",
        "class_level": "6",
        "subject": "science",
        "chapter": "ecosystems",
        "output_type": "study_story",
        "difficulty": "medium",
    },
    {
        "label": "Force misconception",
        "mode": "continue",
        "genre": "comedy",
        "tone": "cinematic",
        "language": "english",
        "idea": "A learner thinks a moving object always needs a forward force. Build a story-based explanation that corrects the misconception.",
        "characters": "Aarav, Meera",
        "length": "long",
        "class_level": "6",
        "subject": "science",
        "chapter": "forces",
        "output_type": "video_demo_plan",
        "difficulty": "medium",
    },
]


@app.route("/")
@app.route("/chat")
def index():
    """The product is a curriculum RAG chatbot. The old story-generator UI
    has been removed -- it framed a tutor as a fiction engine.

    Served as a static file, not a Jinja template: the page has no
    server-side variables, and React inline styles use {{...}} -- which is
    Jinja's own expression syntax, so rendering it as a template makes Jinja
    try to evaluate JSX.
    """
    return app.send_static_file("chat.html")


VIDEO_DIR = os.path.join(_PROJECT_ROOT, "outputs", "videos")


@app.route("/videos")
def videos_page():
    """Watch generated videos in the browser over the same forwarded port as
    the chat, instead of downloading each MP4 from the cluster by hand."""
    return app.send_static_file("videos.html")


@app.route("/api/videos")
def list_videos():
    """Every rendered MP4, newest first, with its sidecar audit data."""
    items = []
    if os.path.isdir(VIDEO_DIR):
        names = [n for n in os.listdir(VIDEO_DIR) if n.endswith(".mp4")]
        names.sort(key=lambda n: os.path.getmtime(os.path.join(VIDEO_DIR, n)), reverse=True)
        for name in names:
            meta = {}
            side = os.path.join(VIDEO_DIR, name[:-4] + ".json")
            if os.path.exists(side):
                try:
                    with open(side, encoding="utf-8") as f:
                        meta = json.load(f)
                except Exception:  # noqa: BLE001 - a bad sidecar must not hide the video
                    meta = {}
            items.append({
                "file": name,
                # Versioned by modification time: a re-rendered video keeps its
                # name, and without this the browser replays its cached copy
                # under the new title and audit data.
                "url": f"/videos/file/{name}?v={int(os.path.getmtime(os.path.join(VIDEO_DIR, name)))}",
                "question": meta.get("question"),
                "title": meta.get("title"),
                "language": meta.get("language"),
                "total_words": meta.get("total_words"),
                "narrated": meta.get("narrated"),
                "verified": (meta.get("grounding") or {}).get("verified"),
                "illustrations": sum(1 for i in (meta.get("illustrations") or []) if i.get("generated")),
                "model_provider": meta.get("model_provider"),
                "sources": meta.get("sources", []),
                "scenes": [{"key": s.get("key"), "narration": s.get("narration")}
                           for s in meta.get("scenes", [])],
            })
    return jsonify(items)


@app.route("/videos/file/<name>")
def video_file(name):
    """Serve one file from outputs/videos. Basenames only -- no traversal."""
    from flask import abort, send_from_directory

    if name != os.path.basename(name) or not name.endswith((".mp4", ".json")):
        abort(404)
    return send_from_directory(VIDEO_DIR, name)


@app.route("/api/tts", methods=["POST"])
def tts():
    """Read text aloud via the voice server (scripts/tts_server.py).

    503 when the server is not running; the chat page then falls back to the
    browser's own speech voice, so the button always does something.
    """
    import urllib.error
    import urllib.request

    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", "")).strip()[:4000]
    language = payload.get("language", "english")
    if language not in {"english", "hindi", "marathi"}:
        language = "english"
    if not text:
        return jsonify({"error": "empty text"}), 400

    url = os.environ.get("STORYTUTOR_TTS_URL", "http://127.0.0.1:5060").rstrip("/") + "/tts"
    body = json.dumps({"text": text, "language": language}).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return Response(r.read(), mimetype="audio/wav",
                            headers={"Cache-Control": "private, max-age=86400"})
    except (urllib.error.URLError, OSError) as exc:
        return jsonify({"error": f"voice server unavailable: {exc}"}), 503


# ------------------------------------------------------------ video from chat
#
# The chat's "Make video" button. One worker thread, one video at a time: a
# video uses the GPU for the script, three pictures and the voice, and two at
# once would only make both slower. The image model is loaded on the first
# video and kept, so later videos skip the load.

import queue as _queue
import threading as _threading
import uuid as _uuid

VIDEO_JOBS: dict = {}
_VIDEO_QUEUE: "_queue.Queue" = _queue.Queue()
_VIDEO_STATE = {"worker": None, "illustrator": None}
_VIDEO_LOCK = _threading.Lock()
_SOURCE_KEYS = ("marker", "class_level", "subject", "language", "chapter", "page", "source_file")


def _video_worker():
    from story_mvp.video.pipeline import load_illustrator, make_video
    from story_mvp.video.tts import get_tts

    while True:
        job_id, concept, rag = _VIDEO_QUEUE.get()
        job = VIDEO_JOBS[job_id]

        def progress(step, fraction, job=job):
            job.update(step=step, progress=round(float(fraction), 2))

        job.update(state="running", step="Starting", started=datetime.now().isoformat(timespec="seconds"))
        try:
            if _VIDEO_STATE["illustrator"] is None:
                progress("Loading the picture model (first video only)", 0.02)
                gen, reason = load_illustrator()
                _VIDEO_STATE["illustrator"] = gen
                if gen is None:
                    job["note"] = f"No pictures: {reason}"
            tts = get_tts()
            if not tts.available:
                job["note"] = (job.get("note", "") + " No voice: the voice server is not running.").strip()
                tts = None
            result = make_video(concept, RAG_ENGINE, LLM_CLIENT, tts, VIDEO_DIR,
                                illustrator=_VIDEO_STATE["illustrator"], rag=rag, progress=progress)
            if result.get("status") != "ok":
                job.update(state="error", error=result.get("error") or result.get("status"))
            else:
                path = os.path.join(VIDEO_DIR, result["file"])
                job.update(state="done", progress=1.0, step="Done", result=result,
                           url=f"/videos/file/{result['file']}?v={int(os.path.getmtime(path))}")
        except Exception as exc:  # noqa: BLE001 - report it on the page, keep the worker alive
            job.update(state="error", error=str(exc)[:400])
        finally:
            _VIDEO_QUEUE.task_done()


@app.route("/api/video", methods=["POST"])
def start_video():
    """Make a 30-second video of an answer the student is looking at."""
    if not AI_ENABLED:
        return jsonify({"error": "videos need the AI models; the app is running UI-only"}), 503
    p = request.get_json(silent=True) or {}
    question = str(p.get("question", "")).strip()[:500]
    answer = str(p.get("answer", "")).strip()[:6000]
    sources = [{k: s.get(k) for k in _SOURCE_KEYS} for s in (p.get("sources") or [])[:10]
               if isinstance(s, dict)]
    if not question or not answer:
        return jsonify({"error": "question and answer are required"}), 400
    if not sources:
        return jsonify({"error": "This answer has no textbook sources, so no video is made for it."}), 400

    language = p.get("language") if p.get("language") in {"english", "hindi", "marathi"} else "english"
    concept = {"question": question, "language": language,
               "class_level": str(p.get("class_level") or ""), "subject": str(p.get("subject") or "")}
    rag = {"answer": answer, "sources": sources, "grounded": True,
           "retrieval_method": p.get("retrieval_method")}

    job_id = _uuid.uuid4().hex[:12]
    VIDEO_JOBS[job_id] = {"id": job_id, "state": "queued", "step": "Waiting for the video maker",
                          "progress": 0.0, "question": question, "queue_position": _VIDEO_QUEUE.qsize()}
    for old in list(VIDEO_JOBS)[:-50]:          # keep the table small
        if VIDEO_JOBS[old]["state"] in {"done", "error"}:
            VIDEO_JOBS.pop(old, None)
    _VIDEO_QUEUE.put((job_id, concept, rag))
    with _VIDEO_LOCK:
        if _VIDEO_STATE["worker"] is None or not _VIDEO_STATE["worker"].is_alive():
            _VIDEO_STATE["worker"] = _threading.Thread(target=_video_worker, daemon=True)
            _VIDEO_STATE["worker"].start()
    return jsonify(VIDEO_JOBS[job_id]), 202


@app.route("/api/video/<job_id>")
def video_status(job_id):
    job = VIDEO_JOBS.get(job_id)
    return (jsonify(job), 200) if job else (jsonify({"error": "unknown job"}), 404)


@app.route("/api/options")
def options():
    return jsonify(
        {
            "modes": MODE_TITLES,
            "genres": list(GENRES.keys()),
            "tones": list(TONES.keys()),
            "languages": ["english", "hindi", "hinglish", "marathi"],
            "lengths": ["short", "medium", "long"],
            "class_levels": ["6", "7", "8"],
            "subjects": ["science", "social_science"],
            "output_types": ["study_story", "video_demo_plan", "explanation", "quiz", "story_video_plan"],
            "difficulties": ["easy", "medium", "hard"],
        }
    )


@app.route("/api/generate", methods=["POST"])
def generate():
    payload = request.get_json(silent=True) or {}
    
    if AI_ENABLED:
        try:
            result = generate_story_piece_ai(payload, RAG_ENGINE, LLM_CLIENT)
        except Exception as e:
            print(f"AI generation failed, falling back: {e}")
            result = generate_story_piece(payload)
    else:
        result = generate_story_piece(payload)
        
    result["generated_at"] = datetime.now().isoformat(timespec="seconds")
    return jsonify(result)


@app.route("/api/chat", methods=["POST"])
def chat_api():
    """Question in, grounded answer + citations out. No story parameters."""
    payload = request.get_json(silent=True) or {}
    if not AI_ENABLED:
        return jsonify({
            "answer": "The retrieval engine is not available. Check the server log.",
            "sources": [], "grounded": False, "model_provider": "none",
        }), 503
    result = answer_question(payload, RAG_ENGINE, LLM_CLIENT)
    result["generated_at"] = datetime.now().isoformat(timespec="seconds")
    return jsonify(result)


@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    """Server-sent events, so the first words appear in under a second.

    Total generation time is the same as /api/chat; what changes is that the
    student reads while the model is still writing instead of watching a
    spinner. Sources are sent first, before any token, so citations can render
    immediately.
    """
    payload = request.get_json(silent=True) or {}
    if not AI_ENABLED:
        return jsonify({"error": "retrieval engine unavailable"}), 503

    def events():
        try:
            for event in stream_answer(payload, RAG_ENGINE, LLM_CLIENT):
                yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
        except Exception as exc:  # noqa: BLE001
            yield "data: " + json.dumps({"type": "error", "message": str(exc)}) + "\n\n"

    return Response(
        events(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/demo")
def demo_prompts():
    return jsonify(DEMO_PROMPTS)


@app.route("/api/health")
def health():
    return jsonify(
        {
            "status": "online",
            "engine": "free_first_hybrid_rag" if AI_ENABLED else "local_story_tutor_mvp",
            "embedding_model": getattr(RAG_ENGINE, "model_name", None) if AI_ENABLED else None,
            "model_provider": getattr(LLM_CLIENT, "last_provider", "not_used") if AI_ENABLED else "deterministic",
            "ready_for_model_swap": True,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
    )


def run_story_studio(host: str = "127.0.0.1", port: int = 5050, debug: bool = False):
    # use_reloader=False prevents double-initialization of the heavy RAG engine
    app.run(host=host, port=port, debug=debug, use_reloader=False)


if __name__ == "__main__":
    # Debug stays off unless asked for: Flask's debugger runs arbitrary code,
    # and on Sol this page is reachable through the VS Code tunnel.
    run_story_studio(
        port=int(os.environ.get("STORYTUTOR_PORT", 5050)),
        debug=os.environ.get("STORYTUTOR_DEBUG", "").lower() in {"1", "true", "yes"},
    )
