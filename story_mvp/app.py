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
    run_story_studio(debug=True)
