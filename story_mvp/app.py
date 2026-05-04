"""Flask app for the Story Studio MVP."""

from __future__ import annotations

import os
import sys
from datetime import datetime

from dotenv import load_dotenv

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify, render_template, request

from story_mvp.generator import GENRES, MODE_TITLES, TONES, generate_story_piece, generate_story_piece_ai

try:
    from story_mvp.rag_engine import StoryRAG
    from story_mvp.llm_client import StoryLLM
    
    DATASET_DIR = os.path.join(_PROJECT_ROOT, "story_datasets")
    RAG_ENGINE = StoryRAG(dataset_dir=DATASET_DIR)
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
app.config["SECRET_KEY"] = "story-studio-mvp-2026"


DEMO_PROMPTS = [
    {
        "label": "Audio thriller",
        "mode": "hook",
        "genre": "thriller",
        "tone": "suspenseful",
        "language": "hinglish",
        "idea": "A struggling podcast writer receives voice notes from a missing listener before each episode releases.",
        "characters": "Rhea, Kabir",
        "length": "medium",
    },
    {
        "label": "Family drama",
        "mode": "expand",
        "genre": "family drama",
        "tone": "emotional",
        "language": "hindi",
        "idea": "A daughter returns home for her brother's wedding and finds her late mother's diary hidden in the prayer room.",
        "characters": "Anaya, Dev",
        "length": "medium",
    },
    {
        "label": "Romance cliffhanger",
        "mode": "continue",
        "genre": "romance",
        "tone": "cinematic",
        "language": "english",
        "idea": "Two former radio hosts meet again during a city blackout, with one final unsent confession between them.",
        "characters": "Aarav, Meera",
        "length": "long",
    },
]


@app.route("/")
def index():
    return render_template(
        "index.html",
        genres=list(GENRES.keys()),
        tones=list(TONES.keys()),
        modes=MODE_TITLES,
    )


@app.route("/api/options")
def options():
    return jsonify(
        {
            "modes": MODE_TITLES,
            "genres": list(GENRES.keys()),
            "tones": list(TONES.keys()),
            "languages": ["english", "hindi", "hinglish"],
            "lengths": ["short", "medium", "long"],
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


@app.route("/api/demo")
def demo_prompts():
    return jsonify(DEMO_PROMPTS)


@app.route("/api/health")
def health():
    return jsonify(
        {
            "status": "online",
            "engine": "groq_rag" if AI_ENABLED else "local_story_mvp",
            "ready_for_model_swap": True,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
    )


def run_story_studio(host: str = "127.0.0.1", port: int = 5050, debug: bool = False):
    # use_reloader=False prevents double-initialization of the heavy RAG engine
    app.run(host=host, port=port, debug=debug, use_reloader=False)


if __name__ == "__main__":
    run_story_studio(debug=True)
