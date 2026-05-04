"""End-to-end integration tests for the full AI pipeline (RAG + LLM + Generator)."""

import os
import shutil
import pytest

from dotenv import load_dotenv

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

from story_mvp.generator import generate_story_piece, generate_story_piece_ai
from story_mvp.rag_engine import StoryRAG
from story_mvp.llm_client import StoryLLM

DATASET_DIR = os.path.join(_PROJECT_ROOT, "story_datasets")
TEST_INDEX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_test_integration_index")

# Required fields in any generate response (same for old and AI engines)
REQUIRED_FIELDS = {"title", "mode", "mode_label", "genre", "tone", "language", "output", "pitch", "style_notes", "story_bible", "safety"}


@pytest.fixture(scope="module")
def ai_components():
    """Initialize RAG + LLM. Skip all if API key is missing."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        pytest.skip("GROQ_API_KEY not set — skipping integration tests")
    rag = StoryRAG(dataset_dir=DATASET_DIR, index_dir=TEST_INDEX_DIR)
    llm = StoryLLM(api_key=api_key)
    yield rag, llm
    if os.path.exists(TEST_INDEX_DIR):
        shutil.rmtree(TEST_INDEX_DIR)


def test_ai_hook_has_all_required_fields(ai_components):
    """AI-generated hook should have all the same fields as the deterministic engine."""
    rag, llm = ai_components
    payload = {
        "mode": "hook",
        "genre": "thriller",
        "tone": "suspenseful",
        "language": "english",
        "idea": "A librarian finds a book that writes itself, adding a new chapter every night.",
        "characters": "Maya, Prof. Sen",
        "length": "medium",
    }
    result = generate_story_piece_ai(payload, rag, llm)
    missing = REQUIRED_FIELDS - set(result.keys())
    assert not missing, f"AI response missing fields: {missing}"
    print(f"  All {len(REQUIRED_FIELDS)} required fields present")


def test_ai_and_fallback_produce_same_shape(ai_components):
    """AI result and fallback result should have the exact same set of keys."""
    rag, llm = ai_components
    payload = {
        "mode": "expand",
        "genre": "romance",
        "tone": "emotional",
        "language": "hinglish",
        "idea": "Two childhood friends meet at a train station after 10 years.",
        "characters": "Priya, Arjun",
        "length": "short",
    }
    ai_result = generate_story_piece_ai(payload, rag, llm)
    fallback_result = generate_story_piece(payload)

    assert set(ai_result.keys()) == set(fallback_result.keys()), (
        f"Key mismatch:\n  AI keys: {sorted(ai_result.keys())}\n  Fallback keys: {sorted(fallback_result.keys())}"
    )
    print("  AI and fallback response shapes match perfectly")


def test_ai_output_is_not_deterministic_template(ai_components):
    """AI output should NOT contain the old deterministic template phrases."""
    rag, llm = ai_components
    payload = {
        "mode": "hook",
        "genre": "fantasy",
        "tone": "cinematic",
        "language": "english",
        "idea": "A mapmaker discovers that erased cities reappear on his maps under moonlight.",
        "characters": "Kael, Lyra",
        "length": "medium",
    }
    result = generate_story_piece_ai(payload, rag, llm)
    output = result["output"]

    # These phrases are from the old deterministic engine templates
    old_templates = [
        "until the same warning appears inside tomorrow's audio script",
        "as if the story had found a body and started breathing",
        "the old version of the plan is useless now",
    ]
    for phrase in old_templates:
        assert phrase not in output, f"AI output contains old template phrase: '{phrase}'"
    print(f"  AI output is unique ({len(output)} chars), not a template copy")


def test_ai_safety_check_runs_on_output(ai_components):
    """Safety check should be applied to the AI-generated output."""
    rag, llm = ai_components
    payload = {
        "mode": "continue",
        "genre": "comedy",
        "tone": "witty",
        "language": "english",
        "idea": "A chef accidentally enters a cooking contest in a language he doesn't speak.",
        "characters": "Bunty, Chef Marco",
        "length": "medium",
    }
    result = generate_story_piece_ai(payload, rag, llm)
    assert "safety" in result
    assert result["safety"]["status"] in ("clear", "review")
    assert isinstance(result["safety"]["flags"], list)
    print(f"  Safety status: {result['safety']['status']}")


def test_ai_mode_label_matches_mode(ai_components):
    """mode_label should correctly correspond to the mode."""
    rag, llm = ai_components
    expected_labels = {
        "hook": "Hook Generator",
        "expand": "Scene Expander",
        "continue": "Story Continuation",
    }
    for mode, expected_label in expected_labels.items():
        payload = {
            "mode": mode,
            "genre": "thriller",
            "tone": "cinematic",
            "language": "english",
            "idea": "A quick test idea.",
            "characters": "Test",
            "length": "short",
        }
        result = generate_story_piece_ai(payload, rag, llm)
        assert result["mode"] == mode
        assert result["mode_label"] == expected_label
        print(f"  {mode} -> {result['mode_label']} OK")
