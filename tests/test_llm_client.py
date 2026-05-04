"""Tests for the LLM client (ChatGroq integration)."""

import os
import pytest

from dotenv import load_dotenv

# Load .env from project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

from story_mvp.generator import StoryRequest
from story_mvp.llm_client import StoryLLM


@pytest.fixture(scope="module")
def llm():
    """Create a StoryLLM instance. Skips all tests if no API key is set."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        pytest.skip("GROQ_API_KEY not set — skipping LLM tests")
    return StoryLLM(api_key=api_key)


SAMPLE_CONTEXT = """Example 1:
When Rhea discovers a locked phone buzzing under the floorboards, Kabir insists it is nothing, until the same warning appears inside tomorrow's audio script.
"""


def test_llm_generates_hook(llm):
    """LLM should generate a valid hook with all required JSON fields."""
    request = StoryRequest(
        mode="hook",
        idea="A taxi driver in Delhi starts getting calls from passengers who died in his cab years ago.",
        genre="thriller",
        tone="suspenseful",
        language="english",
        characters="Rajesh, Inspector Mehra",
        length="medium",
    )
    result = llm.generate(request, SAMPLE_CONTEXT)

    assert "output" in result, "Response missing 'output' field"
    assert "pitch" in result, "Response missing 'pitch' field"
    assert "story_bible" in result, "Response missing 'story_bible' field"
    assert "style_notes" in result, "Response missing 'style_notes' field"
    assert len(result["output"]) > 20, "Output text is too short"
    print(f"  Hook output: {result['output'][:150]}...")
    print(f"  Pitch: {result['pitch']}")


def test_llm_generates_scene_expansion(llm):
    """LLM should generate an expanded scene."""
    request = StoryRequest(
        mode="expand",
        idea="Two siblings find a locked room in their grandmother's house that wasn't there yesterday.",
        genre="horror",
        tone="mythic",
        language="english",
        characters="Riya, Arun",
        length="medium",
    )
    result = llm.generate(request, SAMPLE_CONTEXT)

    assert "output" in result
    assert len(result["output"]) > 50, "Scene expansion is too short"
    print(f"  Scene expansion length: {len(result['output'])} chars")


def test_llm_generates_continuation(llm):
    """LLM should generate a story continuation."""
    request = StoryRequest(
        mode="continue",
        idea="The detective realizes the anonymous tip came from inside the police station.",
        genre="thriller",
        tone="cinematic",
        language="hinglish",
        characters="ACP Verma, Sanya",
        length="long",
    )
    result = llm.generate(request, SAMPLE_CONTEXT)

    assert "output" in result
    assert len(result["output"]) > 50, "Continuation is too short"
    print(f"  Continuation length: {len(result['output'])} chars")


def test_llm_story_bible_structure(llm):
    """story_bible should contain the expected sub-fields."""
    request = StoryRequest(
        mode="hook",
        idea="A wedding planner discovers a note that predicts disasters at every event she organizes.",
        genre="family drama",
        tone="emotional",
        language="hindi",
        characters="Kavya, Rahul",
        length="short",
    )
    result = llm.generate(request, SAMPLE_CONTEXT)
    bible = result.get("story_bible", {})

    assert "central_conflict" in bible, "story_bible missing 'central_conflict'"
    assert "primary_characters" in bible, "story_bible missing 'primary_characters'"
    assert isinstance(bible["primary_characters"], list), "primary_characters should be a list"
    print(f"  Story bible: {bible}")


def test_llm_style_notes_is_list(llm):
    """style_notes should be a list of strings."""
    request = StoryRequest(
        mode="hook",
        idea="A comedian realizes the audience is laughing at jokes he hasn't told yet.",
        genre="comedy",
        tone="witty",
        language="english",
        characters="Sameer",
        length="short",
    )
    result = llm.generate(request, SAMPLE_CONTEXT)

    assert isinstance(result.get("style_notes"), list), "style_notes should be a list"
    assert len(result["style_notes"]) > 0, "style_notes should not be empty"
    print(f"  Style notes: {result['style_notes']}")


def test_llm_raises_without_api_key():
    """StoryLLM should raise ValueError if no API key is provided."""
    # Temporarily unset the env var
    original = os.environ.pop("GROQ_API_KEY", None)
    try:
        with pytest.raises(ValueError, match="API key"):
            StoryLLM(api_key=None)
    finally:
        if original:
            os.environ["GROQ_API_KEY"] = original
