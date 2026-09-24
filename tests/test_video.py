"""Video pipeline tests.

These run offline: no LLM, no TTS model, no network beyond a one-time font
fetch. The things worth locking in are the refusals -- a video is more
persuasive than text, so a confident wrong one is worse than none at all.
"""

import os
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("STORYTUTOR_FONT_DIR", str(Path(tempfile.gettempdir()) / "st_fonts"))

from story_mvp.video import cards
from story_mvp.video.script import (
    SCENE_BUDGET,
    TOTAL_SECONDS,
    Scene,
    ScriptError,
    VideoScript,
    count_words,
    script_from_answer,
    validate,
)


def demo_script(**over) -> VideoScript:
    base = dict(
        question="Why does a spoon get hot in hot water?",
        language="english", class_level="6", subject="science",
        title="Why Spoons Get Hot",
        scenes=[
            Scene("title", "Why Spoons Get Hot", "", "Why does a spoon get hot?", 4.0),
            Scene("idea", "The core idea", "Heat moves from hot to cold.",
                  "Heat always flows from hotter to cooler, never the other way.", 9.0),
            Scene("diagram", "How heat moves", "",
                  "Hot water passes heat to the spoon, and the spoon to your hand.", 11.0),
            Scene("check", "Your turn", "Would a wooden spoon get hot as fast?",
                  "Would a wooden spoon get hot as fast?", 6.0),
        ],
        diagram_nodes=["Hot water", "Metal spoon", "Your hand"],
        diagram_edges=[["Hot water", "Metal spoon"], ["Metal spoon", "Your hand"]],
        source_line="NCERT Class 6, Science, Ch. 8, p. 3",
        sources=[{"marker": "S1", "page": 3}],
    )
    base.update(over)
    return VideoScript(**base)


# ---------------- budgets ----------------

def test_thirty_seconds_is_about_seventy_words():
    """The budget is arithmetic, not taste: ~2.3 words/sec x 30s."""
    assert TOTAL_SECONDS == 30.0
    total_words = sum(w for _, w in SCENE_BUDGET.values())
    assert 65 <= total_words <= 85, f"budget of {total_words} words will not fit 30 seconds"


def test_word_count_handles_devanagari():
    assert count_words("उष्णता गरम पाण्यापासून थंड चमच्याकडे वाहते") == 6
    assert count_words("heat flows from hot to cold") == 6


def test_a_valid_script_passes():
    assert validate(demo_script(), n_sources=1) == []


def test_overlong_narration_is_rejected():
    s = demo_script()
    s.scenes[0].narration = "word " * 40
    problems = validate(s, n_sources=1)
    assert any("limit" in p for p in problems)


def test_invented_citation_is_rejected_before_it_reaches_a_frame():
    """A [S9] on screen looks authoritative and cannot be checked."""
    s = demo_script()
    s.scenes[1].narration = "Heat moves from hot to cold [S9]."
    problems = validate(s, n_sources=1)
    assert any("S9" in p for p in problems)


def test_diagram_must_be_internally_consistent():
    s = demo_script(diagram_edges=[["Hot water", "Nonexistent node"]])
    assert any("unknown node" in p for p in validate(s, n_sources=1))

    s2 = demo_script(diagram_nodes=["only one"])
    assert any("2-6 nodes" in p for p in validate(s2, n_sources=1))


# ---------------- grounding refusal ----------------

def test_no_sources_means_no_video():
    """The central safety property of this pipeline."""
    with pytest.raises(ScriptError) as exc:
        script_from_answer({"answer": "anything", "sources": [], "grounded": False},
                           llm_client=None, request={"language": "english"})
    assert "no video" in str(exc.value).lower()


def test_ungrounded_but_with_sources_still_refuses():
    with pytest.raises(ScriptError):
        script_from_answer({"answer": "x", "sources": [{"marker": "S1"}], "grounded": False},
                           llm_client=None, request={"language": "english"})


# ---------------- rendering ----------------

def test_english_cards_render_at_video_resolution():
    img = cards.title_card("Heat transfer", "Class 6 · Science", "NCERT Class 6")
    assert img.size == (1280, 720)
    assert cards.idea_card("Idea", "Heat moves from hot to cold.").size == (1280, 720)
    assert cards.check_card("Your turn", "Why?").size == (1280, 720)


def test_diagram_renders_with_nodes_and_edges():
    img = cards.diagram_card("How heat moves",
                             ["Hot water", "Metal spoon", "Your hand"],
                             [["Hot water", "Metal spoon"], ["Metal spoon", "Your hand"]])
    assert img.size == (1280, 720)


def test_diagram_survives_an_edge_to_a_missing_node():
    """Validation should have caught it, but rendering must not crash."""
    img = cards.diagram_card("x", ["a", "b", "c"], [["a", "ghost"]])
    assert img.size == (1280, 720)


def test_devanagari_is_refused_without_a_text_shaper():
    """Without libraqm, matras render in the wrong order -- text that looks
    almost right, which is worse than obviously broken."""
    from PIL import features

    if features.check("raqm"):
        img = cards.title_card("उष्णता कशी वाहते?", "इयत्ता 6")
        assert img.size == (1280, 720)
    else:
        with pytest.raises(cards.RenderError) as exc:
            cards.title_card("उष्णता कशी वाहते?", "इयत्ता 6")
        assert "raqm" in str(exc.value)


def test_english_still_renders_when_shaping_is_unavailable():
    assert cards.title_card("Heat transfer", "Class 6").size == (1280, 720)


# ---------------- assembly ----------------

def test_subtitles_are_written_in_srt_time_format():
    from story_mvp.video.render import write_srt

    s = demo_script()
    with tempfile.TemporaryDirectory() as d:
        p = write_srt(s.scenes, [4.0, 9.0, 11.0, 6.0], Path(d) / "s.srt")
        text = p.read_text(encoding="utf-8")
    assert "00:00:00,000 --> 00:00:04,000" in text
    assert "00:00:13,000 --> 00:00:24,000" in text
    assert "Heat always flows" in text


def test_a_near_empty_script_is_sent_back():
    """Four words across four scenes passed on Sol and explained nothing."""
    s = demo_script()
    for scene in s.scenes:
        scene.narration = "पौधे"
    problems = validate(s, n_sources=1)
    assert sum("too short" in p for p in problems) == 4
