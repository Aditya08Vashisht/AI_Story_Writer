"""Illustration tests.

Run offline, without diffusers or a GPU: the image model is the one part that
cannot run here, so these lock in everything around it -- what the script must
ask for, that the same prompt always gets the same seed, that a cached picture
is used without loading a model, and that the video still renders to length
with pictures and motion.
"""

import os
import tempfile
from pathlib import Path

import pytest
from PIL import Image

os.environ.setdefault("STORYTUTOR_FONT_DIR", str(Path(tempfile.gettempdir()) / "st_fonts"))

from story_mvp.video import cards, images, render as rnd
from story_mvp.video.script import VISUAL_SCENES, build_script_prompt, validate
from test_video import demo_script  # noqa: E402  (tests/ is on sys.path under pytest)


def _illustrated_script():
    s = demo_script()
    visuals = {
        "title": "A steel spoon resting in a steaming cup of tea on a kitchen table",
        "idea": "Warm orange waves flowing from a hot cup into a cool metal spoon",
        "check": "A wooden spoon and a metal spoon side by side in a bowl of hot soup",
    }
    for scene in s.scenes:
        scene.visual = visuals.get(scene.key, "")
    return s


def _picture(colour=(40, 120, 200)) -> Image.Image:
    img = Image.new("RGB", (images.GEN_W, images.GEN_H), colour)
    # Some structure, so the push-in and the scrim have something to act on.
    for x in range(0, images.GEN_W, 64):
        img.paste((240, 200, 80), (x, 0, x + 8, images.GEN_H))
    return img


# ---------------- script contract ----------------

def test_visuals_are_only_required_when_asked():
    s = demo_script()  # no visuals at all
    assert validate(s, n_sources=1) == []
    problems = validate(s, n_sources=1, require_visuals=True)
    assert len([p for p in problems if "visual" in p]) == len(VISUAL_SCENES)


def test_a_fully_illustrated_script_passes():
    assert validate(_illustrated_script(), n_sources=1, require_visuals=True) == []


def test_devanagari_visual_is_sent_back():
    """The image model reads English; a Hindi prompt draws something unrelated."""
    s = _illustrated_script()
    s.scenes[0].visual = "गरम चाय में रखा हुआ एक चम्मच"
    problems = validate(s, n_sources=1, require_visuals=True)
    assert any("English" in p for p in problems)


def test_overlong_visual_is_sent_back():
    s = _illustrated_script()
    s.scenes[1].visual = "a spoon " * 30
    assert any("too long" in p for p in validate(s, n_sources=1, require_visuals=True))


def test_prompt_asks_for_visuals_only_when_wanted():
    args = ("Why does a spoon get hot?", "Heat flows [S1].",
            [{"marker": "S1", "text": "Heat flows from hot to cold.", "page": 3}],
            {"language": "hindi", "class_level": "6", "subject": "science"})
    without = build_script_prompt(*args)
    with_v = build_script_prompt(*args, want_visuals=True)
    assert '"visual"' not in without
    assert '"visual"' in with_v
    # Hindi narration, English picture descriptions.
    assert "English" in with_v


# ---------------- generator ----------------

def test_seed_is_deterministic_and_prompt_forbids_text():
    p = images.full_prompt("A spoon in hot tea.")
    assert images.seed_for(p) == images.seed_for(images.full_prompt("A spoon in hot tea"))
    assert images.seed_for(p) != images.seed_for(images.full_prompt("A kettle on a stove"))
    assert "no text" in p and "no labels" in p


def test_cached_picture_is_used_without_loading_a_model(tmp_path, monkeypatch):
    monkeypatch.setenv("STORYTUTOR_IMAGE_CACHE", str(tmp_path))
    gen = images.IllustrationGenerator()
    path, _, seed = gen.cache_path("A spoon in hot tea")
    path.parent.mkdir(parents=True, exist_ok=True)
    _picture().save(path)

    def boom():
        raise AssertionError("model must not load on a cache hit")

    monkeypatch.setattr(gen, "_load", boom)
    img, meta = gen.generate("A spoon in hot tea")
    assert img.size == (images.GEN_W, images.GEN_H)
    assert meta["cached"] is True and meta["seed"] == seed


def test_unavailable_model_returns_none_not_an_exception(tmp_path, monkeypatch):
    monkeypatch.setenv("STORYTUTOR_IMAGE_CACHE", str(tmp_path))
    gen = images.IllustrationGenerator()
    gen._failed = "no CUDA device"
    assert gen.generate("A spoon in hot tea") is None
    assert gen.available is False
    assert gen.generate("   ") is None


# ---------------- cards and video ----------------

def test_illustrated_cards_render_at_video_resolution():
    s = _illustrated_script()
    for scene in s.scenes:
        card = cards.render_scene(scene, s, _picture())
        assert card.size == (1280, 720)


def test_diagram_is_never_replaced_by_a_picture():
    """The one card whose content must be exact stays Pillow-drawn."""
    s = _illustrated_script()
    diagram = next(sc for sc in s.scenes if sc.key == "diagram")
    with_pic = cards.render_scene(diagram, s, _picture((255, 0, 0)))
    plain = cards.render_scene(diagram, s)
    assert with_pic.tobytes() == plain.tobytes()


def test_illustrated_video_renders_to_length_with_motion(tmp_path):
    imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")
    s = _illustrated_script()
    frames = [cards.render_scene(sc, s, _picture() if sc.key in VISUAL_SCENES else None)
              for sc in s.scenes]
    durations = [sc.seconds for sc in s.scenes]
    out = rnd.render_video(s, frames, None, durations, tmp_path / "v.mp4",
                           burn_subtitles=True, motion=True)
    assert out.exists() and out.stat().st_size > 10_000
    n_frames, secs = imageio_ffmpeg.count_frames_and_secs(str(out))
    assert abs(secs - sum(durations)) < 1.0
    assert n_frames >= int(sum(durations) * rnd.FPS) - rnd.FPS


def test_bad_visuals_never_cost_the_video(monkeypatch):
    """Narration and diagram right, pictures described in Hindi on every try:
    keep the script, blank those visuals, let the scenes use text cards."""
    import json

    import story_mvp.rag_chat as rc
    from story_mvp.video.script import script_from_answer

    calls = []

    def reply(provider, system, user, history=None):
        calls.append(1)
        scenes = {k: {"heading": "h", "body": "b", "narration": "एक दो तीन"}
                  for k in ("title", "idea", "diagram", "check")}
        scenes["title"]["visual"] = "गरम चाय में चम्मच"            # wrong language
        scenes["idea"]["visual"] = "A glowing spoon in a cup of tea"  # fine
        return json.dumps({"title": "t", "scenes": scenes,
                           "diagram_nodes": ["क", "ख"], "diagram_edges": [["क", "ख"]]},
                          ensure_ascii=False)

    monkeypatch.setattr(rc, "_call_provider", reply)
    fake = type("P", (), {"provider_name": "ollama"})()
    script = script_from_answer(
        {"question": "q", "answer": "a", "grounded": True,
         "sources": [{"marker": "S1", "text": "t", "page": 1}]},
        fake, {"language": "hindi", "class_level": "7", "subject": "science"},
        want_visuals=True)
    assert len(calls) == 3                      # it did retry for pictures first
    by_key = {s.key: s.visual for s in script.scenes}
    assert by_key["title"] == ""
    assert by_key["idea"] == "A glowing spoon in a cup of tea"
