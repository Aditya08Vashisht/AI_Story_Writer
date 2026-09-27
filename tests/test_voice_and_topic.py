"""Voice, topic anchoring, and the video-mismatch fixes.

Offline: no TTS model, no LLM. The voice engines themselves only run on Sol;
these lock in everything around them.
"""

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("STORYTUTOR_SKIP_AI_INIT", "1")
os.environ.setdefault("STORYTUTOR_FONT_DIR", str(Path(tempfile.gettempdir()) / "st_fonts"))

from story_mvp.video import cards, render as rnd  # noqa: E402
from story_mvp.video.script import build_script_prompt, topic_problems  # noqa: E402
from story_mvp.video.tts import RemoteTTS, clean_for_speech, split_sentences  # noqa: E402
from test_video import demo_script  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


# ---------------- the title/content mismatch ----------------

def test_the_question_is_in_the_script_prompt():
    """It was not -- the model saw only the answer, and titled videos after
    whatever else the answer mentioned."""
    prompt = build_script_prompt("Why does a metal spoon get hot?", "Heat flows [S1].",
                                 [{"marker": "S1"}], {"language": "english"})
    assert "Why does a metal spoon get hot?" in prompt


def test_an_on_topic_script_passes_the_topic_check():
    assert topic_problems(demo_script()) == []


def test_a_title_from_another_video_is_caught():
    s = demo_script(title="Oceans and Continents")
    problems = topic_problems(s)
    assert len(problems) == 1 and "title" in problems[0]


def test_hindi_inflections_count_as_the_same_topic():
    s = demo_script(question="पौधे अपना भोजन कैसे बनाते हैं?", title="पौधों का भोजन", language="hindi")
    for sc in s.scenes:
        sc.narration = "पौधे पत्तियों में सूर्य के प्रकाश से भोजन बनाते हैं।"
    assert topic_problems(s) == []


def test_a_paraphrased_title_is_rescued_by_the_embedding_check():
    import numpy as np

    s = demo_script(title="Conduction in the kitchen")   # shares no word with the question
    same = lambda texts: np.array([[1.0, 0.0], [0.9, 0.1]])   # noqa: E731  similar
    far = lambda texts: np.array([[1.0, 0.0], [0.0, 1.0]])    # noqa: E731  unrelated
    assert topic_problems(s, embed=same) == []
    assert topic_problems(s, embed=far)


def test_two_hindi_questions_never_share_a_file():
    """isalnum() drops vowel signs, so these used to collapse to one name and
    one video overwrote the other."""
    from make_video import slugify

    a, b = slugify("पौधे अपना भोजन कैसे बनाते हैं?"), slugify("पेड़ अपना भोजन कैसे बनाते हैं?")
    assert a != b
    assert slugify("Why is the sky blue?") == slugify("Why is the sky blue?")


# ---------------- speech text ----------------

def test_citations_and_markdown_are_not_read_aloud():
    t = clean_for_speech("**Heat** flows from hot to cold [S1].\n- Metals conduct [S2].")
    assert "[S" not in t and "*" not in t and "-" not in t
    assert "Heat flows from hot to cold" in t


def test_answers_are_split_into_speakable_sentences():
    text = "पौधे भोजन बनाते हैं। वे सूर्य का प्रकाश लेते हैं। " + "Long sentence " * 40 + "."
    pieces = split_sentences(text)
    assert pieces[0].startswith("पौधे")
    assert all(len(p) <= 230 for p in pieces)
    assert len(pieces) >= 3


def test_voice_client_reports_a_missing_server_instead_of_raising(tmp_path):
    tts = RemoteTTS(url="http://127.0.0.1:9")   # nothing listens on port 9
    assert tts.available is False
    assert "no voice server" in tts.failure
    assert tts.speak("hello", "english", tmp_path / "x.wav") is None


# ---------------- routes ----------------

@pytest.fixture(scope="module")
def client():
    from story_mvp.app import app

    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_tts_endpoint_says_503_when_the_voice_server_is_down(client, monkeypatch):
    monkeypatch.setenv("STORYTUTOR_TTS_URL", "http://127.0.0.1:9")
    r = client.post("/api/tts", json={"text": "Heat flows.", "language": "hindi"})
    assert r.status_code == 503
    assert client.post("/api/tts", json={"text": ""}).status_code == 400


def test_chat_page_has_the_listen_button(client):
    body = client.get("/").get_data(as_text=True)
    assert "/api/tts" in body and "Listen" in body and "speechSynthesis" in body


def test_video_urls_change_when_the_video_is_rerendered(client, tmp_path, monkeypatch):
    """Same file name after a re-run let the browser replay its cached copy
    under the new title."""
    import story_mvp.app as appmod

    (tmp_path / "v.mp4").write_bytes(b"x")
    monkeypatch.setattr(appmod, "VIDEO_DIR", str(tmp_path))
    first = client.get("/api/videos").get_json()[0]["url"]
    os.utime(tmp_path / "v.mp4", (1_900_000_000, 1_900_000_000))
    second = client.get("/api/videos").get_json()[0]["url"]
    assert re.search(r"\?v=\d+$", first) and first != second


# ---------------- narrated render ----------------

def test_narration_is_padded_to_each_scene(tmp_path):
    """Joining raw clips let the voice run ahead of the cards; a scene with no
    clip shifted every later line. Each scene now gets exactly its own slot."""
    ff = rnd.ffmpeg_binary()
    s = demo_script()
    frames = [cards.render_scene(sc, s) for sc in s.scenes]
    durations = [3.0, 3.0, 3.0, 3.0]
    clips = []
    for i in range(4):
        if i == 2:
            clips.append(None)          # a line that failed to synthesise
            continue
        wav = tmp_path / f"a{i}.wav"
        subprocess.run([ff, "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=1.2",
                        "-ar", "22050", str(wav)], capture_output=True, check=True)
        clips.append(wav)
    out = rnd.render_video(s, frames, clips, durations, tmp_path / "n.mp4", motion=False)
    info = subprocess.run([ff, "-i", str(out)], capture_output=True, text=True).stderr
    assert "Audio:" in info
    h, m, sec = re.search(r"Duration: (\d+):(\d+):([\d.]+)", info).groups()
    assert abs(int(h) * 3600 + int(m) * 60 + float(sec) - 12.0) < 0.5


def test_voice_server_round_trip(tmp_path, monkeypatch):
    """Server + client + cache, with a stand-in voice so no model is needed."""
    sf = pytest.importorskip("soundfile")
    import threading
    from http.server import ThreadingHTTPServer

    import numpy as np
    import tts_server

    class FakeVoice:
        name, sample_rate, failure, model_id = "fake", 16000, None, "fake"
        calls = 0

        def speak_array(self, text, language):
            FakeVoice.calls += 1
            return np.zeros(8000, dtype=np.float32)

    monkeypatch.setattr(tts_server, "ENGINE", FakeVoice())
    monkeypatch.setattr(tts_server, "CACHE", tmp_path / "cache")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), tts_server.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        client = RemoteTTS(url=f"http://127.0.0.1:{srv.server_address[1]}")
        assert client.available and client.engine == "fake"
        out = client.speak("पौधे भोजन बनाते हैं।", "hindi", tmp_path / "a.wav")
        assert out and abs(sf.info(str(out)).duration - 0.5) < 0.01
        client.speak("पौधे भोजन बनाते हैं।", "hindi", tmp_path / "b.wav")
        assert FakeVoice.calls == 1, "a repeated sentence must come from the cache"
    finally:
        srv.shutdown()
