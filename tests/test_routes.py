import os
os.environ.setdefault("STORYTUTOR_SKIP_AI_INIT", "1")

"""Route smoke tests.

These exist because a JSX template shipped broken: React inline styles use
{{...}}, which is Jinja's expression syntax, so Flask raised
TemplateSyntaxError on every page load. py_compile could not catch that --
only actually requesting the route can.
"""

import os
import pytest

os.environ.setdefault("STORYTUTOR_EMBEDDING_MODEL", "all-MiniLM-L6-v2")


@pytest.fixture(scope="module")
def client():
    from story_mvp.app import app

    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_root_serves_the_chat_ui(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "StoryTutor" in body
    assert "/api/chat" in body, "UI is not wired to the chat endpoint"


def test_chat_alias_serves_the_same_page(client):
    assert client.get("/chat").status_code == 200


def test_ui_carries_no_story_generator_remnants(client):
    body = client.get("/").get_data(as_text=True).lower()
    for gone in ("hook generator", "scene expander", "story continuation", "genreselect"):
        assert gone not in body, f"story-generator UI remnant still present: {gone}"


def test_health_reports_which_engine_is_live(client):
    data = client.get("/api/health").get_json()
    assert data["status"] == "online"
    assert "model_provider" in data


def test_chat_endpoint_rejects_an_empty_question(client):
    r = client.post("/api/chat", json={"question": "   "})
    # 503 when the retrieval engine failed to load, 200 with a refusal otherwise.
    assert r.status_code in (200, 503)
    if r.status_code == 200:
        assert r.get_json()["grounded"] is False
