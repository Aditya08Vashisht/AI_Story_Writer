"""Tests for the RAG chat pipeline.

These run offline: retrieval is stubbed, so no model downloads and no network.
"""

from story_mvp.rag_chat import (
    MIN_EVIDENCE_SCORE,
    answer_question,
    build_chat_prompt,
    normalize_chat_request,
)


class FakeRAG:
    """Returns whatever it is given, recording the filters it was called with."""

    def __init__(self, results):
        self.results = results
        self.last_filters = None

    def retrieve(self, query, filters=None, top_k=5):
        self.last_filters = filters
        return self.results[:top_k]


def chunk(score=0.8, **kw):
    base = {
        "id": "6_science_english_Ch-8_p3_c1",
        "text": "Heat energy moves from the hot water to the cooler spoon until both reach the same temperature.",
        "class_level": "6", "subject": "science", "language": "english",
        "chapter": "8", "page_start": 3, "source_file": "/x/Ch-8_Science_Class6.pdf",
        "retrieval_score": score, "retrieval_method": "hybrid_rrf_rerank",
    }
    base.update(kw)
    return base


def test_normalize_accepts_several_question_keys():
    for key in ("question", "idea", "message"):
        assert normalize_chat_request({key: "why is the sky blue"})["question"] == "why is the sky blue"


def test_normalize_rejects_unknown_enum_values():
    r = normalize_chat_request({"question": "q", "language": "klingon", "class_level": "12", "subject": "art"})
    assert r["language"] == "english"
    assert r["class_level"] == ""
    assert r["subject"] == ""


def test_normalize_maps_sst_aliases():
    assert normalize_chat_request({"question": "q", "subject": "sst"})["subject"] == "social_science"
    assert normalize_chat_request({"question": "q", "subject": "social science"})["subject"] == "social_science"


def test_prompt_instructs_tutoring_not_story_generation():
    """Guards the product identity, not a word list.

    Checked twice before by banning substrings, which kept firing on the very
    lines that FORBID story generation ("do not invent characters"). What
    matters is whether the prompt *directs* fiction, so this asserts on
    directives and on the prohibition being present.
    """
    p = build_chat_prompt(normalize_chat_request({"question": "q", "language": "marathi"}), "[S1] ...")
    lowered = p.lower()

    for directive in ("write a story", "genre:", "tone:", "cliffhanger",
                      "story bible", "scene expander", "hook generator",
                      "audio-first", "serialized listening"):
        assert directive not in lowered, f"story-generation directive leaked: {directive}"

    # The prohibition itself must survive.
    assert "invent characters, dialogue or fictional scenes" in lowered
    assert "never" in lowered


def test_prompt_separates_facts_from_pedagogy():
    """The change that makes it teach: analogies are allowed and uncited,
    curriculum facts are neither."""
    p = build_chat_prompt(normalize_chat_request({"question": "q", "class_level": "6"}), "[S1] ...")
    lowered = p.lower()
    assert "analog" in lowered, "prompt never invites an analogy"
    assert "[s1]" in lowered, "prompt never shows the citation format"
    assert "need no citation" in lowered or "no citation" in lowered
    # class-appropriate pitch actually varies
    p8 = build_chat_prompt(normalize_chat_request({"question": "q", "class_level": "8"}), "[S1] ...")
    assert p != p8, "prompt does not adapt to class level"


def test_prompt_language_and_pitch_adapt():
    mr = build_chat_prompt(normalize_chat_request({"question": "q", "language": "marathi"}), "x")
    en = build_chat_prompt(normalize_chat_request({"question": "q", "language": "english"}), "x")
    # Both prompts name all three languages when describing scope, so assert on
    # the reply directive specifically.
    assert "Reply in Marathi." in mr
    assert "Reply in English." in en


def test_empty_question_short_circuits():
    out = answer_question({"question": "   "}, FakeRAG([chunk()]))
    assert out["grounded"] is False
    assert out["sources"] == []
    assert out["intent"] == "greeting"


def test_weak_evidence_refuses_rather_than_guesses():
    weak = chunk(score=MIN_EVIDENCE_SCORE - 0.05)
    out = answer_question({"question": "explain photosynthesis"}, FakeRAG([weak]))
    assert out["grounded"] is False
    assert "could not find" in out["answer"].lower()


def test_refusal_is_written_in_the_asked_language():
    weak = chunk(score=0.01, language="marathi")
    out = answer_question({"question": "प्रश्न", "language": "marathi"}, FakeRAG([weak]))
    # Devanagari, not an English refusal shown to a Marathi learner.
    assert any("ऀ" <= ch <= "ॿ" for ch in out["answer"])


def test_without_an_llm_it_returns_real_passages_not_invented_prose():
    """Honest degradation. The old engine wrote thriller fiction here."""
    out = answer_question({"question": "how does heat move"}, FakeRAG([chunk()]), llm_client=None)
    assert out["grounded"] is True
    assert out["model_provider"] == "passages_only"
    assert "Heat energy moves from the hot water" in out["answer"]
    assert "apartment" not in out["answer"].lower()


def test_citations_expose_provenance():
    out = answer_question({"question": "how does heat move"}, FakeRAG([chunk()]))
    s = out["sources"][0]
    assert s["marker"] == "S1"
    assert s["source_file"] == "Ch-8_Science_Class6.pdf"   # basename, not a full path
    assert s["page"] == 3
    assert s["excerpt"]


def test_filters_are_passed_through_to_retrieval():
    rag = FakeRAG([chunk()])
    answer_question({"question": "what are the main crops grown in India?", "class_level": "7", "subject": "sst", "language": "hindi"}, rag)
    assert rag.last_filters == {"class_level": "7", "subject": "social_science", "language": "hindi"}


class FakeOllama:
    """Stands in for a live Ollama server, matching the provider duck-type."""

    provider_name = "ollama"

    def __init__(self, reply):
        self.reply = reply
        self.seen_system = None


class FakeLLM:
    def __init__(self, provider):
        self.client = type("chain", (), {"clients": [provider], "last_provider": "none"})()
        self.last_provider = "none"


def test_full_chain_retrieval_to_grounded_answer(monkeypatch):
    """The integration that matters: retrieved chunks reach the model, and its
    answer comes back with citations and the real provider name attached."""
    import story_mvp.rag_chat as rc

    provider = FakeOllama('{"answer": "Heat moves from hot water into the spoon [S1]."}')

    def fake_call(p, system, user, history=None):
        p.seen_system = system
        return p.reply

    monkeypatch.setattr(rc, "_call_provider", fake_call)

    out = rc.answer_question(
        {"question": "how does heat move", "class_level": "6", "subject": "science"},
        FakeRAG([chunk()]),
        FakeLLM(provider),
    )

    assert out["grounded"] is True
    assert out["model_provider"] == "ollama"
    assert "[S1]" in out["answer"]
    assert out["sources"][0]["page"] == 3
    # the retrieved passage must actually be in the prompt the model saw
    assert "Heat energy moves from the hot water" in provider.seen_system
    assert "[S1]" in provider.seen_system


def test_a_failing_model_degrades_to_passages(monkeypatch):
    import story_mvp.rag_chat as rc

    def boom(p, system, user, history=None):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(rc, "_call_provider", boom)
    out = rc.answer_question({"question": "how does heat move between objects?"}, FakeRAG([chunk()]), FakeLLM(FakeOllama("")))
    assert out["model_provider"] == "passages_only"
    assert "Heat energy moves" in out["answer"]


def test_non_json_reply_is_still_usable(monkeypatch):
    """Small models sometimes ignore the JSON instruction. Take the prose."""
    import story_mvp.rag_chat as rc

    monkeypatch.setattr(rc, "_call_provider", lambda p, s, u, h=None: "Heat flows from hot to cold [S1].")
    out = rc.answer_question({"question": "how does heat move between objects?"}, FakeRAG([chunk()]), FakeLLM(FakeOllama("")))
    assert out["model_provider"] == "ollama"
    assert "Heat flows from hot to cold" in out["answer"]


def test_no_sources_means_the_model_is_never_offered_the_chance_to_answer(monkeypatch):
    """Observed in production: 'who was mahatma gandhi' returned a full answer
    with zero retrieved sources. Instruction-following is not a safety
    mechanism at 7B, so with no evidence the model gets a prompt that contains
    no passages and permits only a greeting, a capability reply, or a refusal.
    """
    import story_mvp.rag_chat as rc

    seen = {}

    def capture(p, system, user, history=None):
        seen["system"] = system
        return '{"answer": "I could not find that in those books."}'

    monkeypatch.setattr(rc, "_call_provider", capture)

    # every result scores below the evidence threshold
    weak = chunk(score=0.01)
    out = rc.answer_question({"question": "who was mahatma gandhi"},
                             FakeRAG([weak]), FakeLLM(FakeOllama("")))

    assert out["grounded"] is False
    assert "You have NO textbook passages" in seen["system"]
    assert "must NOT answer a factual question" in seen["system"]
    # the passage text must not be smuggled in
    assert "Heat energy moves" not in seen["system"]


def test_with_sources_the_full_teaching_prompt_is_used(monkeypatch):
    import story_mvp.rag_chat as rc

    seen = {}

    def capture(p, system, user, history=None):
        seen["system"] = system
        return '{"answer": "Heat flows hot to cold [S1]."}'

    monkeypatch.setattr(rc, "_call_provider", capture)
    out = rc.answer_question({"question": "how does heat move between objects?"},
                             FakeRAG([chunk()]), FakeLLM(FakeOllama("")))

    assert out["grounded"] is True
    assert "analog" in seen["system"].lower()
    assert "Heat energy moves from the hot water" in seen["system"]
