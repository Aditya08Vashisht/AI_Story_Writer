"""Reproduce the two script failures seen on Sol, 2026-09-24.

1. Hindi and Marathi: valid JSON that mapped to nothing -- every scene empty,
   zero diagram nodes, identical on all three attempts.
2. English oceans: edges naming nodes the node list never declared.
"""

from story_mvp.video.script import _assemble, recover_structure, validate

REQ = {"language": "hindi", "class_level": "7", "subject": "science", "question": "q"}
SOURCES = [{"marker": "S1", "class_level": "7", "subject": "science", "chapter": "10", "page": 8}]


def structural(problems):
    """These tests use placeholder narration; length has its own tests."""
    return [p for p in problems if "too short" not in p]


def good_scene(n):
    return {"heading": f"शीर्षक {n}", "body": "पौधे सूर्य के प्रकाश से भोजन बनाते हैं।",
            "narration": "पौधे पत्तियों में सूर्य के प्रकाश से भोजन बनाते हैं।"}


def test_translated_top_level_and_scene_keys_are_recovered():
    """The likely Hindi failure: the model translated the JSON keys."""
    reply = {
        "शीर्षक": "पौधों का भोजन",
        "दृश्य": {
            "शीर्षक": {"शीर्षक": "पौधों का भोजन", "मुख्य": "प्रकाश संश्लेषण", "कथन": "पौधे भोजन कैसे बनाते हैं?"},
            "विचार": {"शीर्षक": "मुख्य विचार", "मुख्य": "पत्तियां भोजन बनाती हैं।", "कथन": "पौधे पत्तियों में भोजन बनाते हैं।"},
            "चित्र": {"शीर्षक": "कैसे", "मुख्य": "प्रक्रिया", "कथन": "सूर्य का प्रकाश, पानी और हवा मिलकर भोजन बनाते हैं।"},
            "प्रश्न": {"शीर्षक": "सोचिए", "मुख्य": "जड़ें क्या करती हैं?", "कथन": "सोचिए, जड़ें क्या करती हैं?"},
        },
        "नोड": ["सूर्य का प्रकाश", "पत्ती", "भोजन"],
        "तीर": [["सूर्य का प्रकाश", "पत्ती"], ["पत्ती", "भोजन"]],
    }
    script = _assemble(reply, {"question": "q"}, REQ, SOURCES, "ollama")
    assert all(s.narration for s in script.scenes), "scenes still empty after recovery"
    assert script.diagram_nodes == ["सूर्य का प्रकाश", "पत्ती", "भोजन"]
    assert structural(validate(script, 1)) == []


def test_scenes_as_a_list_are_mapped_in_order():
    reply = {"title": "t", "scenes": [good_scene(i) for i in range(4)],
             "diagram_nodes": ["a", "b", "c"], "diagram_edges": [["a", "b"], ["b", "c"]]}
    script = _assemble(reply, {"question": "q"}, REQ, SOURCES, "ollama")
    assert [s.key for s in script.scenes] == ["title", "idea", "diagram", "check"]
    assert all(s.narration for s in script.scenes)


def test_correct_replies_are_left_untouched():
    reply = {"title": "t",
             "scenes": {k: good_scene(k) for k in ("title", "idea", "diagram", "check")},
             "diagram_nodes": ["a", "b", "c"], "diagram_edges": [["a", "b"]]}
    assert recover_structure(reply)["scenes"]["idea"] == good_scene("idea")


def test_edge_endpoints_missing_from_nodes_are_added():
    """The English oceans failure, which took three attempts to pass."""
    reply = {"title": "Oceans and continents",
             "scenes": {k: {"heading": "h", "body": "b", "narration": "short line"}
                        for k in ("title", "idea", "diagram", "check")},
             "diagram_nodes": ["Earth's surface", "Water"],
             "diagram_edges": [["Oceans", "Continents"], ["Earth's surface", "Oceans"]]}
    script = _assemble(reply, {"question": "q"}, {**REQ, "language": "english"}, SOURCES, "ollama")
    lowered = [n.lower() for n in script.diagram_nodes]
    assert "oceans" in lowered and "continents" in lowered
    assert structural(validate(script, 1)) == []


def test_recovery_never_invents_content():
    """An empty reply stays empty and still fails validation."""
    script = _assemble({}, {"question": "q"}, REQ, SOURCES, "ollama")
    assert not any(s.narration for s in script.scenes)
    assert validate(script, 1), "an empty reply must not pass"


def test_rejected_replies_are_saved_for_diagnosis(tmp_path):
    from story_mvp.video.script import _dump_reply

    _dump_reply(tmp_path, {"language": "hindi", "question": "पौधे भोजन"}, 1, '{"दृश्य": {}}', ["empty"])
    saved = list(tmp_path.iterdir())
    assert len(saved) == 1
    text = saved[0].read_text(encoding="utf-8")
    assert "RAW REPLY" in text and "दृश्य" in text


def test_script_call_sends_an_instruction_not_the_question(monkeypatch):
    """Sending the student's question as the user turn made the model answer
    it ({"answer": ...}) instead of writing a script -- every Hindi and Marathi
    attempt on Sol came back with no scenes at all."""
    import story_mvp.rag_chat as rc
    from story_mvp.video.script import SCRIPT_INSTRUCTION, script_from_answer

    seen = {}

    def capture(provider, system, user, history=None):
        seen.setdefault("user", user)          # the script call comes first
        return ('{"title":"पौधों का भोजन","scenes":{'
                + ",".join(f'"{k}":{{"heading":"h","body":"b","narration":"पौधे भोजन बनाते हैं"}}'
                           for k in ("title", "idea", "diagram", "check"))
                + '},"diagram_nodes":["क","ख"],"diagram_edges":[["क","ख"]]}')

    monkeypatch.setattr(rc, "_call_provider", capture)
    fake = type("P", (), {"provider_name": "ollama"})()
    script_from_answer({"question": "पौधे भोजन कैसे बनाते हैं?", "answer": "a", "grounded": True,
                        "sources": SOURCES}, fake, REQ)
    assert seen["user"] == SCRIPT_INSTRUCTION
    assert "पौधे" not in seen["user"]


def test_two_node_diagram_is_accepted():
    """'hot water -> spoon' is a complete diagram; requiring 3 refused it."""
    reply = {"title": "t",
             "scenes": {k: {"heading": "h", "body": "b", "narration": "short line"}
                        for k in ("title", "idea", "diagram", "check")},
             "diagram_nodes": ["Hot water", "Spoon"], "diagram_edges": [["Hot water", "Spoon"]]}
    script = _assemble(reply, {"question": "q"}, {**REQ, "language": "english"}, SOURCES, "ollama")
    assert structural(validate(script, 1)) == []


def test_one_word_overrun_is_tolerated_but_a_long_one_is_not():
    from story_mvp.video.script import Scene

    base = {"title": "t",
            "scenes": {k: {"heading": "h", "body": "b", "narration": "ok"}
                       for k in ("title", "idea", "diagram", "check")},
            "diagram_nodes": ["a", "b"], "diagram_edges": [["a", "b"]]}
    script = _assemble(base, {"question": "q"}, {**REQ, "language": "english"}, SOURCES, "ollama")
    script.scenes[3].narration = " ".join(["w"] * 15)   # budget 14
    assert structural(validate(script, 1)) == []
    script.scenes[3].narration = " ".join(["w"] * 40)
    assert structural(validate(script, 1))


# ---------------- script vs grounded answer ----------------

def _heat_script(idea="Heat moves from the hot water into the cooler spoon.",
                 edges=(("Hot water", "Metal spoon"),)):
    from story_mvp.video.script import Scene, VideoScript

    return VideoScript(
        question="Why does a spoon get hot?", language="english", class_level="6", subject="science",
        title="Why Spoons Get Hot",
        scenes=[Scene("title", "t", "", "Why does a spoon get hot?", 4.0),
                Scene("idea", "i", "", idea, 9.0),
                Scene("diagram", "d", "", "Heat travels from the water to the spoon.", 11.0),
                Scene("check", "c", "", "Would a wooden spoon get hot as fast?", 6.0)],
        diagram_nodes=[n for e in edges for n in e], diagram_edges=[list(e) for e in edges],
        source_line="NCERT", sources=[{"marker": "S1"}])


ANSWER = "Heat moves from the hotter water to the cooler spoon [S1]. Metals conduct heat well."


class Judge:
    provider_name = "ollama"


def test_the_judge_sees_every_factual_line_and_arrow(monkeypatch):
    import story_mvp.rag_chat as rc
    from story_mvp.video.script import grounding_problems

    seen = {}
    monkeypatch.setattr(rc, "_call_provider",
                        lambda p, s, u, h=None: seen.setdefault("prompt", s) and '{"unsupported": []}')
    problems, report = grounding_problems(_heat_script(), ANSWER, [Judge()])
    assert problems == [] and report["verified"] is True
    assert "[C1] Heat moves" in seen["prompt"] and "[D3] Hot water -> Metal spoon" in seen["prompt"]
    assert "Would a wooden spoon" not in seen["prompt"], "the check question is not a claim"


def test_an_invented_fact_and_arrow_are_sent_back(monkeypatch):
    import story_mvp.rag_chat as rc
    from story_mvp.video.script import grounding_problems

    monkeypatch.setattr(rc, "_call_provider", lambda p, s, u, h=None: (
        '{"unsupported": [{"id": "C1", "why": "the answer gives no temperature"},'
        ' {"id": "D3", "why": "the answer never says the spoon heats the water"}]}'))
    script = _heat_script(idea="The spoon reaches exactly 80 degrees in one minute.",
                          edges=(("Metal spoon", "Hot water"),))
    problems, report = grounding_problems(script, ANSWER, [Judge()])
    assert len(problems) == 2 and report["verified"] is False
    assert any("scene 'idea'" in p and "no temperature" in p for p in problems)
    assert any("diagram arrow 'Metal spoon -> Hot water'" in p for p in problems)


def test_a_line_that_plainly_restates_the_answer_overrides_a_strict_judge(monkeypatch):
    import numpy as np

    import story_mvp.rag_chat as rc
    from story_mvp.video.script import grounding_problems

    monkeypatch.setattr(rc, "_call_provider", lambda p, s, u, h=None: '{"unsupported": [{"id": "C1"}]}')
    same = lambda texts: np.array([[1.0, 0.0]] * len(texts))   # noqa: E731  everything matches
    problems, _ = grounding_problems(_heat_script(), ANSWER, [Judge()], embed=same)
    assert problems == []


def test_without_a_model_embeddings_do_the_check():
    import numpy as np

    from story_mvp.video.script import grounding_problems

    far = lambda texts: np.array([[1.0, 0.0]] + [[0.0, 1.0]] * (len(texts) - 1))  # noqa: E731
    problems, report = grounding_problems(_heat_script(), ANSWER, providers=None, embed=far)
    assert report["method"] == "embedding" and problems
