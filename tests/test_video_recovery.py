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
        seen["user"] = user
        return ('{"title":"t","scenes":{'
                + ",".join(f'"{k}":{{"heading":"h","body":"b","narration":"एक दो तीन"}}'
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
