"""Reproduce the two script failures seen on Sol, 2026-09-24.

1. Hindi and Marathi: valid JSON that mapped to nothing -- every scene empty,
   zero diagram nodes, identical on all three attempts.
2. English oceans: edges naming nodes the node list never declared.
"""

from story_mvp.video.script import _assemble, recover_structure, validate

REQ = {"language": "hindi", "class_level": "7", "subject": "science", "question": "q"}
SOURCES = [{"marker": "S1", "class_level": "7", "subject": "science", "chapter": "10", "page": 8}]


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
    assert validate(script, 1) == []


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
    assert validate(script, 1) == []


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
