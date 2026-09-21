from story_mvp.generator import _normalize_request


def test_class_7_payload_is_preserved():
    request = _normalize_request({"idea": "Explain heat", "class_level": "7"})

    assert request.class_level == "7"
