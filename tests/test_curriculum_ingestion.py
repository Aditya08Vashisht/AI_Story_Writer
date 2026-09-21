from story_mvp.ingest_curriculum import _metadata_from_path, _missing_expected_coverage


def test_metadata_from_ncert_path_detects_subject_language_and_class():
    metadata = _metadata_from_path(
        r"D:\AI_Story_Writer_MVP\story_datasets\NCERT_6th-8th\Class6_Science-marathi\fmrcu101.pdf"
    )

    assert metadata.class_level == "6"
    assert metadata.subject == "science"
    assert metadata.language == "marathi"
    assert metadata.chapter == "101"


def test_metadata_from_sst_path_normalizes_social_science():
    metadata = _metadata_from_path(
        r"D:\AI_Story_Writer_MVP\story_datasets\NCERT_6th-8th\Class6_sst-Hindi\fhes1dd\fhes114.pdf"
    )

    assert metadata.class_level == "6"
    assert metadata.subject == "social_science"
    assert metadata.language == "hindi"


def test_metadata_from_path_detects_class_7_books():
    metadata = _metadata_from_path(
        r"D:\AI_Story_Writer_MVP\story_datasets\NCERT_6th-8th\Class7th_science_Marathi\gmsc101.pdf"
    )

    assert metadata.class_level == "7"
    assert metadata.subject == "science"
    assert metadata.language == "marathi"


def test_missing_expected_coverage_flags_class_8_gaps():
    missing = _missing_expected_coverage({"class_6:science:english": 12})

    assert "class_8:science:english" in missing
    assert "class_7:science:english" in missing
    assert "class_6:science:english" not in missing
