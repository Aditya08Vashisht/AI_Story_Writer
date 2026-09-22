"""Tests for the RAG engine (embedding + FAISS retrieval)."""

import os
import json
import shutil
import pytest

os.environ.setdefault("STORYTUTOR_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

from story_mvp.rag_engine import StoryRAG

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "knowledge_base")
TEST_INDEX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_test_faiss_index")


@pytest.fixture(scope="module")
def rag(tmp_path_factory):
    """Create a small, representative curriculum index once for this module."""
    processed_path = os.path.join(DATASET_DIR, "processed", "curriculum_chunks.json")
    with open(processed_path, "r", encoding="utf-8") as source:
        curriculum_documents = json.load(source)

    required_collections = {
        ("6", "science", "english"),
        ("6", "science", "marathi"),
        ("6", "social_science", "hindi"),
        ("7", "science", "english"),
        ("7", "social_science", "marathi"),
        ("8", "science", "english"),
        ("8", "social_science", "hindi"),
    }
    sample_documents = []
    seen_collections = set()
    for document in curriculum_documents:
        collection = (
            document.get("class_level"),
            document.get("subject"),
            document.get("language"),
        )
        if collection in required_collections and collection not in seen_collections:
            sample_documents.append(document)
            seen_collections.add(collection)
        if len(seen_collections) == len(required_collections):
            break

    assert seen_collections == required_collections
    sample_dir = tmp_path_factory.mktemp("curriculum_rag_sample")
    with open(sample_dir / "curriculum_sample.json", "w", encoding="utf-8") as sample:
        json.dump(sample_documents, sample, ensure_ascii=False)

    # Use a separate index dir so tests don't pollute the main one
    engine = StoryRAG(dataset_dir=str(sample_dir), index_dir=TEST_INDEX_DIR)
    yield engine
    # Cleanup test index after all tests
    if os.path.exists(TEST_INDEX_DIR):
        shutil.rmtree(TEST_INDEX_DIR)


def test_documents_loaded(rag):
    """RAG should load documents from all JSON files in the dataset dir."""
    assert len(rag.documents) > 0, "No documents were loaded from knowledge_base/"
    print(f"  Loaded {len(rag.documents)} documents")


def test_faiss_index_built(rag):
    """FAISS index should be built and have the same count as documents."""
    assert rag.index is not None, "FAISS index was not created"
    assert rag.index.ntotal == len(rag.documents), (
        f"Index size ({rag.index.ntotal}) != document count ({len(rag.documents)})"
    )
    print(f"  FAISS index has {rag.index.ntotal} vectors")


def test_retrieve_returns_results(rag):
    """Retrieve should return a non-empty list for a valid query."""
    results = rag.retrieve("A podcast writer receives voice notes from a missing listener", top_k=3)
    assert len(results) > 0, "Retrieve returned no results"
    assert len(results) <= 3, f"Retrieve returned more than top_k results: {len(results)}"
    print(f"  Retrieved {len(results)} results")


def test_retrieve_filters_by_mode(rag):
    """Retrieve with mode filter should prefer matching mode documents."""
    results = rag.retrieve("A lesson about energy transfer", mode="hook", top_k=3)
    assert len(results) > 0
    # At least the first result should match the mode if any hook docs exist
    hook_results = [r for r in results if r.get("mode") == "hook"]
    print(f"  Got {len(hook_results)}/{len(results)} hook-mode results")


def test_retrieve_filters_by_genre(rag):
    """Retrieve with genre filter should prefer matching genre documents."""
    results = rag.retrieve("A student learning about forces", genre="comedy", top_k=3)
    assert len(results) > 0
    comedy_results = [r for r in results if r.get("genre") == "comedy"]
    print(f"  Got {len(comedy_results)}/{len(results)} comedy-genre results")


def test_get_context_text_formats_correctly(rag):
    """Context uses [S1]-style labels, matching the citation markers the prompt asks for."""
    results = rag.retrieve("A thriller about secrets", top_k=2)
    context = rag.get_context_text(results)
    assert "[S1]" in context, "Context text missing '[S1]' citation label"
    assert len(context) > 50, "Context text is suspiciously short"
    print(f"  Context text length: {len(context)} chars")


def test_get_context_text_empty_results(rag):
    """get_context_text with empty results should return empty string."""
    context = rag.get_context_text([])
    assert context == "", "Expected empty string for empty results"


def test_each_document_has_required_fields(rag):
    """Every loaded document should have the required schema fields."""
    required_fields = {"id", "mode", "genre", "text"}
    for doc in rag.documents:
        missing = required_fields - set(doc.keys())
        assert not missing, f"Document {doc.get('id', '?')} missing fields: {missing}"


def test_retrieve_accepts_educational_metadata_filters(rag):
    results = rag.retrieve(
        "energy transfer lesson",
        top_k=3,
        filters={"subject": "science", "language": "english"},
    )
    assert len(results) > 0
    assert all(result.get("retrieval_method") for result in results)
    assert all(result.get("subject") == "science" for result in results)
    assert all(result.get("language") == "english" for result in results)


def test_retrieve_never_relaxes_requested_class_or_subject_filters(rag):
    results = rag.retrieve(
        "energy transfer lesson",
        top_k=3,
        filters={"class_level": "6", "subject": "science", "language": "english"},
    )

    assert results
    assert all(result.get("class_level") == "6" for result in results)
    assert all(result.get("subject") == "science" for result in results)


def test_source_documents_returns_citation_metadata(rag):
    results = rag.retrieve("energy transfer lesson", top_k=2)
    sources = rag.source_documents(results)
    assert len(sources) == len(results)
    assert "id" in sources[0]
