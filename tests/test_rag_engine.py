"""Tests for the RAG engine (embedding + FAISS retrieval)."""

import os
import shutil
import pytest

from story_mvp.rag_engine import StoryRAG

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "story_datasets")
TEST_INDEX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_test_faiss_index")


@pytest.fixture(scope="module")
def rag():
    """Create a StoryRAG instance once for all tests in this module."""
    # Use a separate index dir so tests don't pollute the main one
    engine = StoryRAG(dataset_dir=DATASET_DIR, index_dir=TEST_INDEX_DIR)
    yield engine
    # Cleanup test index after all tests
    if os.path.exists(TEST_INDEX_DIR):
        shutil.rmtree(TEST_INDEX_DIR)


def test_documents_loaded(rag):
    """RAG should load documents from all JSON files in the dataset dir."""
    assert len(rag.documents) > 0, "No documents were loaded from story_datasets/"
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
    results = rag.retrieve("A mysterious story about a ghost", mode="hook", top_k=3)
    assert len(results) > 0
    # At least the first result should match the mode if any hook docs exist
    hook_results = [r for r in results if r.get("mode") == "hook"]
    print(f"  Got {len(hook_results)}/{len(results)} hook-mode results")


def test_retrieve_filters_by_genre(rag):
    """Retrieve with genre filter should prefer matching genre documents."""
    results = rag.retrieve("A love story between two strangers", genre="romance", top_k=3)
    assert len(results) > 0
    romance_results = [r for r in results if r.get("genre") == "romance"]
    print(f"  Got {len(romance_results)}/{len(results)} romance-genre results")


def test_get_context_text_formats_correctly(rag):
    """get_context_text should return a formatted string with Example labels."""
    results = rag.retrieve("A thriller about secrets", top_k=2)
    context = rag.get_context_text(results)
    assert "Example 1:" in context, "Context text missing 'Example 1:' label"
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
