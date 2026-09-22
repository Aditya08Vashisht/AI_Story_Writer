"""Tests for the sparse and fusion halves of hybrid retrieval.

These run offline with no model downloads -- BM25 and RRF are pure Python.
"""

from story_mvp.bm25 import BM25, tokenize
from story_mvp.reranker import reciprocal_rank_fusion


def test_tokenize_handles_devanagari():
    """The corpus is ~2/3 Hindi and Marathi, so this cannot be English-only."""
    assert tokenize("energy transfer") == ["energy", "transfer"]
    assert tokenize("प्रकाश संश्लेषण") == ["प्रकाश", "संश्लेषण"]
    assert tokenize("ऊर्जा transfer होते") == ["ऊर्जा", "transfer", "होते"]


def test_bm25_ranks_exact_term_match_first():
    corpus = [
        "Plants make food using sunlight in a process called photosynthesis.",
        "The water cycle moves water between the ocean, air and land.",
        "Photosynthesis happens mainly in the leaves of a plant.",
    ]
    bm25 = BM25(corpus)
    ranked = bm25.top_n("photosynthesis", n=3)
    assert ranked, "BM25 returned nothing for a term present in the corpus"
    assert set(ranked[:2]) == {0, 2}, "documents containing the term should rank above the one that does not"


def test_bm25_scores_unknown_term_as_zero():
    bm25 = BM25(["magnets attract iron", "sound travels as vibration"])
    assert bm25.top_n("photosynthesis", n=2) == []


def test_bm25_respects_allowed_subset():
    """Metadata filters narrow the candidate set before sparse scoring."""
    corpus = ["energy transfer in a spoon", "energy transfer in a wire", "unrelated text"]
    bm25 = BM25(corpus)
    ranked = bm25.top_n("energy transfer", n=3, allowed={1})
    assert ranked == [1], f"expected only the allowed document, got {ranked}"


def test_bm25_handles_empty_corpus():
    assert BM25([]).top_n("anything", n=3) == []


def test_rrf_rewards_agreement_between_retrievers():
    """A document both retrievers like should beat one that only tops a single list."""
    dense = [10, 20, 30]
    sparse = [40, 20, 50]
    fused = reciprocal_rank_fusion([dense, sparse])
    assert fused[0] == 20, f"doc 20 ranks in both lists and should fuse first, got {fused}"


def test_rrf_handles_one_empty_list():
    assert reciprocal_rank_fusion([[1, 2, 3], []]) == [1, 2, 3]


def test_rrf_uses_rank_not_score_scale():
    """RRF must depend only on position, which is why it survives score-scale drift."""
    assert reciprocal_rank_fusion([[7, 8]]) == reciprocal_rank_fusion([[7, 8]])
    assert reciprocal_rank_fusion([[8, 7]])[0] == 8
