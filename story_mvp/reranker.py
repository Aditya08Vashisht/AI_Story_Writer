"""Cross-encoder reranking -- the largest single quality win in the pipeline.

Dense and sparse retrieval both score a query against a document *independently*
(bi-encoder: two separate vectors, compared afterwards). A cross-encoder reads
the query and the document together in one forward pass, so it can judge actual
relevance rather than vector proximity. It is far too slow to run over 8k
chunks, which is why it only ever sees the top ~30 candidates that hybrid
retrieval already surfaced.

Opt-in by design: `BAAI/bge-reranker-v2-m3` is a ~2.2 GB download, and the test
suite must stay fast and offline. Enable with STORYTUTOR_ENABLE_RERANK=1.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional


def rerank_enabled() -> bool:
    return os.environ.get("STORYTUTOR_ENABLE_RERANK", "").strip().lower() in {"1", "true", "yes"}


class CrossEncoderReranker:
    """Lazily-loaded multilingual cross-encoder.

    Loading is deferred to the first rerank call so that constructing a
    retrieval engine never pays the model-download cost unless reranking is
    actually used.
    """

    def __init__(self, model_name: str = None, device: str = None, batch_size: int = None):
        self.model_name = model_name or os.environ.get(
            "STORYTUTOR_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"
        )
        self.device = device
        self.batch_size = batch_size or int(os.environ.get("STORYTUTOR_RERANK_BATCH", "32"))
        self._model = None
        self._load_failed = False

    def _ensure_model(self):
        if self._model is not None or self._load_failed:
            return
        try:
            from sentence_transformers import CrossEncoder

            print(f"Loading reranker {self.model_name} on {self.device or 'auto'}...")
            kwargs = {"max_length": 512}
            if self.device:
                kwargs["device"] = self.device
            self._model = CrossEncoder(self.model_name, **kwargs)
        except Exception as exc:
            # A missing reranker degrades ranking quality but must never take
            # down retrieval, which still has usable hybrid results.
            print(f"Warning: reranker unavailable, falling back to fusion order - {exc}")
            self._load_failed = True

    @property
    def available(self) -> bool:
        self._ensure_model()
        return self._model is not None

    def rerank(self, query: str, documents: List[Dict], top_k: int) -> Optional[List[Dict]]:
        """Re-order documents by cross-encoder relevance. None if unavailable."""
        self._ensure_model()
        if self._model is None or not documents:
            return None

        pairs = [(query, str(doc.get("text", ""))) for doc in documents]
        try:
            scores = self._model.predict(pairs, batch_size=self.batch_size)
        except Exception as exc:
            print(f"Warning: rerank failed, keeping fusion order - {exc}")
            return None

        scored = []
        for doc, score in zip(documents, scores):
            enriched = dict(doc)
            enriched["rerank_score"] = float(score)
            scored.append(enriched)

        scored.sort(key=lambda d: d["rerank_score"], reverse=True)
        return scored[:top_k]


def reciprocal_rank_fusion(ranked_lists: List[List[int]], k: int = 60) -> List[int]:
    """Fuse several ranked ID lists into one, best first.

    RRF replaces hand-tuned score blending (the old `0.85*cosine +
    0.15*overlap`). It only uses *rank position*, never the raw scores, so it
    is immune to dense and sparse scores living on completely different and
    unstable scales. k=60 is the standard damping constant from the original
    paper; it keeps any single list from dominating on its top hit alone.
    """
    fused: Dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return [doc_id for doc_id, _ in sorted(fused.items(), key=lambda kv: kv[1], reverse=True)]
