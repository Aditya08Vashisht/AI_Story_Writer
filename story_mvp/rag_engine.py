import os
import json
from hashlib import sha256
import re
from typing import List, Dict, Optional
import faiss
from sentence_transformers import SentenceTransformer

from story_mvp.bm25 import BM25
from story_mvp.reranker import CrossEncoderReranker, reciprocal_rank_fusion, rerank_enabled


def _resolve_device() -> str:
    """Pick the embedding device. STORYTUTOR_DEVICE overrides autodetection."""
    override = os.environ.get("STORYTUTOR_DEVICE", "").strip().lower()
    if override:
        return override
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class StoryRAG:
    def __init__(self, dataset_dir: str, index_dir: str = None, model_name: str = None):
        self.dataset_dir = dataset_dir
        self.index_dir = index_dir or os.path.join(os.path.dirname(__file__), "faiss_index")
        self.model_name = model_name or os.environ.get("STORYTUTOR_EMBEDDING_MODEL", "BAAI/bge-m3")
        self.device = _resolve_device()
        self.embed_batch_size = int(os.environ.get("STORYTUTOR_EMBED_BATCH", "64"))
        self.max_seq_length = int(os.environ.get("STORYTUTOR_MAX_SEQ_LEN", "1024"))
        # How many fused candidates survive to the rerank stage. Large enough
        # that a correct chunk ranked poorly by either retriever alone still
        # gets a second opinion; small enough that the cross-encoder stays cheap.
        self.candidate_pool = int(os.environ.get("STORYTUTOR_CANDIDATE_POOL", "30"))
        self.model = None
        self.index = None
        self.bm25: Optional[BM25] = None
        self.reranker = CrossEncoderReranker(device=self.device) if rerank_enabled() else None
        self.documents: List[Dict] = []
        self._initialize()

    def _initialize(self):
        """Loads data and index, creating index if it doesn't exist."""
        self._load_documents()
        
        os.makedirs(self.index_dir, exist_ok=True)
        index_slug = re.sub(r"[^A-Za-z0-9]+", "_", self.model_name).strip("_").lower()
        index_path = os.path.join(self.index_dir, f"{index_slug}.index")
        signature_path = os.path.join(self.index_dir, f"{index_slug}.signature")
        dataset_signature = self._dataset_signature()

        if self.documents:
            print(f"Building BM25 index over {len(self.documents)} documents...")
            self.bm25 = BM25([doc.get("text", "") for doc in self.documents])

        print(f"Loading embedding model {self.model_name} on {self.device}...")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        # Chunks cap at 1800 characters, so bge-m3's 8192-token default would pad
        # every batch to many times the length any chunk actually needs. Only
        # ever shrink this, never raise it past the model's own native limit --
        # e.g. all-MiniLM-L6-v2 (used in tests) caps at 512 and errors on tensors
        # sized for a longer sequence than its position embeddings support.
        if self.max_seq_length:
            self.model.max_seq_length = min(self.max_seq_length, self.model.max_seq_length)

        if os.path.exists(index_path):
            stored_index = faiss.read_index(index_path)
            stored_signature = self._read_signature(signature_path)
            # Rebuild if document count or content changed.
            if stored_index.ntotal == len(self.documents) and stored_signature == dataset_signature:
                print("Loading existing FAISS index...")
                self.index = stored_index
            else:
                print("Dataset changed. Rebuilding index...")
                self._build_index(index_path, signature_path, dataset_signature)
        else:
            print("Building new FAISS index...")
            self._build_index(index_path, signature_path, dataset_signature)

    def _load_documents(self):
        """Loads all JSON files from the dataset directory."""
        if not os.path.exists(self.dataset_dir):
            print(f"Warning: Dataset directory {self.dataset_dir} does not exist.")
            return

        for root, _, filenames in os.walk(self.dataset_dir):
            for filename in filenames:
                if not filename.endswith(".json"):
                    continue
                filepath = os.path.join(root, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        category = os.path.relpath(root, self.dataset_dir).split(os.sep)[0]
                        for doc in data:
                            if isinstance(doc, dict):
                                doc.setdefault("source_category", category)
                                self.documents.append(doc)

    def _build_index(self, index_path: str, signature_path: str, dataset_signature: str):
        """Embeds all documents and saves the FAISS index."""
        if not self.documents:
            print("No documents found to index.")
            return

        texts = [doc["text"] for doc in self.documents]
        print(f"Embedding {len(texts)} documents on {self.device} (batch={self.embed_batch_size})...")
        import numpy as np
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            batch_size=self.embed_batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        dimension = embeddings.shape[1]
        # Using Inner Product since vectors are normalized (equivalent to cosine similarity)
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        
        faiss.write_index(self.index, index_path)
        with open(signature_path, "w", encoding="utf-8") as f:
            f.write(dataset_signature)
        print("FAISS index saved.")

    def _dataset_signature(self) -> str:
        payload = json.dumps(self.documents, sort_keys=True, ensure_ascii=False)
        return sha256(payload.encode("utf-8")).hexdigest()

    def _read_signature(self, signature_path: str) -> str:
        if not os.path.exists(signature_path):
            return ""
        with open(signature_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    def retrieve(
        self,
        query: str,
        mode: str = None,
        genre: str = None,
        top_k: int = 3,
        filters: Optional[Dict[str, str]] = None,
    ) -> List[Dict]:
        """Hybrid retrieval: dense + BM25, fused with RRF, then cross-encoder reranked."""
        if not self.index or not self.documents:
            return []

        import numpy as np
        query_embedding = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(query_embedding)

        active_filters = self._supported_filters(filters or {})
        supported_mode = mode if any(doc.get("mode") == mode for doc in self.documents) else None
        supported_genre = genre if any(doc.get("genre") == genre for doc in self.documents) else None
        # FAISS does not offer metadata predicates for IndexFlat. When metadata
        # filters are active, retrieve all candidates before filtering so a
        # correct but lower-ranked source cannot be missed.
        fetch_k = len(self.documents) if active_filters else min(len(self.documents), max(top_k * 8, 20))
        distances, indices = self.index.search(query_embedding, fetch_k)

        # --- stage 1: dense ranking (filtered, best first) ---
        dense_ranked: List[int] = []
        dense_scores: Dict[int, float] = {}
        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            if not self._passes(self.documents[idx], supported_mode, supported_genre, active_filters):
                continue
            dense_ranked.append(int(idx))
            dense_scores[int(idx)] = float(distance)

        # --- stage 2: BM25 ranking over the same filtered candidate set ---
        allowed = {
            i for i, doc in enumerate(self.documents)
            if self._passes(doc, supported_mode, supported_genre, active_filters)
        }
        sparse_ranked = self.bm25.top_n(query, self.candidate_pool, allowed=allowed) if self.bm25 else []

        # --- stage 3: reciprocal rank fusion ---
        fused = reciprocal_rank_fusion([dense_ranked, sparse_ranked])[: self.candidate_pool]
        if not fused:
            fused = dense_ranked[: self.candidate_pool]

        candidates = []
        for rank, idx in enumerate(fused):
            doc = dict(self.documents[idx])
            doc["dense_score"] = dense_scores.get(idx)
            doc["fusion_rank"] = rank + 1
            doc["retrieval_score"] = dense_scores.get(idx, 1.0 / (60 + rank + 1))
            doc["retrieval_method"] = "hybrid_rrf"
            candidates.append(doc)

        # --- stage 4: cross-encoder rerank (opt-in) ---
        if self.reranker is not None:
            reranked = self.reranker.rerank(query, candidates, top_k)
            if reranked is not None:
                for doc in reranked:
                    doc["retrieval_method"] = "hybrid_rrf_rerank"
                    doc["retrieval_score"] = doc["rerank_score"]
                return reranked

        return candidates[:top_k]

    def _passes(self, doc: Dict, supported_mode, supported_genre, active_filters) -> bool:
        """Shared filter predicate for both the dense and the sparse stage."""
        if supported_mode and doc.get("mode") != supported_mode and doc.get("mode"):
            return False
        if supported_genre and doc.get("genre") != supported_genre and doc.get("genre"):
            return False
        return self._matches_filters(doc, active_filters)

    def get_context_text(self, results: List[Dict]) -> str:
        """Formats the retrieved documents into a string for the prompt."""
        if not results:
            return ""
        
        context_parts = []
        for i, doc in enumerate(results, 1):
            source = self.source_summary(doc)
            # Label as [S1], [S2]... so the marker the model is asked to emit
            # is literally the label it sees, with no translation step.
            context_parts.append(f"[S{i}] ({source}):\n{doc['text']}\n")
        
        return "\n".join(context_parts)

    def source_summary(self, doc: Dict) -> str:
        parts = []
        for key in ("class_level", "subject", "language", "chapter", "page_start"):
            value = doc.get(key) or doc.get("metadata", {}).get(key)
            if value:
                parts.append(f"{key}={value}")
        source_file = doc.get("source_file") or doc.get("metadata", {}).get("source_file")
        if source_file:
            parts.append(f"source={os.path.basename(source_file)}")
        return ", ".join(parts) or doc.get("id", "unknown source")

    def source_documents(self, results: List[Dict]) -> List[Dict[str, object]]:
        sources = []
        for doc in results:
            sources.append(
                {
                    "id": doc.get("id"),
                    "class_level": doc.get("class_level") or doc.get("metadata", {}).get("class_level"),
                    "subject": doc.get("subject") or doc.get("metadata", {}).get("subject"),
                    "language": doc.get("language") or doc.get("metadata", {}).get("language"),
                    "chapter": doc.get("chapter") or doc.get("metadata", {}).get("chapter"),
                    "page_start": doc.get("page_start"),
                    "page_end": doc.get("page_end"),
                    "source_file": doc.get("source_file") or doc.get("metadata", {}).get("source_file"),
                    "retrieval_score": doc.get("retrieval_score"),
                    "retrieval_method": doc.get("retrieval_method"),
                    # Surface the per-stage scores so a citation shows *why* it
                    # ranked where it did, not just that it was retrieved.
                    "dense_score": doc.get("dense_score"),
                    "fusion_rank": doc.get("fusion_rank"),
                    "rerank_score": doc.get("rerank_score"),
                }
            )
        return sources

    def _matches_filters(self, doc: Dict, filters: Dict[str, str]) -> bool:
        for key, expected in filters.items():
            if not expected:
                continue
            actual = doc.get(key) or doc.get("metadata", {}).get(key)
            if actual is None:
                continue
            if str(actual).lower() != str(expected).lower():
                return False
        return True

    def _supported_filters(self, filters: Dict[str, str]) -> Dict[str, str]:
        """Keep only filters represented by the ingested corpus.

        Chapter and topic labels supplied by a learner may use natural language
        while NCERT files often use numeric identifiers. In that case the
        request text still guides retrieval, but the unavailable exact filter
        does not suppress every curriculum source.
        """
        active_filters = {}
        for key, expected in filters.items():
            if not expected:
                continue
            if any(
                str(doc.get(key) or doc.get("metadata", {}).get(key) or "").lower()
                == str(expected).lower()
                for doc in self.documents
            ):
                active_filters[key] = expected
        return active_filters

    def _hybrid_score(self, vector_score: float, query: str, doc: Dict) -> float:
        """Combine dense semantic similarity and lexical overlap for reranking."""
        return (0.85 * vector_score) + (0.15 * self._keyword_score(query, doc))

    def _keyword_score(self, query: str, doc: Dict) -> float:
        query_terms = set(re.findall(r"\w+", query.lower()))
        text_terms = set(re.findall(r"\w+", str(doc.get("text", "")).lower()))
        if not query_terms or not text_terms:
            return 0.0
        return len(query_terms & text_terms) / max(len(query_terms), 1)


class RetrievalService(StoryRAG):
    """Educational retrieval service name for the StoryTutor-MM architecture."""

    def retrieve(self, *args, **kwargs) -> List[Dict]:
        """Retrieve in the learner's language, then add English support sources."""
        top_k = int(kwargs.get("top_k", 3))
        native_results = super().retrieve(*args, **kwargs)
        filters = dict(kwargs.get("filters") or {})
        language = str(filters.get("language") or "").lower()

        if language not in {"hindi", "marathi"} or len(native_results) >= top_k:
            return native_results

        fallback_filters = dict(filters)
        fallback_filters["language"] = "english"
        fallback_kwargs = dict(kwargs)
        fallback_kwargs["filters"] = fallback_filters
        fallback_kwargs["top_k"] = top_k - len(native_results)
        english_results = super().retrieve(*args, **fallback_kwargs)
        for document in english_results:
            document["retrieval_language_fallback"] = "english"

        combined = native_results + [
            document
            for document in english_results
            if document.get("id") not in {result.get("id") for result in native_results}
        ]
        return combined[:top_k]
