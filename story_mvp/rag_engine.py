import os
import json
from typing import List, Dict
import faiss
from sentence_transformers import SentenceTransformer


class StoryRAG:
    def __init__(self, dataset_dir: str, index_dir: str = None):
        self.dataset_dir = dataset_dir
        self.index_dir = index_dir or os.path.join(os.path.dirname(__file__), "faiss_index")
        self.model_name = "all-MiniLM-L6-v2"
        self.model = None
        self.index = None
        self.documents: List[Dict] = []
        self._initialize()

    def _initialize(self):
        """Loads data and index, creating index if it doesn't exist."""
        self._load_documents()
        
        os.makedirs(self.index_dir, exist_ok=True)
        index_path = os.path.join(self.index_dir, "story_rag.index")

        print("Loading embedding model...")
        self.model = SentenceTransformer(self.model_name)

        if os.path.exists(index_path):
            stored_index = faiss.read_index(index_path)
            # Rebuild if document count changed (datasets were updated)
            if stored_index.ntotal == len(self.documents):
                print("Loading existing FAISS index...")
                self.index = stored_index
            else:
                print(f"Dataset size changed ({stored_index.ntotal} -> {len(self.documents)}). Rebuilding index...")
                self._build_index(index_path)
        else:
            print("Building new FAISS index...")
            self._build_index(index_path)

    def _load_documents(self):
        """Loads all JSON files from the dataset directory."""
        if not os.path.exists(self.dataset_dir):
            print(f"Warning: Dataset directory {self.dataset_dir} does not exist.")
            return

        for filename in os.listdir(self.dataset_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.dataset_dir, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.documents.extend(data)

    def _build_index(self, index_path: str):
        """Embeds all documents and saves the FAISS index."""
        if not self.documents:
            print("No documents found to index.")
            return

        texts = [doc["text"] for doc in self.documents]
        print(f"Embedding {len(texts)} documents...")
        import numpy as np
        embeddings = self.model.encode(texts, convert_to_numpy=True).astype(np.float32)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        dimension = embeddings.shape[1]
        # Using Inner Product since vectors are normalized (equivalent to cosine similarity)
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        
        faiss.write_index(self.index, index_path)
        print("FAISS index saved.")

    def retrieve(self, query: str, mode: str = None, genre: str = None, top_k: int = 3) -> List[Dict]:
        """Retrieves top-K relevant documents for the given query."""
        if not self.index or not self.documents:
            return []

        import numpy as np
        query_embedding = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(query_embedding)
        
        # Retrieve more than top_k initially to allow for filtering
        fetch_k = min(len(self.documents), max(top_k * 5, 10))
        distances, indices = self.index.search(query_embedding, fetch_k)
        
        results = []
        for idx in indices[0]:
            if idx == -1:
                continue
            doc = self.documents[idx]
            
            # Simple metadata filtering
            if mode and doc.get("mode") != mode and doc.get("mode"):
                continue
            if genre and doc.get("genre") != genre and doc.get("genre"):
                continue
                
            results.append(doc)
            if len(results) == top_k:
                break
                
        # If we didn't find enough matches after filtering, just take the top semantic matches
        if len(results) < top_k:
            for idx in indices[0]:
                if idx == -1:
                    continue
                doc = self.documents[idx]
                if doc not in results:
                    results.append(doc)
                if len(results) == top_k:
                    break

        return results

    def get_context_text(self, results: List[Dict]) -> str:
        """Formats the retrieved documents into a string for the prompt."""
        if not results:
            return ""
        
        context_parts = []
        for i, doc in enumerate(results, 1):
            context_parts.append(f"Example {i}:\n{doc['text']}\n")
        
        return "\n".join(context_parts)
