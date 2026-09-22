"""Okapi BM25 sparse retrieval, implemented directly rather than pulled in.

Two reasons this isn't a dependency:
  1. The corpus is ~8k chunks, where a plain inverted index is instant.
  2. The tokenizer has to handle Devanagari. Python's `re` module treats `\\w`
     as Unicode-aware by default, so one regex covers English, Hindi and
     Marathi -- but relying on that explicitly is safer than inheriting some
     library's English-centric default tokenizer.

BM25 earns its place next to dense retrieval because embeddings blur exact
technical vocabulary. A learner asking about "प्रकाश संश्लेषण" or
"photosynthesis" should match the chunk containing that exact term, even when
a dozen chunks are semantically about plants.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Sequence

import numpy as np

# `\w` alone is NOT enough for Devanagari: matras, anusvara and virama are
# Unicode marks (categories Mn/Mc), which `\w` excludes. Relying on it shreds
# "प्रकाश" into 7 single-character fragments and makes sparse retrieval useless
# on the ~2/3 of this corpus that is Hindi or Marathi.
#
# U+0900-U+097F is the Devanagari block; U+0964-U+0965 (danda, double danda)
# are punctuation and are deliberately excluded so sentence marks don't get
# glued onto tokens. Adding another script means extending this range.
_TOKEN = re.compile(r"[\wऀ-ॣ०-ॿ]+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    return _TOKEN.findall(str(text or "").lower())


class BM25:
    """Okapi BM25 over a fixed corpus.

    k1 controls term-frequency saturation, b controls length normalisation.
    The 1.5 / 0.75 defaults are the standard Robertson values.
    """

    def __init__(self, corpus: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(text) for text in corpus]
        self.doc_len = np.array([len(t) for t in self.doc_tokens], dtype=np.float32)
        self.n_docs = len(self.doc_tokens)
        self.avgdl = float(self.doc_len.mean()) if self.n_docs else 0.0

        # term -> {doc_index: term_frequency}
        self.postings: Dict[str, Dict[int, int]] = {}
        for idx, tokens in enumerate(self.doc_tokens):
            for term, freq in Counter(tokens).items():
                self.postings.setdefault(term, {})[idx] = freq

        self.idf: Dict[str, float] = {}
        for term, posting in self.postings.items():
            df = len(posting)
            # Probabilistic IDF with the +1 smoothing that keeps common terms
            # from going negative.
            self.idf[term] = math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> np.ndarray:
        """Score every document against the query. Returns shape (n_docs,)."""
        out = np.zeros(self.n_docs, dtype=np.float32)
        if not self.n_docs or self.avgdl == 0:
            return out

        for term in tokenize(query):
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = self.idf[term]
            for doc_idx, freq in posting.items():
                denom = freq + self.k1 * (1.0 - self.b + self.b * self.doc_len[doc_idx] / self.avgdl)
                out[doc_idx] += idf * (freq * (self.k1 + 1.0)) / denom
        return out

    def top_n(self, query: str, n: int, allowed: Sequence[int] = None) -> List[int]:
        """Return the indices of the n best-scoring documents, best first.

        `allowed` restricts scoring to a candidate subset (used when metadata
        filters have already narrowed the corpus).
        """
        scores = self.scores(query)
        if allowed is not None:
            mask = np.full(self.n_docs, False)
            mask[np.asarray(list(allowed), dtype=int)] = True
            scores = np.where(mask, scores, -np.inf)

        n = min(n, self.n_docs)
        if n <= 0:
            return []
        idx = np.argpartition(-scores, n - 1)[:n]
        idx = idx[np.argsort(-scores[idx])]
        return [int(i) for i in idx if scores[i] > 0]
