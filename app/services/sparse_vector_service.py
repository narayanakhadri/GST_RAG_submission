import threading

import numpy as np
from scipy.sparse import csc_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.models import RetrievedChunk


class SparseVectorIndex:
    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.documents: list[dict] = []
        self.matrix = None
        self._lock = threading.RLock()


    def fit(self, documents: list[dict]) -> None:
        with self._lock:
            self.documents = documents
            if not documents:
                self.matrix = None
                return

            texts = [doc.get("text", "") for doc in documents]
            try:
                self.matrix = self.vectorizer.fit_transform(texts)
            except ValueError:
                self.matrix = None
                return

            if self.matrix.shape[1] == 0:
                self.matrix = None


    def search(self, query: str, top_k: int = 20) -> list[RetrievedChunk]:
        """Return top-k chunks by TF-IDF cosine similarity."""
        with self._lock:
            if self.matrix is None or len(self.documents) == 0:
                return []

            query_vec = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, self.matrix).flatten()
            top_indices = similarities.argsort()[::-1][:top_k]

            results: list[RetrievedChunk] = []
            for idx in top_indices:
                score = float(similarities[idx])
                if score <= 0:
                    continue
                doc = self.documents[idx]
                results.append(
                    RetrievedChunk(
                        text=doc.get("text", ""),
                        source=doc.get("source", ""),
                        score=score,
                    )
                )
            return results

class BM25Index:
    """Okapi BM25 over the same documents as SparseVectorIndex, with the same fit/search interface.

    Unlike the TF-IDF index it keeps one-character tokens, so section/rule numbers such as "9" or "7"
    still match, and it saturates term frequency (k1) and normalises by document length (b), which stops
    long provisions from being rewarded for repetition.
    """

    TOKEN_PATTERN = r"(?u)\b\w+\b"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.vectorizer = CountVectorizer(stop_words="english", token_pattern=self.TOKEN_PATTERN)
        self.documents: list[dict] = []
        self.weights = None  # documents x terms BM25 weight matrix (CSC, so query columns slice fast)
        self._lock = threading.RLock()

    def fit(self, documents: list[dict]) -> None:
        with self._lock:
            self.documents = documents
            self.weights = None
            if not documents:
                return
            try:
                tf = self.vectorizer.fit_transform([doc.get("text", "") for doc in documents]).tocoo()
            except ValueError:
                return
            n_docs = tf.shape[0]
            doc_len = np.asarray(tf.sum(axis=1)).ravel()
            avg_len = doc_len.mean() or 1.0
            df = np.bincount(tf.col, minlength=tf.shape[1])
            idf = np.log1p((n_docs - df + 0.5) / (df + 0.5))
            denom = tf.data + self.k1 * (1 - self.b + self.b * doc_len[tf.row] / avg_len)
            data = idf[tf.col] * tf.data * (self.k1 + 1) / denom
            self.weights = csc_matrix((data, (tf.row, tf.col)), shape=tf.shape)

    def scores(self, query: str) -> np.ndarray:
        analyzer = self.vectorizer.build_analyzer()
        vocab = self.vectorizer.vocabulary_
        cols = sorted({vocab[t] for t in analyzer(query) if t in vocab})
        if not cols:
            return np.zeros(len(self.documents))
        return np.asarray(self.weights[:, cols].sum(axis=1)).ravel()

    def search(self, query: str, top_k: int = 20) -> list[RetrievedChunk]:
        """Return top-k chunks by BM25 score."""
        with self._lock:
            if self.weights is None:
                return []
            scores = self.scores(query)
            results: list[RetrievedChunk] = []
            for idx in scores.argsort()[::-1][:top_k]:
                if scores[idx] <= 0:
                    break
                doc = self.documents[idx]
                results.append(RetrievedChunk(text=doc.get("text", ""), source=doc.get("source", ""),
                                              score=float(scores[idx])))
            return results


def fuse_rrf(
    result_lists: list[list[RetrievedChunk]],
    rrf_k: int = 60,
    weights: list[float] | None = None,
) -> list[RetrievedChunk]:
    """Fuse multiple ranked result lists using Reciprocal Rank Fusion (optionally weighted per list)."""
    scores: dict[str, float] = {}
    meta: dict[str, dict] = {}

    for i, result_list in enumerate(result_lists):
        w = 1.0 if weights is None else weights[i]
        for rank, chunk in enumerate(result_list):
            key = chunk.text
            scores[key] = scores.get(key, 0.0) + w / (rrf_k + rank + 1)
            if key not in meta:
                meta[key] = {"text": chunk.text, "source": chunk.source}

    return [
        RetrievedChunk(text=text, source=meta[text]["source"], score=score)
        for text, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ]
