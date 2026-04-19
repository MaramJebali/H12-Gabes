import faiss
import numpy as np
from typing import List
from .schemas import RAGChunk, RetrievalResult


class FAISSVectorStore:
    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatIP(dimension)  # cosine approx si vecteurs normalisés
        self.chunks: List[RAGChunk] = []
        self.embeddings = None

    def add(self, chunks: List[RAGChunk], embeddings: np.ndarray):
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype("float32")

        self.index.add(embeddings)
        self.chunks.extend(chunks)

        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

    def search(self, query_embedding, top_k: int = 5) -> List[RetrievalResult]:
        query_embedding = np.array([query_embedding], dtype="float32")
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(
                RetrievalResult(
                    chunk=self.chunks[idx],
                    score=float(score),
                    source="vector",
                )
            )
        return results