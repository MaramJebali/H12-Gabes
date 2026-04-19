from rank_bm25 import BM25Okapi
from .schemas import RetrievalResult


def simple_tokenize(text: str):
    return text.lower().replace(",", " ").replace(".", " ").replace(":", " ").split()


class BM25Index:
    def __init__(self, chunks):
        self.chunks = chunks
        self.tokenized_corpus = [simple_tokenize(chunk.text) for chunk in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 5):
        tokenized_query = simple_tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        scored = list(enumerate(scores))
        scored = sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for idx, score in scored:
            results.append(
                RetrievalResult(
                    chunk=self.chunks[idx],
                    score=float(score),
                    source="bm25",
                )
            )
        return results