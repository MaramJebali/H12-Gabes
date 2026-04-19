from sentence_transformers import CrossEncoder


RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self, model_name: str = RERANKER_MODEL_NAME):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, retrieved_items: list, top_k: int = 5):
        """
        retrieved_items = list de dicts de la forme :
        {
            "chunk": ...,
            "vector_score": ...,
            "bm25_score": ...,
            "final_score": ...
        }
        """
        if not retrieved_items:
            return []

        pairs = [(query, item["chunk"].text) for item in retrieved_items]
        rerank_scores = self.model.predict(pairs)

        reranked = []
        for item, rerank_score in zip(retrieved_items, rerank_scores):
            item_copy = item.copy()
            item_copy["rerank_score"] = float(rerank_score)
            reranked.append(item_copy)

        reranked = sorted(reranked, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]