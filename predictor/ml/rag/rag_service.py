from typing import Dict
from .ingest import load_gabes_knowledge_base
from .chunker import build_project_chunks
from .embedder import Embedder
from .vector_store import FAISSVectorStore
from .bm25_index import BM25Index
from .reranker import Reranker
from .context_builder import build_grounded_context


class AdvancedRAGService:
    def __init__(self):
        self.projects = load_gabes_knowledge_base()
        self.chunks = build_project_chunks(self.projects)

        self.embedder = Embedder()
        texts = [chunk.text for chunk in self.chunks]
        embeddings = self.embedder.encode_texts(texts)

        self.vector_store = FAISSVectorStore(dimension=embeddings.shape[1])
        self.vector_store.add(self.chunks, embeddings)

        self.bm25_index = BM25Index(self.chunks)
        self.reranker = Reranker()

    def build_query(self, fused_analysis: Dict) -> str:
        forecast = fused_analysis.get("forecast", {})
        env = fused_analysis.get("environment_profile", {})

        alert = forecast.get("alert_level", "")
        dominant_factors = forecast.get("dominant_factors", [])
        cluster_profile = env.get("cluster_profile", "")
        cluster_summary = env.get("cluster_summary", "")

        parts = [
            "Gabès",
            str(alert),
            str(cluster_profile),
            str(cluster_summary),
            " ".join(dominant_factors) if isinstance(dominant_factors, list) else str(dominant_factors),
            "oasis eau biodiversité agriculture irrigation résilience restauration agroécologie",
        ]

        return " ".join([p for p in parts if p])

    def hybrid_search(self, fused_analysis: Dict, top_k: int = 8) -> Dict:
        query = self.build_query(fused_analysis)

        query_embedding = self.embedder.encode_query(query)
        vector_results = self.vector_store.search(query_embedding, top_k=top_k)
        bm25_results = self.bm25_index.search(query, top_k=top_k)

        merged = {}

        for result in vector_results:
            key = result.chunk.chunk_id
            merged[key] = {
                "chunk": result.chunk,
                "vector_score": result.score,
                "bm25_score": 0.0,
                "final_score": 0.6 * result.score,
            }

        max_bm25 = max([r.score for r in bm25_results], default=1.0)
        if max_bm25 == 0:
            max_bm25 = 1.0

        for result in bm25_results:
            key = result.chunk.chunk_id
            normalized_bm25 = result.score / max_bm25

            if key not in merged:
                merged[key] = {
                    "chunk": result.chunk,
                    "vector_score": 0.0,
                    "bm25_score": normalized_bm25,
                    "final_score": 0.4 * normalized_bm25,
                }
            else:
                merged[key]["bm25_score"] = normalized_bm25
                merged[key]["final_score"] += 0.4 * normalized_bm25

        ranked = sorted(merged.values(), key=lambda x: x["final_score"], reverse=True)[:top_k]

        return {
            "query": query,
            "results": ranked
        }

    def retrieve_and_rerank(self, fused_analysis: Dict, retrieve_k: int = 8, rerank_k: int = 5) -> Dict:
        search_output = self.hybrid_search(fused_analysis, top_k=retrieve_k)
        reranked = self.reranker.rerank(
            query=search_output["query"],
            retrieved_items=search_output["results"],
            top_k=rerank_k
        )

        grounded_context = build_grounded_context(fused_analysis, reranked)

        return {
            "query": search_output["query"],
            "retrieved": search_output["results"],
            "reranked": reranked,
            "grounded_context": grounded_context,
        }