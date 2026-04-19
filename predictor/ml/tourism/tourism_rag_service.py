import numpy as np
import faiss

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

from .data_loader import load_all_tourism_data
from .profile_builder import profile_to_text


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def simple_tokenize(text: str):
    return (
        text.lower()
        .replace(",", " ")
        .replace(".", " ")
        .replace(":", " ")
        .replace(";", " ")
        .replace("(", " ")
        .replace(")", " ")
        .split()
    )


class TourismRAGService:
    def __init__(self):
        data = load_all_tourism_data()
        self.places = data["places"]
        self.services = data["services"]
        self.sources = data["sources"]

        self.chunks = self._build_chunks(self.places)

        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.reranker = CrossEncoder(RERANKER_MODEL_NAME)

        texts = [chunk["text"] for chunk in self.chunks]
        embeddings = self.embedder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype("float32")

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

        self.tokenized_corpus = [simple_tokenize(chunk["text"]) for chunk in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _build_chunks(self, places):
        chunks = []

        for place in places:
            place_id = place["place_id"]
            name = place.get("name", "")
            category = place.get("category", "")
            short_description = place.get("short_description", "")
            long_description = place.get("long_description", "")
            story = place.get("storytelling_seed", "")
            tags = ", ".join(place.get("ai_tags", []))
            experiences = ", ".join(place.get("experience_types", []))
            econ = ", ".join(place.get("economic_potential_tags", []))

            chunks.append({
                "chunk_id": f"{place_id}_overview",
                "place_id": place_id,
                "place_name": name,
                "chunk_type": "overview",
                "text": (
                    f"Nom: {name}. "
                    f"Catégorie: {category}. "
                    f"Description: {short_description}. "
                    f"Expériences: {experiences}. "
                    f"Tags: {tags}."
                )
            })

            if long_description:
                chunks.append({
                    "chunk_id": f"{place_id}_details",
                    "place_id": place_id,
                    "place_name": name,
                    "chunk_type": "details",
                    "text": long_description
                })

            if story:
                chunks.append({
                    "chunk_id": f"{place_id}_story",
                    "place_id": place_id,
                    "place_name": name,
                    "chunk_type": "story",
                    "text": story
                })

            if econ:
                chunks.append({
                    "chunk_id": f"{place_id}_economic",
                    "place_id": place_id,
                    "place_name": name,
                    "chunk_type": "economic",
                    "text": (
                        f"Potentiel économique local de {name}: {econ}."
                    )
                })

        return chunks

    def build_query(self, profile, recommended_places=None):
        profile_text = profile_to_text(profile)

        top_place_names = ""
        if recommended_places:
            top_place_names = " ".join([p.name for p in recommended_places[:5]])

        query = (
            f"{profile_text} "
            f"Lieux recommandés: {top_place_names}. "
            f"Patrimoine culture nature gastronomie artisanat expérience immersive Gabès."
        )
        return query

    def vector_search(self, query: str, top_k: int = 8):
        query_vec = self.embedder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype("float32")

        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append({
                "chunk": chunk,
                "vector_score": float(score),
                "bm25_score": 0.0,
                "hybrid_score": float(score) * 0.6
            })
        return results

    def bm25_search(self, query: str, top_k: int = 8):
        tokenized_query = simple_tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        ranked_idx = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        max_score = max([scores[i] for i in ranked_idx], default=1.0)
        if max_score == 0:
            max_score = 1.0

        results = []
        for idx in ranked_idx:
            chunk = self.chunks[idx]
            normalized = float(scores[idx] / max_score)
            results.append({
                "chunk": chunk,
                "vector_score": 0.0,
                "bm25_score": normalized,
                "hybrid_score": normalized * 0.4
            })
        return results

    def hybrid_retrieve(self, query: str, top_k: int = 8):
        vector_results = self.vector_search(query, top_k=top_k)
        bm25_results = self.bm25_search(query, top_k=top_k)

        merged = {}

        for item in vector_results:
            key = item["chunk"]["chunk_id"]
            merged[key] = item

        for item in bm25_results:
            key = item["chunk"]["chunk_id"]
            if key in merged:
                merged[key]["bm25_score"] = item["bm25_score"]
                merged[key]["hybrid_score"] += item["hybrid_score"]
            else:
                merged[key] = item

        ranked = sorted(
            merged.values(),
            key=lambda x: x["hybrid_score"],
            reverse=True
        )[:top_k]

        return ranked

    def rerank(self, query: str, retrieved_items: list, top_k: int = 5):
        if not retrieved_items:
            return []

        pairs = [(query, item["chunk"]["text"]) for item in retrieved_items]
        rerank_scores = self.reranker.predict(pairs)

        reranked = []
        for item, rerank_score in zip(retrieved_items, rerank_scores):
            item_copy = item.copy()
            item_copy["rerank_score"] = float(rerank_score)
            reranked.append(item_copy)

        reranked = sorted(reranked, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

    def link_services(self, reranked_chunks: list):
        linked_services = []
        seen = set()

        place_ids = {item["chunk"]["place_id"] for item in reranked_chunks}

        for service in self.services:
            nearby = set(service.get("nearby_place_ids", []))
            if place_ids.intersection(nearby):
                if service["service_id"] not in seen:
                    linked_services.append(service)
                    seen.add(service["service_id"])

        return linked_services

    def build_grounded_context(self, profile, recommended_places: list, top_k: int = 5):
        query = self.build_query(profile, recommended_places=recommended_places)
        retrieved = self.hybrid_retrieve(query, top_k=8)
        reranked = self.rerank(query, retrieved, top_k=top_k)
        linked_services = self.link_services(reranked)

        context_chunks = []
        cited_places = []

        for item in reranked:
            chunk = item["chunk"]
            context_chunks.append({
                "place_name": chunk["place_name"],
                "place_id": chunk["place_id"],
                "chunk_type": chunk["chunk_type"],
                "text": chunk["text"],
                "vector_score": item.get("vector_score", 0.0),
                "bm25_score": item.get("bm25_score", 0.0),
                "hybrid_score": item.get("hybrid_score", 0.0),
                "rerank_score": item.get("rerank_score", 0.0),
            })

            if chunk["place_name"] not in cited_places:
                cited_places.append(chunk["place_name"])

        return {
            "query": query,
            "retrieved_chunks": context_chunks,
            "cited_places": cited_places,
            "linked_services": linked_services,
        }