import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .schemas import RecommendedPlace
from .profile_builder import profile_to_text


def build_semantic_text(place: dict) -> str:
    return " ".join([
        place.get("name", ""),
        place.get("category", ""),
        " ".join(place.get("subcategories", [])),
        place.get("short_description", ""),
        place.get("long_description", ""),
        " ".join(place.get("experience_types", [])),
        " ".join(place.get("ai_tags", [])),
        place.get("storytelling_seed", "")
    ])


def compute_scores(profile, places):
    texts = [build_semantic_text(p) for p in places]
    profile_text = profile_to_text(profile)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts + [profile_text])

    place_vectors = X[:-1]
    profile_vector = X[-1]

    similarities = cosine_similarity(profile_vector, place_vectors)[0]

    results = []

    for i, place in enumerate(places):
        semantic_score = similarities[i]

        # profil fit
        profile_fit = place.get("visitor_profile_fit", {})
        interest_score = np.mean([
            profile_fit.get(i, 0.5) for i in profile.interests
        ]) if profile.interests else 0.5

        # budget
        budget_score = 1 if place.get("budget_level") == profile.budget else 0.6

        # durée
        duration_score = 1 if place.get("visit_duration_minutes", 0) < (profile.duration_days * 480) else 0.7

        # score global
        final_score = (
            0.4 * semantic_score +
            0.3 * interest_score +
            0.15 * budget_score +
            0.15 * duration_score
        )

        results.append(
            RecommendedPlace(
                place_id=place["place_id"],
                name=place["name"],
                category=place["category"],
                zone=place.get("zone", ""),
                score=float(final_score),
                semantic_score=float(semantic_score),
                profile_score=float(interest_score),
                budget_score=float(budget_score),
                duration_score=float(duration_score),
                economic_score=0.5,
                tags=place.get("ai_tags", []),
                short_description=place.get("short_description", ""),
                storytelling_seed=place.get("storytelling_seed", "")
            )
        )

    results.sort(key=lambda x: x.score, reverse=True)

    return results


def recommend_places(profile, places, top_k=5):
    ranked = compute_scores(profile, places)
    return ranked[:top_k]