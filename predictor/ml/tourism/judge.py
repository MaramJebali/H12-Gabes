from __future__ import annotations

import json
import os
import re
from openai import OpenAI


def _build_profile_text(profile) -> str:
    return (
        f"Type: {profile.tourist_type}\n"
        f"Durée: {profile.duration_days} jours\n"
        f"Budget: {profile.budget}\n"
        f"Intérêts: {', '.join(profile.interests)}\n"
        f"Style: {profile.travel_style}\n"
        f"Moment préféré: {profile.preferred_time}\n"
        f"Saison: {profile.season}\n"
        f"Mobilité: {profile.mobility}\n"
        f"Langue: {profile.language}"
    )


def _build_places_text(recommended_places: list) -> str:
    if not recommended_places:
        return "Aucun lieu recommandé."

    lines = []
    for p in recommended_places:
        lines.append(
            f"- {p.name} | catégorie={p.category} | zone={p.zone} | "
            f"score={p.score:.3f} | semantic={p.semantic_score:.3f} | "
            f"profile_fit={p.profile_score:.3f}"
        )
    return "\n".join(lines)


def _build_itinerary_text(itinerary_days: list) -> str:
    if not itinerary_days:
        return "Aucun itinéraire généré."

    lines = []
    for day in itinerary_days:
        lines.append(f"Jour {day.day} — {day.theme}")
        for stop in day.stops:
            lines.append(
                f"- {stop.get('name', 'N/A')} ({stop.get('category', 'N/A')}): "
                f"{stop.get('description', '')}"
            )
    return "\n".join(lines)


def _build_rag_text(rag_context: dict) -> str:
    chunks = rag_context.get("retrieved_chunks", [])
    if not chunks:
        return "Aucun contexte RAG récupéré."

    lines = []
    for idx, chunk in enumerate(chunks, start=1):
        lines.append(
            f"[Chunk {idx}] "
            f"Lieu={chunk.get('place_name', 'N/A')} | "
            f"Type={chunk.get('chunk_type', 'N/A')} | "
            f"Rerank={chunk.get('rerank_score', 0.0):.3f}\n"
            f"{chunk.get('text', '')}"
        )
    return "\n\n".join(lines)


def _fallback_local_judge(profile, recommended_places, itinerary_days, rag_context, generated_recommendation):
    scores = {
        "coherence_score": 0.0,
        "diversity_score": 0.0,
        "profile_fit_score": 0.0,
        "feasibility_score": 0.0,
    }

    issues = []
    suggestions = []

    non_empty_days = sum(1 for d in itinerary_days if len(d.stops) > 0)
    scores["coherence_score"] = non_empty_days / len(itinerary_days) if itinerary_days else 0.0

    categories = []
    for d in itinerary_days:
        for stop in d.stops:
            categories.append(stop.get("category"))

    unique_categories = len(set(categories)) if categories else 0
    total_categories = len(categories) if categories else 1
    scores["diversity_score"] = min(1.0, unique_categories / max(1, min(total_categories, 4)))

    if recommended_places:
        avg_profile_score = sum(p.profile_score for p in recommended_places) / len(recommended_places)
        scores["profile_fit_score"] = avg_profile_score
    else:
        scores["profile_fit_score"] = 0.0
        issues.append("Aucun lieu recommandé.")

    overloaded_days = sum(1 for d in itinerary_days if len(d.stops) > 3)
    scores["feasibility_score"] = max(0.0, 1.0 - (overloaded_days / len(itinerary_days))) if itinerary_days else 0.0

    if unique_categories < 2:
        issues.append("Le circuit manque de diversité.")
        suggestions.append("Ajouter un lieu d’une autre catégorie.")

    if overloaded_days > 0:
        issues.append("Certaines journées semblent surchargées.")
        suggestions.append("Répartir les visites sur plus de temps.")

    overall_score = (
        0.30 * scores["coherence_score"] +
        0.20 * scores["diversity_score"] +
        0.30 * scores["profile_fit_score"] +
        0.20 * scores["feasibility_score"]
    )

    if overall_score >= 0.8:
        verdict = "Très bon circuit"
    elif overall_score >= 0.6:
        verdict = "Circuit correct mais améliorable"
    else:
        verdict = "Circuit insuffisant"

    return {
        "overall_score": round(overall_score, 3),
        "coherence_score": round(scores["coherence_score"], 3),
        "diversity_score": round(scores["diversity_score"], 3),
        "profile_fit_score": round(scores["profile_fit_score"], 3),
        "feasibility_score": round(scores["feasibility_score"], 3),
        "issues": issues,
        "suggestions": suggestions,
        "verdict": verdict,
        "mode": "fallback_local"
    }


def _safe_parse_json(content: str) -> dict:
    try:
        return json.loads(content)
    except Exception:
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except Exception:
                pass

    return {
        "overall_score": 0.5,
        "coherence_score": 0.5,
        "diversity_score": 0.5,
        "profile_fit_score": 0.5,
        "feasibility_score": 0.5,
        "issues": ["Parsing JSON échoué"],
        "suggestions": ["Vérifier la réponse du juge LLM"],
        "verdict": "Erreur parsing"
    }


def evaluate_tourism_plan(profile, recommended_places, itinerary_days, rag_context: dict, generated_recommendation: str) -> dict:
    api_key = os.getenv("GROQ_API_KEY")
    model_name = os.getenv("GROQ_JUDGE_MODEL", "llama-3.1-8b-instant")

    if not api_key:
        return _fallback_local_judge(
            profile=profile,
            recommended_places=recommended_places,
            itinerary_days=itinerary_days,
            rag_context=rag_context,
            generated_recommendation=generated_recommendation,
        )

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    profile_text = _build_profile_text(profile)
    places_text = _build_places_text(recommended_places)
    itinerary_text = _build_itinerary_text(itinerary_days)
    rag_text = _build_rag_text(rag_context)

    prompt = f"""
Tu es un évaluateur expert de circuits touristiques patrimoniaux pour Gabès.

Ta mission :
évaluer objectivement la qualité d’un circuit généré par une IA pour un touriste donné.

Tu dois juger selon 5 critères, chacun noté entre 0 et 1 :
- overall_score
- coherence_score
- diversity_score
- profile_fit_score
- feasibility_score

Définitions :
- coherence_score : le circuit a-t-il une logique globale claire ?
- diversity_score : le circuit évite-t-il la répétition et propose-t-il des expériences variées ?
- profile_fit_score : le circuit correspond-il réellement au profil utilisateur ?
- feasibility_score : le programme semble-t-il réaliste sur la durée et le nombre de visites ?
- overall_score : synthèse globale

==============================
PROFIL TOURISTE
==============================
{profile_text}

==============================
LIEUX RECOMMANDÉS
==============================
{places_text}

==============================
ITINÉRAIRE GÉNÉRÉ
==============================
{itinerary_text}

==============================
CONTEXTE RAG
==============================
{rag_text}

==============================
TEXTE FINAL GÉNÉRÉ
==============================
{generated_recommendation}

==============================
INSTRUCTIONS
==============================
Réponds STRICTEMENT sous la forme d’un objet JSON valide.
N’ajoute aucun texte avant ou après le JSON.
Le format attendu est exactement :

{{
  "overall_score": 0.0,
  "coherence_score": 0.0,
  "diversity_score": 0.0,
  "profile_fit_score": 0.0,
  "feasibility_score": 0.0,
  "issues": ["..."],
  "suggestions": ["..."],
  "verdict": "..."
}}
"""

    response = client.chat.completions.create(
        model=model_name,
        temperature=0.1,
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu es un juge expert en circuits touristiques, "
                    "spécialisé en qualité d’itinéraires, diversité, faisabilité et personnalisation. "
                    "Tu réponds uniquement en JSON valide."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )

    content = response.choices[0].message.content.strip()
    parsed = _safe_parse_json(content)
    parsed["mode"] = "llm_judge"

    return parsed