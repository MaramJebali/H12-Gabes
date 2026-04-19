from __future__ import annotations

import os
from openai import OpenAI


def build_itinerary_text(itinerary_days: list) -> str:
    lines = []
    for day in itinerary_days:
        lines.append(f"Jour {day.day} — {day.theme}")
        for stop in day.stops:
            lines.append(
                f"- {stop.get('name', 'N/A')} "
                f"({stop.get('category', 'N/A')}) : "
                f"{stop.get('description', '')}"
            )
    return "\n".join(lines)


def build_services_text(services: list) -> str:
    if not services:
        return "Aucun service complémentaire récupéré."

    lines = []
    for service in services:
        lines.append(
            f"- {service.get('name', 'N/A')} | "
            f"type: {service.get('type', 'N/A')} | "
            f"zone: {service.get('zone', 'N/A')} | "
            f"niveau de prix: {service.get('price_level', 'N/A')} | "
            f"statut: {service.get('verification_status', 'N/A')}"
        )
    return "\n".join(lines)


def build_rag_text(rag_context: dict) -> str:
    chunks = rag_context.get("retrieved_chunks", [])
    if not chunks:
        return "Aucun contexte patrimonial récupéré."

    lines = []
    for idx, chunk in enumerate(chunks, start=1):
        lines.append(
            f"[Document {idx}] "
            f"Lieu: {chunk.get('place_name', 'N/A')} | "
            f"Type: {chunk.get('chunk_type', 'N/A')}\n"
            f"{chunk.get('text', '')}"
        )
    return "\n\n".join(lines)


def build_web_data_text(web_data: list) -> str:
    if not web_data:
        return "Aucune donnée web disponible."

    lines = []

    for item in web_data:
        lines.append(f"Lieu: {item.get('place', 'N/A')}")

        for r in item.get("restaurants", []):
            lines.append(
                f"Restaurant: {r.get('name', 'N/A')} "
                f"(rating {r.get('rating', 'N/A')}, "
                f"prix {r.get('price_level', 'N/A')}, "
                f"source {r.get('source', 'N/A')})"
            )

        for h in item.get("hotels", []):
            lines.append(
                f"Hôtel: {h.get('name', 'N/A')} "
                f"(rating {h.get('rating', 'N/A')}, "
                f"prix {h.get('price_level', 'N/A')}, "
                f"source {h.get('source', 'N/A')})"
            )

        for e in item.get("events", []):
            lines.append(
                f"Événement: {e.get('title', 'N/A')} "
                f"(date {e.get('date', 'N/A')}, source {e.get('source', 'N/A')})"
            )

    return "\n".join(lines)


def generate_tourism_recommendation(
    profile,
    recommended_places,
    itinerary_days,
    rag_context: dict,
    web_data: list | None = None,
) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    profile_text = (
        f"Touriste: {profile.tourist_type}\n"
        f"Durée: {profile.duration_days} jours\n"
        f"Budget: {profile.budget}\n"
        f"Intérêts: {', '.join(profile.interests)}\n"
        f"Style: {profile.travel_style}\n"
        f"Moment préféré: {profile.preferred_time}\n"
        f"Saison: {profile.season}\n"
        f"Mobilité: {profile.mobility}\n"
        f"Langue: {profile.language}\n"
    )

    places_text = "\n".join([
        f"- {p.name} | score={p.score:.3f} | catégorie={p.category} | zone={p.zone}"
        for p in recommended_places
    ]) if recommended_places else "Aucun lieu recommandé."

    itinerary_text = build_itinerary_text(itinerary_days)
    rag_text = build_rag_text(rag_context)
    services_text = build_services_text(rag_context.get("linked_services", []))
    web_text = build_web_data_text(web_data or [])

    if not api_key:
        return (
            "[Mode sans Groq] Circuit touristique généré.\n\n"
            f"Profil:\n{profile_text}\n"
            f"Lieux retenus:\n{places_text}\n\n"
            f"Itinéraire:\n{itinerary_text}\n\n"
            "Recommandation : proposer un circuit patrimonial immersif équilibré, "
            "mettant en avant les oasis, le patrimoine culturel, les services de proximité "
            "et les données web récentes disponibles."
        )

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )

    prompt = f"""
Tu es un expert en tourisme culturel, patrimoine, storytelling territorial et expérience visiteur pour la région de Gabès.

Ta mission :
générer une recommandation touristique finale, immersive, cohérente et utile, à partir :
1. du profil du touriste,
2. des lieux recommandés par le moteur IA,
3. du circuit candidat déjà construit,
4. du contexte patrimonial récupéré par le RAG,
5. des services complémentaires disponibles,
6. des données web dynamiques.

==============================
PROFIL DU TOURISTE
==============================
{profile_text}

==============================
LIEUX RECOMMANDÉS PAR LE MOTEUR IA
==============================
{places_text}

==============================
CIRCUIT CANDIDAT
==============================
{itinerary_text}

==============================
CONTEXTE PATRIMONIAL RAG
==============================
{rag_text}

==============================
SERVICES COMPLÉMENTAIRES
==============================
{services_text}

==============================
DONNÉES WEB EN TEMPS RÉEL
==============================
{web_text}

==============================
INSTRUCTIONS
==============================
Réponds en français.
Sois clair, structuré, attractif et grounded.
Tu dois t'appuyer sur les lieux et le contexte fournis.
Tu peux enrichir le ton, mais ne dois pas inventer des lieux absents du contexte.

Donne exactement :
1. Titre du circuit
2. Pourquoi ce circuit correspond au profil
3. Programme par jour
4. Conseils pratiques
5. Valeur culturelle et économique locale
"""

    response = client.chat.completions.create(
        model=model_name,
        temperature=0.3,
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu es un concepteur de circuits patrimoniaux premium pour Gabès. "
                    "Tu écris en français, de manière claire, séduisante, crédible et structurée."
                )
            },
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content.strip()