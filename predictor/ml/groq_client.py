from __future__ import annotations

import os
from openai import OpenAI


def generate_recommendation(fused_analysis: dict, rag_payload: dict | None = None) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    forecast = fused_analysis.get("forecast", {})
    profile = fused_analysis.get("environment_profile", {})
    selected_date = fused_analysis.get("selected_date", "N/A")

    current_gwetroot = forecast.get("current_gwetroot", None)
    predicted_gwetroot = forecast.get("predicted_gwetroot_tplus7", None)
    current_t2m_max = forecast.get("current_t2m_max", None)
    predicted_t2m_max = forecast.get("predicted_t2m_max_tplus7", None)
    alert_level = forecast.get("alert_level", "inconnu")
    dominant_factors = forecast.get("dominant_factors", [])

    cluster_id = profile.get("cluster_id", "N/A")
    cluster_profile = profile.get("cluster_profile", "profil non disponible")
    cluster_summary = profile.get("cluster_summary", "résumé non disponible")

    dominant_text = ", ".join(dominant_factors) if dominant_factors else "non spécifiés"

    retrieved_context = ""
    cited_projects = []

    if rag_payload:
        grounded_context = rag_payload.get("grounded_context", {})
        cited_projects = grounded_context.get("cited_projects", [])

        chunks = grounded_context.get("retrieved_chunks", [])
        formatted_chunks = []
        for idx, chunk in enumerate(chunks, start=1):
            formatted_chunks.append(
                f"[Document {idx}] "
                f"Projet: {chunk.get('project_title', 'N/A')} | "
                f"Type: {chunk.get('chunk_type', 'N/A')}\n"
                f"{chunk.get('text', '')}"
            )
        retrieved_context = "\n\n".join(formatted_chunks)

    if not api_key:
        cited_text = ", ".join(cited_projects) if cited_projects else "aucun projet local récupéré"
        return (
            "[Mode sans Groq] Recommandation fusionnée : "
            f"alerte {alert_level}, profil {cluster_profile}. "
            f"Facteurs dominants : {dominant_text}. "
            f"Projets locaux considérés : {cited_text}. "
            "Action recommandée : surveiller l’humidité du sol, anticiper les épisodes de chaleur, "
            "et prioriser les actions cohérentes avec les initiatives locales identifiées."
        )

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )

    prompt = f"""
Tu es un expert en résilience climatique, agriculture oasienne, gestion de l’eau et stratégies territoriales à Gabès.

Ta mission :
produire une recommandation FINALE, contextualisée, grounded et utile, à partir :
1. de la prévision climatique à J+7,
2. du profil environnemental détecté,
3. des projets et actions réels récupérés dans la base locale de Gabès.

==============================
CONTEXTE 1 — PRÉVISION J+7
==============================
- Date analysée : {selected_date}
- Humidité du sol actuelle : {current_gwetroot if current_gwetroot is not None else "N/A"}
- Humidité du sol prédite à J+7 : {predicted_gwetroot if predicted_gwetroot is not None else "N/A"}
- Température max actuelle : {current_t2m_max if current_t2m_max is not None else "N/A"} °C
- Température max prédite à J+7 : {predicted_t2m_max if predicted_t2m_max is not None else "N/A"} °C
- Niveau d’alerte : {alert_level}
- Facteurs dominants : {dominant_text}

==============================
CONTEXTE 2 — PROFIL ENVIRONNEMENTAL
==============================
- Cluster ID : {cluster_id}
- Profil : {cluster_profile}
- Résumé du cluster : {cluster_summary}

==============================
CONTEXTE 3 — CONNAISSANCES LOCALES (RAG)
==============================
{retrieved_context if retrieved_context else "Aucun document récupéré."}

==============================
INSTRUCTIONS
==============================
Réponds en français.
Sois concret, structuré, pertinent et orienté décision.
Utilise les projets récupérés si c’est pertinent.
Ne fais pas d’affirmations génériques sans lien avec le contexte.
Ne cite que les projets effectivement fournis dans les documents récupérés.

Donne exactement :
1. Situation :
2. Recommandations :
   1.
   2.
   3.
3. Projets locaux pertinents :
4. Action prioritaire :
"""

    response = client.chat.completions.create(
        model=model_name,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu réponds en français, de façon grounded, structurée, concise et stratégique. "
                    "Tu relies les recommandations aux documents fournis quand c’est possible."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content.strip()