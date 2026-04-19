def build_grounded_context(fused_analysis: dict, reranked_items: list) -> dict:
    """
    Construit le contexte final à envoyer au LLM :
    - résumé objectif 1
    - résumé objectif 2
    - documents RAG rerankés
    """

    forecast = fused_analysis.get("forecast", {})
    env = fused_analysis.get("environment_profile", {})

    llm_context_chunks = []
    cited_projects = []

    for item in reranked_items:
        chunk = item["chunk"]
        llm_context_chunks.append(
            {
                "project_title": chunk.project_title,
                "chunk_type": chunk.chunk_type,
                "text": chunk.text,
                "metadata": chunk.metadata,
                "rerank_score": item.get("rerank_score", 0.0),
            }
        )

        if chunk.project_title not in cited_projects:
            cited_projects.append(chunk.project_title)

    return {
        "forecast_summary": {
            "selected_date": fused_analysis.get("selected_date"),
            "alert_level": forecast.get("alert_level"),
            "predicted_gwetroot_tplus7": forecast.get("predicted_gwetroot_tplus7"),
            "predicted_t2m_max_tplus7": forecast.get("predicted_t2m_max_tplus7"),
            "dominant_factors": forecast.get("dominant_factors", []),
        },
        "environment_summary": {
            "cluster_id": env.get("cluster_id"),
            "cluster_profile": env.get("cluster_profile"),
            "cluster_summary": env.get("cluster_summary"),
        },
        "retrieved_chunks": llm_context_chunks,
        "cited_projects": cited_projects,
    }