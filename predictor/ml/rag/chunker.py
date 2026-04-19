from typing import List, Dict
from .schemas import RAGChunk


def safe_join(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return " ; ".join(str(v) for v in value if v is not None)
    return str(value)


def build_project_chunks(projects: List[Dict]) -> List[RAGChunk]:
    chunks = []

    for idx, project in enumerate(projects):
        project_title = project.get("titre", f"project_{idx}")
        source_file = project.get("_source_file", "unknown.json")

        objectif = project.get("objectif", "")
        contexte = project.get("contexte", "")
        realisations = safe_join(project.get("realisations", []))
        resultats = safe_join(project.get("resultats_chiffres", []))
        zones = safe_join(project.get("zones_geographiques", []))
        themes = safe_join(project.get("themes", []))
        mots_cles = safe_join(project.get("mots_cles", []))

        infrastructures = project.get("infrastructure_disponible", [])
        infrastructures_text = []
        for infra in infrastructures:
            if isinstance(infra, dict):
                infra_text = (
                    f"Nom: {infra.get('nom', '')}. "
                    f"Type: {infra.get('type', '')}. "
                    f"Statut: {infra.get('statut', '')}. "
                    f"Localisation: {infra.get('localisation', '')}. "
                    f"Description: {infra.get('description', '')}."
                )
                infrastructures_text.append(infra_text)
        infrastructures_text = " ".join(infrastructures_text)

        base_metadata = {
            "zones": project.get("zones_geographiques", []),
            "themes": project.get("themes", []),
            "keywords": project.get("mots_cles", []),
            "source_file": source_file,
        }

        # Chunk overview
        overview_text = (
            f"Titre du projet : {project_title}. "
            f"Objectif : {objectif}. "
            f"Contexte : {contexte}. "
            f"Zones géographiques : {zones}. "
            f"Thèmes : {themes}. "
            f"Mots-clés : {mots_cles}."
        )
        chunks.append(
            RAGChunk(
                chunk_id=f"{project_title}_overview_{idx}",
                source_file=source_file,
                project_title=project_title,
                chunk_type="overview",
                text=overview_text,
                metadata=base_metadata,
            )
        )

        # Chunk réalisations
        if realisations:
            chunks.append(
                RAGChunk(
                    chunk_id=f"{project_title}_actions_{idx}",
                    source_file=source_file,
                    project_title=project_title,
                    chunk_type="actions",
                    text=f"Projet {project_title}. Réalisations : {realisations}",
                    metadata=base_metadata,
                )
            )

        # Chunk résultats
        if resultats:
            chunks.append(
                RAGChunk(
                    chunk_id=f"{project_title}_results_{idx}",
                    source_file=source_file,
                    project_title=project_title,
                    chunk_type="results",
                    text=f"Projet {project_title}. Résultats chiffrés : {resultats}",
                    metadata=base_metadata,
                )
            )

        # Chunk infrastructures
        if infrastructures_text:
            chunks.append(
                RAGChunk(
                    chunk_id=f"{project_title}_infra_{idx}",
                    source_file=source_file,
                    project_title=project_title,
                    chunk_type="infrastructure",
                    text=f"Projet {project_title}. Infrastructures disponibles : {infrastructures_text}",
                    metadata=base_metadata,
                )
            )

    return chunks