from .schemas import ItineraryDay


def build_itinerary(profile, recommended_places):
    """
    Ancienne logique : construit un itinéraire à partir des lieux recommandés
    par le moteur amont.
    """
    days = []
    if not recommended_places:
        return days

    places_per_day = max(1, len(recommended_places) // profile.duration_days)
    index = 0

    for d in range(profile.duration_days):
        stops = []

        for _ in range(places_per_day):
            if index >= len(recommended_places):
                break

            place = recommended_places[index]

            stops.append({
                "name": place.name,
                "category": place.category,
                "description": place.short_description,
                "tags": place.tags,
            })

            index += 1

        days.append(
            ItineraryDay(
                day=d + 1,
                theme="Découverte culturelle et naturelle",
                stops=stops
            )
        )

    while index < len(recommended_places):
        days[-1].stops.append({
            "name": recommended_places[index].name,
            "category": recommended_places[index].category,
            "description": recommended_places[index].short_description,
            "tags": recommended_places[index].tags,
        })
        index += 1

    return days


def build_itinerary_from_rag(profile, rag_map_places):
    """
    Nouvelle logique finale : construit l’itinéraire uniquement
    à partir des lieux réellement retenus par le RAG.
    """
    days = []
    if not rag_map_places:
        return days

    places_per_day = max(1, len(rag_map_places) // profile.duration_days)
    index = 0

    for d in range(profile.duration_days):
        stops = []

        for _ in range(places_per_day):
            if index >= len(rag_map_places):
                break

            place = rag_map_places[index]

            stops.append({
                "name": place.get("name", ""),
                "category": place.get("category", ""),
                "description": place.get("text", ""),
                "tags": [],
            })

            index += 1

        days.append(
            ItineraryDay(
                day=d + 1,
                theme="Découverte de Gabès",
                stops=stops
            )
        )

    while index < len(rag_map_places):
        days[-1].stops.append({
            "name": rag_map_places[index].get("name", ""),
            "category": rag_map_places[index].get("category", ""),
            "description": rag_map_places[index].get("text", ""),
            "tags": [],
        })
        index += 1

    return days