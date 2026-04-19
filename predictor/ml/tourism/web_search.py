import requests
import random


def search_restaurants(place_name):
    # Simulation enrichie (remplaçable plus tard par API réelle)
    return [
        {
            "name": f"Restaurant local à {place_name}",
            "type": "restaurant",
            "rating": round(random.uniform(3.8, 4.7), 1),
            "price_level": random.choice(["low", "medium"]),
            "source": "simulated"
        }
    ]


def search_hotels(place_name):
    return [
        {
            "name": f"Hôtel proche de {place_name}",
            "type": "hotel",
            "rating": round(random.uniform(3.5, 4.8), 1),
            "price_level": random.choice(["medium", "high"]),
            "source": "simulated"
        }
    ]


def search_events(place_name):
    return [
        {
            "title": f"Événement culturel à {place_name}",
            "type": "event",
            "date": "prochainement",
            "source": "simulated"
        }
    ]


def enrich_with_web_data(recommended_places):
    enriched_data = []

    for place in recommended_places:
        place_name = place.name

        restaurants = search_restaurants(place_name)
        hotels = search_hotels(place_name)
        events = search_events(place_name)

        enriched_data.append({
            "place": place_name,
            "restaurants": restaurants,
            "hotels": hotels,
            "events": events
        })

    return enriched_data