from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class TouristProfile:
    tourist_type: str
    duration_days: int
    budget: str
    interests: List[str]
    travel_style: str
    preferred_time: str
    season: str
    mobility: str = "normal"
    language: str = "fr"


@dataclass
class RecommendedPlace:
    place_id: str
    name: str
    category: str
    zone: str
    score: float
    semantic_score: float
    profile_score: float
    budget_score: float
    duration_score: float
    economic_score: float
    tags: List[str] = field(default_factory=list)
    short_description: str = ""
    storytelling_seed: str = ""


@dataclass
class ItineraryDay:
    day: int
    theme: str
    stops: List[Dict]


@dataclass
class TourismRecommendation:
    title: str
    days: List[ItineraryDay]
    top_places: List[RecommendedPlace]
    services: List[Dict] = field(default_factory=list)
    generated_text: Optional[str] = None
    judge_score: Optional[Dict] = None