from .schemas import TouristProfile


def build_tourist_profile(form_data: dict) -> TouristProfile:
    return TouristProfile(
        tourist_type=form_data.get("tourist_type", "general"),
        duration_days=int(form_data.get("duration_days", 1)),
        budget=form_data.get("budget", "medium"),
        interests=form_data.get("interests", []),
        travel_style=form_data.get("travel_style", "balanced"),
        preferred_time=form_data.get("preferred_time", "morning"),
        season=form_data.get("season", "spring"),
        mobility=form_data.get("mobility", "normal"),
        language=form_data.get("language", "fr"),
    )


def profile_to_text(profile: TouristProfile) -> str:
    interests_text = ", ".join(profile.interests)

    return (
        f"Touriste {profile.tourist_type}, "
        f"séjour de {profile.duration_days} jours, "
        f"budget {profile.budget}, "
        f"intérêts : {interests_text}, "
        f"style de voyage {profile.travel_style}, "
        f"préférence horaire {profile.preferred_time}, "
        f"saison {profile.season}, "
        f"mobilité {profile.mobility}, "
        f"langue {profile.language}."
    )