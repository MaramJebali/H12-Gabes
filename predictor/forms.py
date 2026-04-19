from django import forms


class PredictionForm(forms.Form):
    selected_date = forms.DateField(
        widget=forms.DateInput(attrs={"type": "date"})
    )


class TourismForm(forms.Form):
    TOURIST_TYPE_CHOICES = [
        ("general", "Touriste général"),
        ("étranger", "Touriste étranger"),
        ("famille", "Famille"),
        ("couple", "Couple"),
        ("photographer", "Photographe"),
        ("ecotourist", "Écotouriste"),
    ]

    BUDGET_CHOICES = [
        ("low", "Faible"),
        ("medium", "Moyen"),
        ("high", "Élevé"),
        ("low_to_medium", "Faible à moyen"),
    ]

    TRAVEL_STYLE_CHOICES = [
        ("balanced", "Équilibré"),
        ("immersive", "Immersif"),
        ("slow", "Slow tourism"),
        ("premium", "Premium"),
    ]

    TIME_CHOICES = [
        ("morning", "Matin"),
        ("afternoon", "Après-midi"),
        ("late_afternoon", "Fin d'après-midi"),
        ("daytime", "Journée"),
    ]

    SEASON_CHOICES = [
        ("spring", "Printemps"),
        ("summer", "Été"),
        ("autumn", "Automne"),
        ("winter", "Hiver"),
    ]

    MOBILITY_CHOICES = [
        ("normal", "Normale"),
        ("reduced", "Mobilité réduite"),
    ]

    LANGUAGE_CHOICES = [
        ("fr", "Français"),
        ("en", "Anglais"),
    ]

    INTEREST_CHOICES = [
        ("culture", "Culture"),
        ("nature", "Nature"),
        ("photography", "Photographie"),
        ("family", "Famille"),
        ("ecotourist", "Écotourisme"),
        ("slow_tourism", "Slow tourism"),
        ("roadtrip", "Road trip"),
        ("artisanat", "Artisanat"),
        ("gastronomy", "Gastronomie"),
    ]

    tourist_type = forms.ChoiceField(choices=TOURIST_TYPE_CHOICES)
    duration_days = forms.IntegerField(min_value=1, max_value=5, initial=2)
    budget = forms.ChoiceField(choices=BUDGET_CHOICES, initial="medium")
    interests = forms.MultipleChoiceField(
        choices=INTEREST_CHOICES,
        widget=forms.CheckboxSelectMultiple,
    )
    travel_style = forms.ChoiceField(choices=TRAVEL_STYLE_CHOICES, initial="immersive")
    preferred_time = forms.ChoiceField(choices=TIME_CHOICES, initial="morning")
    season = forms.ChoiceField(choices=SEASON_CHOICES, initial="autumn")
    mobility = forms.ChoiceField(choices=MOBILITY_CHOICES, initial="normal")
    language = forms.ChoiceField(choices=LANGUAGE_CHOICES, initial="fr")
    
    
class PalmLeafForm(forms.Form):
    image = forms.ImageField(label="Image de feuille de palmier")