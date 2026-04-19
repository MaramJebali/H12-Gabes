from django.urls import path
from .views import landing_page, home, analyze, tourism_home, tourism_analyze , palm_home,environment_palm_analyze,pollution_data_api


urlpatterns = [
    # ── Landing ───────────────────────────────────────────────────────────
    path("", landing_page, name="landing_page"),

    # ── Environment ───────────────────────────────────────────────────────
    path("environment/", home, name="home"),
    path("analyze/", analyze, name="analyze"),

    # ── Tourism (unified template) ────────────────────────────────────────
    # GET  → tourism_home  → renders form + map + guides panel
    # POST → tourism_analyze → runs pipeline → renders result in same template
    path("tourism/", tourism_home, name="tourism_home"),
    path("tourism/analyze/", tourism_analyze, name="tourism_analyze"),
    
    path("environment/palm-analyze/", environment_palm_analyze, name="environment_palm_analyze"),
    path('environment/pollution-data/', pollution_data_api, name='pollution_data_api'),

    path("palm-health/", palm_home, name="palm_home"),
    
]