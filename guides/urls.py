from django.urls import path
from . import views

urlpatterns = [
    # ── Legacy entry (now redirects to tourism_home) ─────────────────────
    path('', views.tourism, name='tourism'),

    # ── Guide list fallback (also redirects to tourism_home) ─────────────
    path('list/', views.guide_list, name='guide_list'),

    # ── Guide CRUD endpoints ──────────────────────────────────────────────
    path('add/', views.add_guide, name='add_guide'),
    path('delete/<int:guide_id>/', views.delete_guide, name='delete_guide'),

    # ── Reservation endpoints ─────────────────────────────────────────────
    path('reserve/', views.reserve_guide, name='reserve_guide'),
    path('reservation-success/', views.reservation_success, name='reservation_success'),

    # JSON endpoint — called via fetch() from the unified template
    path('reservations/<int:guide_id>/', views.guide_reservations, name='guide_reservations'),

    # ── Map views (kept from original) ───────────────────────────────────
    path('map/', views.map_view, name='map'),
    path('place/<str:place_name>/', views.place_details, name='place_details'),
]