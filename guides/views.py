import json
import os

from django.shortcuts import render, get_object_or_404, redirect
from django.http import HttpResponse, JsonResponse
from django.contrib import messages
from django.conf import settings

from .models import Guide, Reservation
from .forms import GuideForm, ReservationForm


# ---------------------------------------------------------------------------
# TOURISM — redirects to the unified predictor page (tourism_home)
# The tourism app no longer renders its own template.
# All guide/reservation CRUD still lives here; after each action
# we redirect to 'tourism_home' where the unified template renders.
# ---------------------------------------------------------------------------

def tourism(request):
    """Legacy entry point — forward to the unified page."""
    return redirect('tourism_home')


def guide_list(request):
    """Legacy fallback — forward to the unified page."""
    return redirect('tourism_home')


# ---------------------------------------------------------------------------
# ADD GUIDE
# POST-only. Saves the guide then redirects back to the unified page.
# ---------------------------------------------------------------------------

def add_guide(request):
    if request.method == 'POST':
        form = GuideForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            messages.success(request, '✓ Guide ajouté avec succès.')
        else:
            error_text = ' | '.join(
                f"{field}: {', '.join(errs)}"
                for field, errs in form.errors.items()
            )
            messages.error(request, f'Impossible d\'ajouter le guide : {error_text}')
    return redirect('tourism_home')


# ---------------------------------------------------------------------------
# RESERVE GUIDE
# POST-only. Conflict check: same guide + same date = blocked.
# ---------------------------------------------------------------------------

def reserve_guide(request):
    if request.method == 'POST':
        guide_id = request.POST.get('guide_id')
        guide = get_object_or_404(Guide, id=guide_id)
        form = ReservationForm(request.POST)

        if form.is_valid():
            tour_date = form.cleaned_data['tour_date']

            already_booked = Reservation.objects.filter(
                guide=guide, tour_date=tour_date
            ).exists()

            if already_booked:
                messages.error(
                    request,
                    f'{guide.name} est déjà réservé le {tour_date}. Veuillez choisir une autre date.'
                )
            else:
                reservation = form.save(commit=False)
                reservation.guide = guide
                reservation.save()
                messages.success(
                    request,
                    f'✓ Réservation confirmée ! {guide.name} est réservé pour le {tour_date}.'
                )
        else:
            error_text = ' | '.join(
                f"{field}: {', '.join(errs)}"
                for field, errs in form.errors.items()
            )
            messages.error(request, f'Réservation échouée : {error_text}')

    return redirect('tourism_home')


# ---------------------------------------------------------------------------
# RESERVATION SUCCESS  (kept for backward compatibility)
# ---------------------------------------------------------------------------

def reservation_success(request):
    messages.success(request, '✓ Votre réservation a bien été enregistrée !')
    return redirect('tourism_home')


# ---------------------------------------------------------------------------
# GUIDE RESERVATIONS  — JSON endpoint for the reservations modal
# Called via fetch() from the template. No redirect needed.
# ---------------------------------------------------------------------------

def guide_reservations(request, guide_id):
    guide = get_object_or_404(Guide, id=guide_id)
    reservations = Reservation.objects.filter(guide=guide).values(
        'id', 'tourist_name', 'tourist_phone_number', 'tour_date', 'message'
    )
    data = {
        'guide_name': guide.name,
        'reservations': list(reservations),
    }
    return JsonResponse(data)


# ---------------------------------------------------------------------------
# DELETE GUIDE
# ---------------------------------------------------------------------------

def delete_guide(request, guide_id):
    guide = get_object_or_404(Guide, id=guide_id)
    if request.method == 'POST':
        name = guide.name
        guide.delete()
        messages.success(request, f'Guide {name} supprimé.')
    return redirect('tourism_home')


# ---------------------------------------------------------------------------
# MAP VIEW  (kept from original)
# ---------------------------------------------------------------------------

def map_view(request):
    json_file_path = os.path.join(settings.BASE_DIR, 'data', 'Details.json')
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    return render(request, 'predictor/gabes-map.html', {'places': data})


# ---------------------------------------------------------------------------
# PLACE DETAILS  (kept from original)
# ---------------------------------------------------------------------------

def place_details(request, place_name):
    json_file_path = os.path.join(settings.BASE_DIR, 'data', 'Details.json')
    with open(json_file_path, 'r') as f:
        places_data = json.load(f)
    place = places_data.get(place_name)
    if place:
        return render(request, 'predictor/details.html', {'place': place})
    return HttpResponse('Place not found', status=404)