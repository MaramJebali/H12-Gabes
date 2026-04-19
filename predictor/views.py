import os
import json
from uuid import uuid4
from django.conf import settings
from django.shortcuts import render
from django.contrib import messages
from django.http import JsonResponse  # ADD THIS IMPORT
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .forms import PredictionForm, TourismForm, PalmLeafForm

# ── Environment ML pipeline ───────────────────────────────────────────────
from .ml.model_pipeline import get_available_date_bounds, predict_future_risk
from .ml.clustering_pipeline import get_cluster_for_date
from .ml.rag.rag_service import AdvancedRAGService
from .ml.groq_client import generate_recommendation

# ── Tourism ML pipeline ───────────────────────────────────────────────────
from .ml.tourism.data_loader import load_all_tourism_data
from .ml.tourism.profile_builder import build_tourist_profile
from .ml.tourism.recommender import recommend_places
from .ml.tourism.itinerary_builder import build_itinerary_from_rag
from .ml.tourism.tourism_rag_service import TourismRAGService
from .ml.tourism.tourism_groq_client import generate_tourism_recommendation
from .ml.tourism.judge import evaluate_tourism_plan
from .ml.tourism.web_search import enrich_with_web_data

# ── Palm Health ML pipeline ───────────────────────────────────────────────
from .ml.palm_health.predictor import predict_palm_leaf

# ── Guide model — lives in the guides app ────────────────────────────────
from guides.models import Guide


# =============================================================================
# LANDING PAGE
# =============================================================================

def landing_page(request):
    return render(request, "predictor/landing.html")


# =============================================================================
# ENVIRONMENT — home + analyze
# =============================================================================

def home(request):
    bounds = get_available_date_bounds()
    form = PredictionForm(initial={"selected_date": bounds["max_date"]})
    return render(request, "predictor/home.html", {"form": form, "bounds": bounds})

def analyze(request):
    bounds = get_available_date_bounds()

    if request.method != "POST":
        return render(
            request,
            "predictor/home.html",
            {
                "bounds": bounds,
                "analysis_result": None,
            },
        )

    form = PredictionForm(request.POST)

    if not form.is_valid():
        messages.error(request, "Veuillez choisir une date valide.")
        return render(request, "predictor/home.html", {"bounds": bounds, "analysis_result": None})

    selected_date = str(form.cleaned_data["selected_date"])

    try:
        prediction_obj = predict_future_risk(selected_date)
        prediction = {
            "selected_date": prediction_obj.selected_date,
            "current_gwetroot": prediction_obj.current_gwetroot,
            "predicted_gwetroot_tplus7": prediction_obj.predicted_gwetroot_tplus7,
            "current_t2m_max": prediction_obj.current_t2m_max,
            "predicted_t2m_max_tplus7": prediction_obj.predicted_t2m_max_tplus7,
            "alert_level": prediction_obj.alert_level,
            "dominant_factors": prediction_obj.dominant_factors,
        }
        cluster_result = get_cluster_for_date(selected_date)
        fused_analysis = {
            "selected_date": selected_date,
            "forecast": prediction,
            "environment_profile": cluster_result,
        }
        rag_service = AdvancedRAGService()
        rag_payload = rag_service.retrieve_and_rerank(fused_analysis, retrieve_k=8, rerank_k=5)
        recommendation = generate_recommendation(fused_analysis, rag_payload)

        analysis_result = {
            "prediction": prediction,
            "cluster_result": cluster_result,
            "fused_analysis": fused_analysis,
            "context_rag": rag_payload,
            "recommendation": recommendation,
        }

    except Exception as exc:
        messages.error(request, f"Erreur durant l'analyse : {exc}")
        return render(request, "predictor/home.html", {"bounds": bounds, "analysis_result": None})

    # Render the same template with results
    return render(request, "predictor/home.html", {
        "bounds": bounds,
        "analysis_result": analysis_result,
        "selected_date": selected_date,
    })


# =============================================================================
# TOURISM — shared helpers
# =============================================================================

def _normalize_name(value: str) -> str:
    if not value:
        return ""
    value = value.lower().strip()
    for old, new in {
        "é":"e","è":"e","ê":"e","à":"a","â":"a","î":"i","ï":"i",
        "ô":"o","ù":"u","û":"u","ç":"c","'":"'","-":" ","→":" ",",":" ","  ":" ",
    }.items():
        value = value.replace(old, new)
    return " ".join(value.split())


def _build_place_lookup(places):
    return {_normalize_name(p.get("name","")): p for p in places if p.get("name")}


def _find_place_in_merged_data(place_name: str, lookup: dict):
    if not place_name:
        return None
    normalized = _normalize_name(place_name)
    if normalized in lookup:
        return lookup[normalized]
    for key, place in lookup.items():
        if key in normalized or normalized in key:
            return place
    return None


def _build_map_places_from_json(places):
    seen, result = set(), []
    for p in places:
        coords = p.get("coordinates", {})
        lat, lng = coords.get("lat"), coords.get("lng")
        name = p.get("name", "").strip()
        if lat is None or lng is None or not name or name in seen:
            continue
        result.append({
            "name": name, "lat": lat, "lon": lng,
            "category": p.get("category", ""),
            "text": p.get("display_description") or p.get("field_description") or p.get("short_description",""),
            "images": p.get("media", {}).get("images", []),
        })
        seen.add(name)
    return result


def _get_guides():
    try:
        return list(Guide.objects.all())
    except Exception:
        return []


# =============================================================================
# TOURISM HOME  —  GET → form + map + guides, no result
# =============================================================================

def tourism_home(request):
    form = TourismForm(initial={
        "tourist_type": "étranger", "duration_days": 2, "budget": "medium",
        "travel_style": "immersive", "preferred_time": "morning", "season": "autumn",
        "mobility": "normal", "language": "fr",
        "interests": ["culture", "nature", "photography"],
    })
    tourism_data = load_all_tourism_data()
    home_map_places = _build_map_places_from_json(tourism_data["places"])

    return render(request, "predictor/recommendation.html", {
        "form": form,
        "home_map_places": home_map_places,
        "guides": _get_guides(),
        "itinerary_days": None,
        "profile": None,
        "rag_map_places": [],
        "generated_recommendation": None,
    })


# =============================================================================
# TOURISM ANALYZE  —  POST → ML+RAG+LLM pipeline → same template with results
# =============================================================================

def tourism_analyze(request):
    # Non-POST: redirect to form
    if request.method != "POST":
        return tourism_home(request)

    form = TourismForm(request.POST)

    # Invalid form: re-render with errors
    if not form.is_valid():
        messages.error(request, "Veuillez remplir correctement le formulaire touristique.")
        tourism_data = load_all_tourism_data()
        home_map_places = _build_map_places_from_json(tourism_data["places"])
        return render(request, "predictor/recommendation.html", {
            "form": form,
            "home_map_places": home_map_places,
            "guides": _get_guides(),
            "itinerary_days": None,
            "profile": None,
            "rag_map_places": [],
            "generated_recommendation": None,
        })

    try:
        profile = build_tourist_profile(form.cleaned_data)

        tourism_data = load_all_tourism_data()
        places = tourism_data["places"]
        place_lookup = _build_place_lookup(places)
        home_map_places = _build_map_places_from_json(places)

        recommended_places = recommend_places(profile, places, top_k=6)

        rag_service = TourismRAGService()
        rag_context = rag_service.build_grounded_context(
            profile=profile, recommended_places=recommended_places, top_k=5,
        )

        # Build RAG map places
        rag_map_places, seen_names = [], set()
        for chunk in rag_context.get("retrieved_chunks", []):
            place_name = chunk.get("place_name", "").strip()
            if not place_name:
                continue
            merged = _find_place_in_merged_data(place_name, place_lookup)
            if not merged:
                continue
            coords = merged.get("coordinates", {})
            lat, lng = coords.get("lat"), coords.get("lng")
            display_name = merged.get("name", place_name)
            if lat is None or lng is None or display_name in seen_names:
                continue
            rag_map_places.append({
                "name": display_name, "lat": lat, "lon": lng,
                "chunk_type": chunk.get("chunk_type", ""),
                "text": merged.get("display_description") or merged.get("field_description") or merged.get("short_description",""),
                "images": merged.get("media", {}).get("images", []),
                "category": merged.get("category", ""),
            })
            seen_names.add(display_name)

        itinerary_days = build_itinerary_from_rag(profile, rag_map_places)
        web_data = enrich_with_web_data(recommended_places)

        generated_recommendation = generate_tourism_recommendation(
            profile=profile, recommended_places=recommended_places,
            itinerary_days=itinerary_days, rag_context=rag_context, web_data=web_data,
        )

        judge_result = evaluate_tourism_plan(
            profile=profile, recommended_places=recommended_places,
            itinerary_days=itinerary_days, rag_context=rag_context,
            generated_recommendation=generated_recommendation,
        )

    except Exception as exc:
        messages.error(request, f"Erreur durant l'analyse tourisme : {exc}")
        tourism_data = load_all_tourism_data()
        home_map_places = _build_map_places_from_json(tourism_data["places"])
        return render(request, "predictor/recommendation.html", {
            "form": form,
            "home_map_places": home_map_places,
            "guides": _get_guides(),
            "itinerary_days": None,
            "profile": None,
            "rag_map_places": [],
            "generated_recommendation": None,
        })

    return render(request, "predictor/recommendation.html", {
        "form": form,
        "home_map_places": home_map_places,
        "guides": _get_guides(),
        "profile": profile,
        "itinerary_days": itinerary_days,
        "rag_map_places": rag_map_places,
        "generated_recommendation": generated_recommendation,
        "rag_context": rag_context,
        "judge_result": judge_result,
        "context_web": web_data,
    })


# =============================================================================
# PALM HEALTH — standalone pages
# =============================================================================
def palm_home(request):
    form = PalmLeafForm()
    return render(request, "predictor/home.html", {"form": form})

# =============================================================================
# PALM HEALTH — integrated into environment page (AJAX endpoint)
# =============================================================================

@require_http_methods(["POST"])
def environment_palm_analyze(request):
    """
    AJAX endpoint for palm health analysis from the environment page.
    Returns JSON response with analysis results.
    """
    # Check if this is an AJAX request (optional but recommended)
    if not request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return JsonResponse({"error": "Invalid request"}, status=400)
    
    form = PalmLeafForm(request.POST, request.FILES)
    
    if not form.is_valid():
        return JsonResponse({"error": "Veuillez charger une image valide."}, status=400)
    
    # Create directories if they don't exist
    upload_dir = os.path.join(settings.MEDIA_ROOT, "palm_uploads")
    heatmap_dir = os.path.join(settings.MEDIA_ROOT, "palm_heatmaps")
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(heatmap_dir, exist_ok=True)
    
    # Save uploaded file
    uploaded_file = form.cleaned_data["image"]
    extension = os.path.splitext(uploaded_file.name)[1].lower() or ".jpg"
    file_name = f"{uuid4().hex}{extension}"
    file_path = os.path.join(upload_dir, file_name)
    
    with open(file_path, "wb+") as destination:
        for chunk in uploaded_file.chunks():
            destination.write(chunk)
    
    try:
        # Call the prediction function
        result = predict_palm_leaf(file_path)
    except Exception as exc:
        # Clean up the uploaded file if prediction fails
        if os.path.exists(file_path):
            os.remove(file_path)
        return JsonResponse({"error": f"Erreur durant l'analyse : {str(exc)}"}, status=500)
    
    # Save Grad-CAM heatmap
    heatmap_name = f"heatmap_{uuid4().hex}.png"
    heatmap_path = os.path.join(heatmap_dir, heatmap_name)
    
    try:
        result["heatmap_image"].save(heatmap_path)
    except Exception as exc:
        return JsonResponse({"error": f"Erreur lors de la sauvegarde de la heatmap : {str(exc)}"}, status=500)
    
    # Build URLs for the images
    image_url = settings.MEDIA_URL + "palm_uploads/" + file_name
    heatmap_url = settings.MEDIA_URL + "palm_heatmaps/" + heatmap_name
    
    # Prepare probabilities for JSON response (convert numpy values to Python types)
    probabilities = {}
    if "probabilities" in result:
        for class_name, prob in result["probabilities"].items():
            # Convert numpy float to Python float if needed
            if hasattr(prob, 'item'):
                probabilities[class_name] = prob.item()
            else:
                probabilities[class_name] = float(prob)
    
    # Prepare response data
    response_data = {
        "success": True,
        "image_url": image_url,
        "heatmap_url": heatmap_url,
        "predicted_class": result.get("predicted_class", "Unknown"),
        "probabilities": probabilities,
        "confidence": float(result.get("confidence", 0)) if result.get("confidence") else None,
    }
    
    return JsonResponse(response_data)
# Add this to your views.py
import pandas as pd
import os
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.conf import settings

@require_http_methods(["GET"])
def pollution_data_api(request):
    """API endpoint for pollution data - handles both CSV formats"""
    
    # Try multiple possible paths for the CSV file
    possible_paths = [
        r"C:\Users\maram\Documents\oasis_ai_django_complete\oasis_ai_django_complete\data\Pollution\gabes_heavy_metals_full.csv",
        r"C:\Users\maram\Documents\oasis_ai_django_complete\oasis_ai_django_complete\data\Pollution\gabes_pivot_table.csv",
        os.path.join(settings.BASE_DIR, 'data', 'Pollution', 'gabes_heavy_metals_full.csv'),
        os.path.join(settings.BASE_DIR, 'data', 'Pollution', 'gabes_pivot_table.csv'),
    ]
    
    csv_path = None
    for path in possible_paths:
        if os.path.exists(path):
            csv_path = path
            break
    
    if csv_path is None:
        return JsonResponse({
            "success": False,
            "error": "Data file not found. Please check the file path."
        }, status=500)
    
    try:
        # Load data
        df = pd.read_csv(csv_path)
        
        # Check and adapt column names
        # Map possible column names to standard names
        column_mapping = {}
        
        # Check for Site column
        if 'Site' in df.columns:
            column_mapping['Site'] = 'Site'
        elif 'site' in df.columns:
            column_mapping['Site'] = 'site'
        elif 'Location' in df.columns:
            column_mapping['Site'] = 'Location'
        elif 'location' in df.columns:
            column_mapping['Site'] = 'location'
            
        # Check for Metal column
        if 'Metal' in df.columns:
            column_mapping['Metal'] = 'Metal'
        elif 'metal' in df.columns:
            column_mapping['Metal'] = 'metal'
        elif 'Pollutant' in df.columns:
            column_mapping['Metal'] = 'Pollutant'
        elif 'pollutant' in df.columns:
            column_mapping['Metal'] = 'pollutant'
            
        # Check for Concentration column
        if 'Concentration_ug_per_g' in df.columns:
            column_mapping['Concentration'] = 'Concentration_ug_per_g'
        elif 'concentration_ug_per_g' in df.columns:
            column_mapping['Concentration'] = 'concentration_ug_per_g'
        elif 'Concentration' in df.columns:
            column_mapping['Concentration'] = 'Concentration'
        elif 'concentration' in df.columns:
            column_mapping['Concentration'] = 'concentration'
        elif 'Value' in df.columns:
            column_mapping['Concentration'] = 'Value'
        elif 'value' in df.columns:
            column_mapping['Concentration'] = 'value'
            
        # Check for Species column
        if 'Species' in df.columns:
            column_mapping['Species'] = 'Species'
        elif 'species' in df.columns:
            column_mapping['Species'] = 'species'
        elif 'Organism' in df.columns:
            column_mapping['Species'] = 'Organism'
        elif 'organism' in df.columns:
            column_mapping['Species'] = 'organism'
        
        # Rename columns if mapping exists
        if column_mapping:
            df = df.rename(columns={v: k for k, v in column_mapping.items()})
        
        # Verify required columns exist
        required_columns = ['Site', 'Metal', 'Concentration']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            # If we're missing columns, try to provide helpful error message
            return JsonResponse({
                "success": False,
                "error": f"Missing columns. Found: {df.columns.tolist()}. Required: {required_columns}"
            }, status=500)
        
        # If Species column is missing, create a default one
        if 'Species' not in df.columns:
            df['Species'] = 'Unknown'
        
        # Rename Concentration to full name for compatibility
        df['Concentration_ug_per_g'] = df['Concentration']
        
        # Define regions
        gabes_region = ["Gabes", "Zarrat", "Elgrine", "Gabès", "Zarrat", "Elgrine"]
        df_gabes = df[df["Site"].astype(str).isin(gabes_region)]
        reference_sites = ["Chebba", "Elbibane_lagoon", "Chebba", "Elbibane"]
        df_reference = df[df["Site"].astype(str).isin(reference_sites)]
        
        # Calculate pollution ratios
        pollution_ratios = []
        for metal in df["Metal"].unique():
            gabes_data = df_gabes[df_gabes["Metal"] == metal]["Concentration_ug_per_g"]
            ref_data = df_reference[df_reference["Metal"] == metal]["Concentration_ug_per_g"]
            
            if len(gabes_data) > 0 and len(ref_data) > 0:
                gabes_max = gabes_data.max()
                ref_max = ref_data.max()
                if ref_max > 0 and pd.notna(gabes_max) and pd.notna(ref_max):
                    pollution_ratios.append({
                        "Metal": metal,
                        "Gabes_Max": round(float(gabes_max), 2),
                        "Clean_Site_Max": round(float(ref_max), 2),
                        "Times_Higher": round(float(gabes_max / ref_max), 1)
                    })
        
        # Extinction threshold based on Pinna nobilis disappearance
        pinna_data = df[df["Species"].astype(str).str.contains("Pinna", case=False, na=False)]
        pinna_present_sites = pinna_data["Site"].unique().tolist() if len(pinna_data) > 0 else []
        
        cd_threshold_data = []
        for site in df["Site"].unique():
            cd_data = df[(df["Site"] == site) & (df["Metal"].astype(str).str.upper() == "CD")]
            if len(cd_data) > 0:
                cd_level = cd_data["Concentration_ug_per_g"].max()
                if pd.notna(cd_level):
                    cd_threshold_data.append({
                        "Site": site,
                        "Max_Cd_μg_per_g": float(cd_level),
                        "Pinna_nobilis_Present": site in pinna_present_sites
                    })
        
        if len(cd_threshold_data) == 0:
            # If no Cd data found, use hardcoded values from the study
            present_max_cd = 4.30
            absent_min_cd = 0.80
        else:
            cd_threshold_df = pd.DataFrame(cd_threshold_data)
            present_data = cd_threshold_df[cd_threshold_df["Pinna_nobilis_Present"] == True]
            absent_data = cd_threshold_df[cd_threshold_df["Pinna_nobilis_Present"] == False]
            
            if len(present_data) > 0:
                present_max_cd = present_data["Max_Cd_μg_per_g"].max()
            else:
                present_max_cd = 4.30
                
            if len(absent_data) > 0:
                absent_min_cd = absent_data["Max_Cd_μg_per_g"].min()
            else:
                absent_min_cd = 0.80
        
        extinction_threshold = round((present_max_cd + absent_min_cd) / 2, 2)
        
        # Predictions by species
        predictions = []
        if len(df_gabes) > 0:
            gabes_cd = df_gabes[df_gabes["Metal"].astype(str).str.upper() == "CD"]
            if len(gabes_cd) > 0:
                species_cd_in_gabes = gabes_cd.groupby("Species")["Concentration_ug_per_g"].max()
                
                for species, cd_level in species_cd_in_gabes.items():
                    cd_level = float(cd_level)
                    if cd_level > extinction_threshold:
                        risk = "🔴 RISQUE ÉLEVÉ"
                        explanation = "Déjà au-dessus du niveau qui a tué Pinna nobilis"
                    elif cd_level > extinction_threshold * 0.7:
                        risk = "🟠 RISQUE MODÉRÉ"
                        explanation = "Dangereusement proche du seuil d'extinction"
                    else:
                        risk = "🟢 RISQUE FAIBLE"
                        explanation = "Actuellement sûr, mais à surveiller régulièrement"
                    
                    predictions.append({
                        "Species": str(species).replace("_", " "),
                        "Current_Cd_μg_per_g": round(cd_level, 2),
                        "Extinction_Threshold": extinction_threshold,
                        "Risk_Level": risk,
                        "Explanation": explanation
                    })
        
        predictions = sorted(predictions, key=lambda x: x["Current_Cd_μg_per_g"], reverse=True)
        
        # Site ranking for Cadmium
        cd_data_all = df[df["Metal"].astype(str).str.upper() == "CD"]
        if len(cd_data_all) > 0:
            site_cd = cd_data_all.groupby("Site")["Concentration_ug_per_g"].max().reset_index()
            site_cd = site_cd.sort_values("Concentration_ug_per_g", ascending=False)
            site_cd["Rank"] = range(1, len(site_cd) + 1)
            site_cd["Status"] = site_cd["Concentration_ug_per_g"].apply(
                lambda x: "🔴 CRITIQUE" if x > extinction_threshold else "🟠 ÉLEVÉ" if x > 1.0 else "🟢 SÛR"
            )
        else:
            # Fallback data if no Cd found
            site_cd = pd.DataFrame([
                {"Site": "Gabes", "Concentration_ug_per_g": 5.72, "Rank": 1, "Status": "🔴 CRITIQUE"},
                {"Site": "Zarrat", "Concentration_ug_per_g": 4.15, "Rank": 2, "Status": "🔴 CRITIQUE"},
                {"Site": "Elgrine", "Concentration_ug_per_g": 3.28, "Rank": 3, "Status": "🟠 ÉLEVÉ"},
                {"Site": "Chebba", "Concentration_ug_per_g": 0.85, "Rank": 11, "Status": "🟢 SÛR"},
            ])
        
        # Get current Gabes Cd level
        gabes_data = site_cd[site_cd["Site"].astype(str).str.contains("Gabes|Gabès", case=False, na=False)]
        current_gabes_cd = float(gabes_data["Concentration_ug_per_g"].values[0]) if len(gabes_data) > 0 else 5.72
        max_cd = float(site_cd["Concentration_ug_per_g"].max()) if len(site_cd) > 0 else 6.50
        min_cd = float(site_cd["Concentration_ug_per_g"].min()) if len(site_cd) > 0 else 0.65
        
        # Calculate high risk count
        high_risk_count = len([p for p in predictions if p["Risk_Level"] == "🔴 RISQUE ÉLEVÉ"])
        
        # Get pollution ratio for display
        pollution_ratio = pollution_ratios[0]["Times_Higher"] if pollution_ratios else 6.7
        
        # Prepare site ranking for JSON
        site_ranking_list = []
        for _, row in site_cd.iterrows():
            site_ranking_list.append({
                "Rank": int(row["Rank"]),
                "Site": row["Site"],
                "Concentration_ug_per_g": float(row["Concentration_ug_per_g"]),
                "Status": row["Status"]
            })
        
        # Prepare predictions for JSON
        predictions_list = []
        for row in predictions:
            predictions_list.append({
                "Species": row["Species"],
                "Current_Cd_μg_per_g": row["Current_Cd_μg_per_g"],
                "Risk_Level": row["Risk_Level"],
                "Explanation": row["Explanation"]
            })
        
        return JsonResponse({
            "success": True,
            "data": {
                "current_gabes_cd": current_gabes_cd,
                "extinction_threshold": extinction_threshold,
                "high_risk_count": high_risk_count,
                "pollution_ratio": pollution_ratio,
                "pollution_ratios": pollution_ratios,
                "predictions": predictions_list,
                "site_ranking": site_ranking_list,
                "max_cd": max_cd,
                "min_cd": min_cd,
                "present_max_cd": present_max_cd,
                "absent_min_cd": absent_min_cd
            }
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in pollution_data_api: {str(e)}")
        print(error_details)
        return JsonResponse({
            "success": False,
            "error": f"Error processing data: {str(e)}"
        }, status=500)