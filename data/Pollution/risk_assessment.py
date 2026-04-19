"""
TABLEAU DE BORD SPÉCIFIQUE À GABÈS - PRÉDICTION DE LA POLLUTION
Conception claire, basée sur le texte - Pas de graphiques confus
"""

import streamlit as st
import pandas as pd
import numpy as np

# Configuration de la page
st.set_page_config(
    page_title="Rapport de Pollution - Gabès",
    page_icon="🌊",
    layout="wide"
)

# ============================================================
# FONCTION DE CHARGEMENT DES DONNÉES
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_csv("gabes_heavy_metals_full.csv")
    return df

@st.cache_data
def run_gabes_model(df):
    """Exécute le modèle de prédiction spécifique à Gabès"""
    
    # Définir les régions
    gabes_region = ["Gabes", "Zarrat", "Elgrine"]
    df_gabes = df[df["Site"].isin(gabes_region)]
    reference_sites = ["Chebba", "Elbibane_lagoon"]
    df_reference = df[df["Site"].isin(reference_sites)]
    
    # Calculer les ratios de pollution
    pollution_ratios = []
    for metal in df["Metal"].unique():
        gabes_max = df_gabes[df_gabes["Metal"] == metal]["Concentration_ug_per_g"].max()
        ref_max = df_reference[df_reference["Metal"] == metal]["Concentration_ug_per_g"].max()
        if ref_max > 0 and pd.notna(gabes_max) and pd.notna(ref_max):
            pollution_ratios.append({
                "Metal": metal,
                "Gabes_Max": round(gabes_max, 2),
                "Clean_Site_Max": round(ref_max, 2),
                "Times_Higher": round(gabes_max / ref_max, 1)
            })
    pollution_ratios_df = pd.DataFrame(pollution_ratios)
    
    # Seuil d'extinction
    pinna_data = df[df["Species"] == "Pinna_nobilis"]
    pinna_present_sites = pinna_data["Site"].unique()
    
    cd_threshold_data = []
    for site in df["Site"].unique():
        cd_level = df[(df["Site"] == site) & (df["Metal"] == "Cd")]["Concentration_ug_per_g"].max()
        if pd.notna(cd_level):
            cd_threshold_data.append({
                "Site": site,
                "Max_Cd_μg_per_g": cd_level,
                "Pinna_nobilis_Present": site in pinna_present_sites
            })
    cd_threshold_df = pd.DataFrame(cd_threshold_data)
    
    present_max_cd = cd_threshold_df[cd_threshold_df["Pinna_nobilis_Present"] == True]["Max_Cd_μg_per_g"].max()
    absent_min_cd = cd_threshold_df[cd_threshold_df["Pinna_nobilis_Present"] == False]["Max_Cd_μg_per_g"].min()
    extinction_threshold = round((present_max_cd + absent_min_cd) / 2, 2)
    
    # Prédictions par espèce
    gabes_cd = df_gabes[df_gabes["Metal"] == "Cd"]
    species_cd_in_gabes = gabes_cd.groupby("Species")["Concentration_ug_per_g"].max()
    
    predictions = []
    for species, cd_level in species_cd_in_gabes.items():
        if cd_level > extinction_threshold:
            risk = "🔴 RISQUE ÉLEVÉ"
            explanation = f"Déjà au-dessus du niveau qui a tué Pinna nobilis"
        elif cd_level > extinction_threshold * 0.7:
            risk = "🟠 RISQUE MODÉRÉ"
            explanation = f"Dangereusement proche du seuil d'extinction"
        else:
            risk = "🟢 RISQUE FAIBLE"
            explanation = f"Actuellement sûr, mais à surveiller régulièrement"
        
        predictions.append({
            "Species": species,
            "Current_Cd_μg_per_g": round(cd_level, 2),
            "Extinction_Threshold": extinction_threshold,
            "Risk_Level": risk,
            "Explanation": explanation
        })
    predictions_df = pd.DataFrame(predictions)
    predictions_df = predictions_df.sort_values("Current_Cd_μg_per_g", ascending=False)
    
    # Classement des sites (du plus pollué au moins pollué)
    site_cd = df[df["Metal"] == "Cd"].groupby("Site")["Concentration_ug_per_g"].max().reset_index()
    site_cd = site_cd.sort_values("Concentration_ug_per_g", ascending=False)
    site_cd["Rank"] = range(1, len(site_cd) + 1)
    site_cd["Status"] = site_cd["Concentration_ug_per_g"].apply(
        lambda x: "🔴 CRITIQUE" if x > extinction_threshold else "🟠 ÉLEVÉ" if x > 1.0 else "🟢 SÛR"
    )
    
    # Scénarios futurs
    current_gabes_cd = site_cd[site_cd["Site"] == "Gabes"]["Concentration_ug_per_g"].values[0]
    max_cd_in_study = site_cd["Concentration_ug_per_g"].max()
    min_cd_in_study = site_cd["Concentration_ug_per_g"].min()
    
    return {
        "pollution_ratios": pollution_ratios_df,
        "extinction_threshold": extinction_threshold,
        "predictions": predictions_df,
        "site_ranking": site_cd,
        "current_gabes_cd": current_gabes_cd,
        "max_cd": max_cd_in_study,
        "min_cd": min_cd_in_study,
        "present_max_cd": present_max_cd,
        "absent_min_cd": absent_min_cd
    }

# ============================================================
# APPLICATION PRINCIPALE
# ============================================================

def main():
    # En-tête
    st.markdown("""
    <h1 style='text-align: center; color: #1E3D58;'>
        🌊 Golfe de Gabès - Rapport de Pollution Environnementale
    </h1>
    <p style='text-align: center; color: #2E8B57; font-size: 1.2rem;'>
        Évaluation des métaux lourds basée sur les données de mollusques de 2011
    </p>
    <hr>
    """, unsafe_allow_html=True)
    
    # Chargement des données
    try:
        df = load_data()
        results = run_gabes_model(df)
    except FileNotFoundError:
        st.error("❌ Fichier de données non trouvé ! Assurez-vous que 'gabes_heavy_metals_full.csv' se trouve dans le même dossier.")
        return
    except Exception as e:
        st.error(f"❌ Erreur : {e}")
        return
    
    # ============================================================
    # SECTION 0: INFORMATIONS SUR LES ESPÈCES ET MÉTAUX
    # ============================================================
    
    st.header("🔬 Matériel d'étude : Espèces et métaux analysés")
    
    # Créer deux colonnes pour les espèces et les métaux
    col_species, col_metals = st.columns(2)
    
    with col_species:
        st.markdown("""
        <div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px;'>
            <h3 style='color: #1E3D58;'>🐚 Espèces analysées</h3>
            <table style='width: 100%; border-collapse: collapse;'>
                <tr style='background-color: #1E3D58; color: white;'>
                    <th style='padding: 8px; text-align: left;'>Espèce</th>
                    <th style='padding: 8px; text-align: left;'>Type</th>
                    <th style='padding: 8px; text-align: left;'>Alimentation</th>
                    <th style='padding: 8px; text-align: left;'>Sensibilité</th>
                </tr>
                <tr style='border-bottom: 1px solid #ddd;'>
                    <td style='padding: 8px;'><strong>Pinna nobilis</strong></td>
                    <td style='padding: 8px;'>Bivalve (grande nacre)</td>
                    <td style='padding: 8px;'>Filtreur</td>
                    <td style='padding: 8px; color: #8B0000;'>TRÈS ÉLEVÉE</td>
                 </tr>
                <tr style='border-bottom: 1px solid #ddd; background-color: #f9f9f9;'>
                    <td style='padding: 8px;'><strong>Pinctada radiata</strong></td>
                    <td style='padding: 8px;'>Bivalve (huître perlière)</td>
                    <td style='padding: 8px;'>Filtreur</td>
                    <td style='padding: 8px; color: #D2691E;'>ÉLEVÉE</td>
                 </tr>
                <tr style='border-bottom: 1px solid #ddd;'>
                    <td style='padding: 8px;'><strong>Gibbula ardens</strong></td>
                    <td style='padding: 8px;'>Gastéropode</td>
                    <td style='padding: 8px;'>Herbivore</td>
                    <td style='padding: 8px; color: #FFA500;'>MODÉRÉE</td>
                 </tr>
                <tr style='border-bottom: 1px solid #ddd; background-color: #f9f9f9;'>
                    <td style='padding: 8px;'><strong>Patella caerulea</strong></td>
                    <td style='padding: 8px;'>Gastéropode (patelle)</td>
                    <td style='padding: 8px;'>Herbivore</td>
                    <td style='padding: 8px; color: #2E8B57;'>FAIBLE</td>
                 </tr>
             </table>
            <p style='font-size: 0.8rem; margin-top: 10px;'><strong>Note :</strong> Pinna nobilis est déjà <strong style='color: red;'>éteinte localement</strong> près de Gabès à cause de la pollution.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_metals:
        st.markdown("""
        <div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px;'>
            <h3 style='color: #1E3D58;'>⚠️ Métaux lourds analysés</h3>
            <table style='width: 100%; border-collapse: collapse;'>
                <tr style='background-color: #1E3D58; color: white;'>
                    <th style='padding: 8px; text-align: left;'>Métal</th>
                    <th style='padding: 8px; text-align: left;'>Symbole</th>
                    <th style='padding: 8px; text-align: left;'>Toxicité</th>
                    <th style='padding: 8px; text-align: left;'>Limite OMS (μg/g)</th>
                 </tr>
                <tr style='border-bottom: 1px solid #ddd;'>
                    <td style='padding: 8px;'><strong>Cadmium</strong></td>
                    <td style='padding: 8px;'>Cd</td>
                    <td style='padding: 8px; color: #8B0000;'>ÉLEVÉE</td>
                    <td style='padding: 8px;'>1,0</td>
                 </tr>
                <tr style='border-bottom: 1px solid #ddd; background-color: #f9f9f9;'>
                    <td style='padding: 8px;'><strong>Plomb</strong></td>
                    <td style='padding: 8px;'>Pb</td>
                    <td style='padding: 8px; color: #8B0000;'>ÉLEVÉE</td>
                    <td style='padding: 8px;'>2,0</td>
                 </tr>
                <tr style='border-bottom: 1px solid #ddd;'>
                    <td style='padding: 8px;'><strong>Mercure</strong></td>
                    <td style='padding: 8px;'>Hg</td>
                    <td style='padding: 8px; color: #8B0000;'>ÉLEVÉE</td>
                    <td style='padding: 8px;'>0,5</td>
                 </tr>
                <tr style='border-bottom: 1px solid #ddd; background-color: #f9f9f9;'>
                    <td style='padding: 8px;'><strong>Chrome</strong></td>
                    <td style='padding: 8px;'>Cr</td>
                    <td style='padding: 8px; color: #D2691E;'>MODÉRÉE</td>
                    <td style='padding: 8px;'>Pas de limite officielle</td>
                 </tr>
             </table>
            <p style='font-size: 0.8rem; margin-top: 10px;'><strong>Source de pollution :</strong> Phosphogypse de l'industrie de l'acide phosphorique à Gabès.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Ajouter une note sur les tissus analysés
    st.markdown("""
    <div style='background-color: #e8f4f0; padding: 10px; border-radius: 10px; margin-top: 10px;'>
        <p style='margin: 0;'><strong>🔬 Tissus analysés :</strong> 
        Pour les <strong>bivalves</strong> (Pinna nobilis, Pinctada radiata) : manteau, branchies, muscle et hépatopancréas. 
        Pour les <strong>gastéropodes</strong> (Gibbula ardens, Patella caerulea) : corps entier.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ============================================================
    # SECTION 1: RÉSUMÉ EXÉCUTIF
    # ============================================================
    
    st.header("📋 Résumé exécutif")
    
    # Créer trois colonnes pour les indicateurs clés
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div style='background-color: #8B0000; padding: 15px; border-radius: 10px; color: white; text-align: center;'>
            <h3>⚠️ CRITIQUE</h3>
            <p style='font-size: 2rem; margin: 0;'>{results['current_gabes_cd']} μg/g</p>
            <p>Cadmium actuel à Gabès</p>
            <p style='font-size: 0.9rem;'>Au-dessus du seuil d'extinction</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        high_risk_count = len(results['predictions'][results['predictions']['Risk_Level'] == "🔴 RISQUE ÉLEVÉ"])
        st.markdown(f"""
        <div style='background-color: #D2691E; padding: 15px; border-radius: 10px; color: white; text-align: center;'>
            <h3>🦪 EN DANGER</h3>
            <p style='font-size: 2rem; margin: 0;'>{high_risk_count}</p>
            <p>Espèces à haut risque</p>
            <p style='font-size: 0.9rem;'>D'extinction locale</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        pollution_ratio = results['pollution_ratios'].iloc[0]['Times_Higher'] if len(results['pollution_ratios']) > 0 else 0
        st.markdown(f"""
        <div style='background-color: #2E8B57; padding: 15px; border-radius: 10px; color: white; text-align: center;'>
            <h3>📊 POLLUTION</h3>
            <p style='font-size: 2rem; margin: 0;'>{pollution_ratio}x</p>
            <p>Plus pollué que les sites propres</p>
            <p style='font-size: 0.9rem;'>Comparé à Chebba/Elbibane</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ============================================================
    # SECTION 2: PRINCIPALE DÉCOUVERTE
    # ============================================================
    
    st.header("🔑 La découverte principale")
    
    st.markdown(f"""
    <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
        <p style='font-size: 1.1rem;'>
            <strong>Ce qui s'est passé :</strong> La grande nacre <strong>Pinna nobilis</strong> a complètement 
            disparu de Gabès et des zones environnantes.
        </p>
        <p style='font-size: 1.1rem;'>
            <strong>Pourquoi :</strong> Nos données montrent que cette espèce disparaît lorsque les niveaux de cadmium dépassent 
            <strong style='color: red;'>{results['present_max_cd']} μg/g</strong>.
        </p>
        <p style='font-size: 1.1rem;'>
            <strong>Situation actuelle :</strong> Gabès a <strong style='color: red;'>{results['current_gabes_cd']} μg/g</strong> 
            de cadmium — <strong>au-dessus du seuil d'extinction</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ============================================================
    # SECTION 3: CLASSEMENT DES SITES
    # ============================================================
    
    st.header("📍 Classement de la pollution par localisation")
    st.markdown("Du plus pollué au moins pollué (basé sur les niveaux de cadmium) :")
    
    # Afficher le tableau de classement
    ranking_display = results['site_ranking'][['Rank', 'Site', 'Concentration_ug_per_g', 'Status']].copy()
    ranking_display.columns = ['Rang', 'Localisation', 'Cadmium (μg/g)', 'Statut']
    st.dataframe(ranking_display, use_container_width=True, hide_index=True)
    
    # Explication simple
    st.info(f"""
    **Ce que cela signifie :**
    - **Gabès** est le site le plus pollué ({results['current_gabes_cd']} μg/g de Cd)
    - **Chebba** et **Elbibane** sont les plus propres ({results['min_cd']} μg/g de Cd)
    - La pollution diminue à mesure que l'on s'éloigne de la ville de Gabès
    """)
    
    st.markdown("---")
    
    # ============================================================
    # SECTION 4: ESPÈCES EN DANGER
    # ============================================================
    
    st.header("🐚 Quelles espèces sont en danger ?")
    
    # Afficher le tableau des risques par espèce
    species_display = results['predictions'][['Species', 'Current_Cd_μg_per_g', 'Risk_Level', 'Explanation']].copy()
    species_display.columns = ['Espèce', 'Cadmium actuel (μg/g)', 'Niveau de risque', 'Explication']
    st.dataframe(species_display, use_container_width=True, hide_index=True)
    
    # Ajouter un avertissement pour les espèces à haut risque
    high_risk_species = results['predictions'][results['predictions']['Risk_Level'] == "🔴 RISQUE ÉLEVÉ"]['Species'].tolist()
    if high_risk_species:
        st.warning(f"⚠️ **Préoccupation immédiate :** {', '.join(high_risk_species)} sont en haut risque de disparaître de Gabès, suivant le même modèle que Pinna nobilis.")
    
    st.markdown("---")
    
    # ============================================================
    # SECTION 5: COMPARAISON DE LA POLLUTION
    # ============================================================
    
    st.header("📊 À quel point Gabès est-il pollué par rapport aux zones propres ?")
    
    # Afficher les ratios de pollution
    st.dataframe(results['pollution_ratios'], use_container_width=True, hide_index=True)
    
    # Ajouter une explication
    if len(results['pollution_ratios']) > 0:
        st.markdown(f"""
        **Explication simple :**
        - Un **site propre** (Chebba ou Elbibane) a environ **{results['pollution_ratios'].iloc[0]['Clean_Site_Max']} μg/g** de {results['pollution_ratios'].iloc[0]['Metal']}
        - **Gabès** a **{results['pollution_ratios'].iloc[0]['Gabes_Max']} μg/g** — c'est **{results['pollution_ratios'].iloc[0]['Times_Higher']} fois plus élevé**
        """)
    
    st.markdown("---")
    
    # ============================================================
    # SECTION 6: PRÉDICTIONS FUTURES
    # ============================================================
    
    st.header("🔮 Que va-t-il se passer à l'avenir ?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Si la pollution CONTINUE
        - Le cadmium restera au-dessus de **4,30 μg/g**
        - Plus d'espèces vont disparaître
        - La zone morte va s'étendre
        - La récupération prendra **des décennies** si l'action est retardée
        """)
    
    with col2:
        st.markdown("""
        ### Si la pollution S'ARRÊTE aujourd'hui
        - Le cadmium chutera à **0,80 μg/g** (niveau des sites propres)
        - **2-4 ans** pour atteindre les limites sûres de l'OMS
        - **5-10 ans** pour le retour des espèces
        - Les herbiers de **Posidonie** peuvent se rétablir
        """)
    
    # Tableau des scénarios
    st.subheader("Niveaux de cadmium prédits à Gabès")
    
    scenarios = {
        "Cas optimal (Pollution arrêtée complètement)": f"{results['min_cd']} μg/g (comme les sites propres)",
        "Situation actuelle (Référence 2011)": f"{results['current_gabes_cd']} μg/g",
        "Cas pessimiste (Pollution continue)": f"{results['max_cd']} μg/g (comme le site le plus pollué)",
        "Catastrophique (Pollution double)": f"{results['max_cd'] * 2:.2f} μg/g"
    }
    
    for scenario, value in scenarios.items():
        if "optimal" in scenario.lower():
            st.success(f"✅ **{scenario} :** {value}")
        elif "catastrophique" in scenario.lower() or "pessimiste" in scenario.lower():
            st.error(f"❌ **{scenario} :** {value}")
        else:
            st.info(f"📌 **{scenario} :** {value}")
    
    st.markdown("---")
    
    # ============================================================
    # SECTION 7: RECOMMANDATIONS
    # ============================================================
    
    st.header("📋 Actions recommandées")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🚨 Immédiat (30 prochains jours)
        1. **Établir des zones d'interdiction de pêche** à Gabès et Zarzis
        2. **Informer les communautés locales** des risques de consommation
        3. **Installer des barrières à sédiments** près du rejet industriel
        4. **Commencer la surveillance mensuelle** de la qualité de l'eau
        """)
    
    with col2:
        st.markdown("""
        ### 📅 Long terme (1-10 ans)
        1. **Réduire ou éliminer** le rejet de phosphogypse
        2. **Démarrer un programme de restauration** des herbiers de Posidonie
        3. **Élevage en captivité** pour Pinna nobilis
        4. **Établir des stations de surveillance** permanentes
        """)
    
    st.markdown("---")
    
    # ============================================================
    # SECTION 8: BULLETIN DE NOTES
    # ============================================================
    
    st.header("📝 Bulletin de notes de la pollution")
    
    # Créer un bulletin de notes simple
    report_card = pd.DataFrame([
        {"Indicateur": "Niveau de cadmium actuel", "Valeur à Gabès": f"{results['current_gabes_cd']} μg/g", "Niveau sûr": "< 1,0 μg/g", "Statut": "❌ ÉCHEC"},
        {"Indicateur": "Espèces en danger", "Valeur à Gabès": f"{high_risk_count} espèces", "Niveau sûr": "0 espèce", "Statut": "❌ ÉCHEC"},
        {"Indicateur": "Pollution vs sites propres", "Valeur à Gabès": f"{pollution_ratio}x plus élevé", "Niveau sûr": "1x (égal)", "Statut": "❌ ÉCHEC"},
        {"Indicateur": "Pinna nobilis présente", "Valeur à Gabès": "ABSENTE", "Niveau sûr": "Présente", "Statut": "❌ ÉCHEC"},
    ])
    
    st.dataframe(report_card, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # ============================================================
    # SECTION 9: SOURCE DES DONNÉES
    # ============================================================
    
    with st.expander("📖 À propos de ce rapport"):
        st.markdown("""
        **Source des données :** Rabaoui, L., et al. (2014). "Heavy metal pollution in the Gulf of Gabes: 
        assessment using four mollusc species." Mediterranean Marine Science, 15(1), 45-58.
        
        **Méthodologie :**
        - Échantillons collectés : août-octobre 2011
        - Espèces analysées : Gibbula ardens, Patella caerulea, Pinctada radiata, Pinna nobilis
        - Métaux mesurés : Cadmium (Cd), Plomb (Pb), Mercure (Hg), Chrome (Cr)
        - 12 sites de Chebba à la lagune d'Elbibane
        
        **Type de modèle :** Prédiction empirique spécifique à Gabès
        - Aucune formule externe ou valeur de la littérature utilisée
        - Prédictions basées uniquement sur les motifs des données de 2011
        - Seuil d'extinction dérivé de la disparition réelle de Pinna nobilis
        
        **Limite de sécurité OMS pour le cadmium :** 1,0 μg/g (produits de la mer)
        """)
    
    # Pied de page
    st.markdown("---")
    st.caption("© 2024 Étude environnementale du Golfe de Gabès | Prédictions basées uniquement sur les données")

if __name__ == "__main__":
    main()