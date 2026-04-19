import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "POWER_Point_Daily_20200101_20241231_033d90N_010d10E_LST.csv"


def load_nasa_power_csv(file_path: Path) -> pd.DataFrame:
    header_row = None
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if "YEAR" in line and "DOY" in line and "T2M" in line:
                header_row = i
                break

    if header_row is None:
        raise ValueError("Impossible de détecter l'en-tête NASA POWER.")

    return pd.read_csv(file_path, skiprows=header_row)


def prepare_base_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = [
        "YEAR", "DOY", "T2M", "T2M_MAX", "T2M_MIN",
        "PRECTOTCORR", "RH2M", "WS10M", "GWETROOT"
    ]
    df = df[required_cols].copy()

    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().copy()
    df["DATE"] = pd.to_datetime(df["YEAR"].astype(int).astype(str), format="%Y") + pd.to_timedelta(df["DOY"] - 1, unit="D")
    df["month"] = df["DATE"].dt.month
    df["dayofyear"] = df["DATE"].dt.dayofyear
    df = df.sort_values("DATE").reset_index(drop=True)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["temp_range"] = df["T2M_MAX"] - df["T2M_MIN"]
    df["soil_deficit"] = 1 - df["GWETROOT"]

    for col in ["T2M_MAX", "GWETROOT", "RH2M", "WS10M", "PRECTOTCORR"]:
        df[f"{col}_roll7"] = df[col].rolling(7).mean()
        df[f"{col}_roll14"] = df[col].rolling(14).mean()

    df["PRECTOTCORR_cum7"] = df["PRECTOTCORR"].rolling(7).sum()
    df["PRECTOTCORR_cum30"] = df["PRECTOTCORR"].rolling(30).sum()

    # jours secs consécutifs
    dry = (df["PRECTOTCORR"] < 0.1).astype(int)
    streak = []
    c = 0
    for v in dry:
        if v == 1:
            c += 1
        else:
            c = 0
        streak.append(c)
    df["dry_days_streak"] = streak

    # jours chauds consécutifs
    hot = (df["T2M_MAX"] > 35).astype(int)
    streak = []
    c = 0
    for v in hot:
        if v == 1:
            c += 1
        else:
            c = 0
        streak.append(c)
    df["hot_days_streak"] = streak

    return df.dropna().reset_index(drop=True)


def profile_from_cluster_stats(row: pd.Series) -> tuple[str, str]:
    # logique d'interprétation métier simple et stable
    if row["GWETROOT"] < 0.23 and row["T2M_MAX"] > 32:
        return "stress hydrique élevé", "Humidité du sol faible, chaleur forte, conditions critiques."
    if row["GWETROOT"] < 0.24 and row["PRECTOTCORR"] < 0.2:
        return "stress hydrique modéré", "Déficit hydrique modéré avec faible pluviométrie."
    if row["GWETROOT"] >= 0.24 and row["T2M_MAX"] < 28:
        return "conditions équilibrées", "Conditions plus favorables avec humidité du sol correcte."
    return "conditions intermédiaires", "Profil intermédiaire ou instable entre plusieurs régimes climatiques."


def build_clustering_model(n_clusters: int = 4):
    raw_df = load_nasa_power_csv(DATA_PATH)
    df = prepare_base_dataframe(raw_df)
    df = add_features(df)

    feature_cols = [
        "T2M", "T2M_MAX", "T2M_MIN", "PRECTOTCORR", "RH2M", "WS10M", "GWETROOT",
        "temp_range", "soil_deficit",
        "T2M_MAX_roll7", "T2M_MAX_roll14",
        "GWETROOT_roll7", "GWETROOT_roll14",
        "RH2M_roll7", "RH2M_roll14",
        "WS10M_roll7", "WS10M_roll14",
        "PRECTOTCORR_roll7", "PRECTOTCORR_roll14",
        "PRECTOTCORR_cum7", "PRECTOTCORR_cum30",
        "dry_days_streak", "hot_days_streak",
        "month", "dayofyear"
    ]

    X = df[feature_cols].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    df["cluster_id"] = kmeans.fit_predict(X_scaled)

    cluster_stats = df.groupby("cluster_id")[[
        "T2M_MAX", "GWETROOT", "PRECTOTCORR", "RH2M", "WS10M"
    ]].mean().reset_index()

    cluster_profiles = {}
    for _, row in cluster_stats.iterrows():
        profile, summary = profile_from_cluster_stats(row)
        cluster_profiles[int(row["cluster_id"])] = {
            "profile": profile,
            "summary": summary
        }

    return {
        "df": df,
        "feature_cols": feature_cols,
        "scaler": scaler,
        "kmeans": kmeans,
        "cluster_profiles": cluster_profiles
    }


def get_cluster_for_date(selected_date: str):
    model_bundle = build_clustering_model()
    df = model_bundle["df"]

    selected_date = pd.to_datetime(selected_date)
    row = df[df["DATE"] == selected_date]

    if row.empty:
        nearest_idx = (df["DATE"] - selected_date).abs().idxmin()
        row = df.loc[[nearest_idx]]

    cluster_id = int(row.iloc[0]["cluster_id"])
    profile_info = model_bundle["cluster_profiles"][cluster_id]

    return {
        "date_used": str(row.iloc[0]["DATE"].date()),
        "cluster_id": cluster_id,
        "cluster_profile": profile_info["profile"],
        "cluster_summary": profile_info["summary"],
        "cluster_metrics": {
            "T2M_MAX": float(row.iloc[0]["T2M_MAX"]),
            "GWETROOT": float(row.iloc[0]["GWETROOT"]),
            "PRECTOTCORR": float(row.iloc[0]["PRECTOTCORR"]),
            "RH2M": float(row.iloc[0]["RH2M"]),
            "WS10M": float(row.iloc[0]["WS10M"]),
        }
    }