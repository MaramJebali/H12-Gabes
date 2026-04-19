from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .data_loader import load_nasa_power_csv

FORECAST_HORIZON = 7
BASE_FEATURES = [
    'YEAR', 'DOY', 'T2M', 'T2M_MAX', 'T2M_MIN',
    'PRECTOTCORR', 'RH2M', 'WS10M', 'GWETROOT'
]


@dataclass
class PredictionResult:
    selected_date: str
    current_gwetroot: float
    current_t2m_max: float
    predicted_gwetroot_tplus7: float
    predicted_t2m_max_tplus7: float
    alert_level: str
    risk_score: int
    dominant_factors: List[str]


def _prepare_base(df: pd.DataFrame) -> pd.DataFrame:
    df = df[BASE_FEATURES].copy()
    for col in BASE_FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['T2M_MAX', 'PRECTOTCORR', 'RH2M', 'WS10M', 'GWETROOT']).copy()
    df['DATE'] = pd.to_datetime(df['YEAR'].astype(int).astype(str), format='%Y') + pd.to_timedelta(df['DOY'] - 1, unit='D')
    df['month'] = df['DATE'].dt.month
    df['dayofyear'] = df['DATE'].dt.dayofyear
    df = df.sort_values('DATE').reset_index(drop=True)
    return df


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['temp_range'] = df['T2M_MAX'] - df['T2M_MIN']

    for col in ['T2M_MAX', 'GWETROOT', 'RH2M', 'WS10M', 'PRECTOTCORR']:
        df[f'{col}_roll7'] = df[col].rolling(7).mean()
        df[f'{col}_roll14'] = df[col].rolling(14).mean()
        df[f'{col}_lag1'] = df[col].shift(1)
        df[f'{col}_lag3'] = df[col].shift(3)
        df[f'{col}_lag7'] = df[col].shift(7)

    df['PRECTOTCORR_cum7'] = df['PRECTOTCORR'].rolling(7).sum()
    df['PRECTOTCORR_cum30'] = df['PRECTOTCORR'].rolling(30).sum()

    dry = (df['PRECTOTCORR'] < 0.1).astype(int)
    hot = (df['T2M_MAX'] > 35).astype(int)

    dry_streak, hot_streak = [], []
    d = h = 0
    for is_dry, is_hot in zip(dry, hot):
        d = d + 1 if is_dry else 0
        h = h + 1 if is_hot else 0
        dry_streak.append(d)
        hot_streak.append(h)

    df['dry_days_streak'] = dry_streak
    df['hot_days_streak'] = hot_streak
    return df


FEATURE_COLUMNS = [
    'T2M', 'T2M_MAX', 'T2M_MIN', 'PRECTOTCORR', 'RH2M', 'WS10M', 'GWETROOT',
    'month', 'dayofyear', 'temp_range',
    'T2M_MAX_roll7', 'T2M_MAX_roll14',
    'GWETROOT_roll7', 'GWETROOT_roll14',
    'RH2M_roll7', 'RH2M_roll14',
    'WS10M_roll7', 'WS10M_roll14',
    'PRECTOTCORR_roll7', 'PRECTOTCORR_roll14',
    'PRECTOTCORR_cum7', 'PRECTOTCORR_cum30',
    'T2M_MAX_lag1', 'T2M_MAX_lag3', 'T2M_MAX_lag7',
    'GWETROOT_lag1', 'GWETROOT_lag3', 'GWETROOT_lag7',
    'RH2M_lag1', 'RH2M_lag3', 'RH2M_lag7',
    'WS10M_lag1', 'WS10M_lag3', 'WS10M_lag7',
    'PRECTOTCORR_lag1', 'PRECTOTCORR_lag3', 'PRECTOTCORR_lag7',
    'dry_days_streak', 'hot_days_streak'
]


def _build_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['GWETROOT_tplus7'] = df['GWETROOT'].shift(-FORECAST_HORIZON)
    df['T2M_MAX_tplus7'] = df['T2M_MAX'].shift(-FORECAST_HORIZON)
    df = df.dropna(subset=['GWETROOT_tplus7', 'T2M_MAX_tplus7']).copy()
    return df


@lru_cache(maxsize=1)
def get_trained_artifacts() -> Dict[str, object]:
    raw_df = load_nasa_power_csv()
    df = _prepare_base(raw_df)
    df = _add_features(df)
    df = _build_targets(df)

    features = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[features]
    y_gwet = df['GWETROOT_tplus7']
    y_tmax = df['T2M_MAX_tplus7']

    model_gwet = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('rf', RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_split=8,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ))
    ])

    model_tmax = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('rf', RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_split=8,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ))
    ])

    model_gwet.fit(X, y_gwet)
    model_tmax.fit(X, y_tmax)

    return {
        'df': df,
        'features': features,
        'model_gwet': model_gwet,
        'model_tmax': model_tmax,
    }


def _compute_alert(pred_gwet: float, pred_tmax: float) -> tuple[str, int, List[str]]:
    score = 0
    factors = []

    if pred_gwet < 0.22:
        score += 2
        factors.append('Humidité du sol future très faible')
    elif pred_gwet < 0.25:
        score += 1
        factors.append('Humidité du sol future modérée')

    if pred_tmax > 38:
        score += 2
        factors.append('Température maximale future très élevée')
    elif pred_tmax > 32:
        score += 1
        factors.append('Température maximale future élevée')

    if score >= 4:
        return 'Critique', score, factors
    if score >= 2:
        return 'Moyenne', score, factors
    return 'Stable', score, factors or ['Conditions relativement stables']


def predict_future_risk(selected_date: str) -> PredictionResult:
    artifacts = get_trained_artifacts()
    df: pd.DataFrame = artifacts['df']
    features: List[str] = artifacts['features']

    target_date = pd.to_datetime(selected_date)
    row_df = df[df['DATE'] == target_date].copy()

    if row_df.empty:
        raise ValueError('La date choisie n\'est pas disponible dans le dataset.')

    row = row_df.iloc[[0]][features]
    pred_gwet = float(artifacts['model_gwet'].predict(row)[0])
    pred_tmax = float(artifacts['model_tmax'].predict(row)[0])

    current_row = df[df['DATE'] == target_date].iloc[0]
    alert_level, risk_score, factors = _compute_alert(pred_gwet, pred_tmax)

    return PredictionResult(
        selected_date=str(target_date.date()),
        current_gwetroot=float(current_row['GWETROOT']),
        current_t2m_max=float(current_row['T2M_MAX']),
        predicted_gwetroot_tplus7=pred_gwet,
        predicted_t2m_max_tplus7=pred_tmax,
        alert_level=alert_level,
        risk_score=risk_score,
        dominant_factors=factors,
    )


def get_available_date_bounds() -> Dict[str, str]:
    artifacts = get_trained_artifacts()
    df: pd.DataFrame = artifacts['df']
    return {
        'min_date': str(df['DATE'].min().date()),
        'max_date': str(df['DATE'].max().date()),
    }
