from pathlib import Path
import pandas as pd
from django.conf import settings

DATA_FILE = settings.BASE_DIR / 'data' / 'POWER_Point_Daily_20200101_20241231_033d90N_010d10E_LST.csv'


def load_nasa_power_csv() -> pd.DataFrame:
    if not Path(DATA_FILE).exists():
        raise FileNotFoundError(f"NASA POWER CSV not found at {DATA_FILE}")

    header_row = None
    with open(DATA_FILE, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if 'YEAR' in line and 'DOY' in line and 'T2M' in line:
                header_row = i
                break

    if header_row is None:
        raise ValueError('Could not locate NASA POWER CSV header row.')

    df = pd.read_csv(DATA_FILE, skiprows=header_row)
    return df