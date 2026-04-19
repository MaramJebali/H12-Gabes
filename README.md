# OASIS AI — Django Project

A Django web application that:
- loads NASA POWER climate data for Gabès,
- engineers time-series features,
- predicts **soil moisture (GWETROOT)** and **maximum temperature (T2M_MAX)** at **J+7**,
- generates an alert level,
- optionally uses **Groq LLM** to explain the risk and recommend actions.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
copy .env.example .env
python manage.py migrate
python manage.py runserver
```

Open: `http://127.0.0.1:8000/`

## Data
The NASA POWER CSV is expected at:

`data/POWER_Point_Daily_20200101_20241231_033d90N_010d10E_LST.csv`

A copy is already included in this package.

## Production notes
- Set `DEBUG=False`
- Configure `ALLOWED_HOSTS`
- Store `GROQ_API_KEY` as an environment variable
- Run `python manage.py collectstatic`
- Deploy with Gunicorn/Whitenoise
