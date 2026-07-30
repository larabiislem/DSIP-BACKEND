# DSIP Backend

FastAPI backend for ingesting raw transaction data, building hourly windows, and writing statistical downtime forecasts to PostgreSQL.

## What it does

- Parses raw SVFE fixed-width posting files (`.dat`) directly, running gap
  detection (local rolling-neighborhood ceiling, Bonferroni-corrected z,
  3-sec resolution floor) and confidence scoring before insertion.
- Loads already-labeled raw transaction rows into Postgres (CSV or bulk JSON).
- Aggregates the raw feed into hourly windows.
- Builds lag and calendar features from historical windows.
- Produces hourly downtime forecasts for a target day.
- Stores both windows and predictions in the database.

## Environment

Create a `.env` file from `.env.example` and set your Neon connection string:

```bash
DATABASE_URL=postgresql+psycopg://...
APP_NAME=dsip-backend
MODEL_VERSION=statistical_v1
TRAIN_DAYS=7
GRANULARITY=H
```

For local testing without Neon, `DATABASE_URL=sqlite:///./local_dev.db` also
works (SQLAlchemy handles both dialects transparently).

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## API

- `GET /health`
- `POST /transactions/bulk` — insert already gap-labeled rows (JSON)
- `POST /transactions/upload-csv` — insert already gap-labeled rows (CSV)
- `POST /transactions/upload-raw-dat` — **new**: upload a raw fixed-width
  `.dat` posting file; runs parsing + gap detection + confidence scoring
  (`app/svfe_ingest.py`) and inserts the resulting rows automatically.
  Params: `year` (default 2023), `target_date` (optional MMDD filter, e.g. `1031`).
- `POST /pipeline/run`
- `GET /predictions/{predict_day}`
- `GET /windows/latest`

## Example: raw file straight through the whole pipeline

```bash
curl -X POST "http://localhost:8000/transactions/upload-raw-dat?year=2023&target_date=1031" \
     -F "file=@path/to/posting_file.dat"

curl -X POST "http://localhost:8000/pipeline/run" \
     -H "Content-Type: application/json" \
     -d '{"predict_day": "2023-11-08", "train_start": "2023-11-02", "train_end": "2023-11-07"}'
```

## Notes

- The database is initialized automatically on startup.
- If you only have the raw transaction table, upload those rows first, then run the pipeline.
- The forecast is statistical, not a trained ML model, so it is safe to use with a small history window.
