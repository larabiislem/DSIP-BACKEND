# DSIP Backend

FastAPI backend for ingesting raw transaction data, building hourly windows, and writing statistical downtime forecasts to PostgreSQL.

## What it does

- Loads raw transaction rows into Postgres.
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

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## API

- `GET /health`
- `POST /transactions/bulk`
- `POST /transactions/upload-csv`
- `POST /pipeline/run`
- `GET /predictions/{predict_day}`
- `GET /windows/latest`

## Example pipeline run

```json
{
  "predict_day": "2023-11-08",
  "train_start": "2023-11-02",
  "train_end": "2023-11-07",
  "granularity": "H",
  "model_version": "statistical_v1"
}
```

## Notes

- The database is initialized automatically on startup.
- If you only have the raw transaction table, upload those rows first, then run the pipeline.
- The forecast is statistical, not a trained ML model, so it is safe to use with a small history window.
