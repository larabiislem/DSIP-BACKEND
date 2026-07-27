from __future__ import annotations

from datetime import date

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pandas import read_csv
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import Base, engine, get_db
from app.models import HourlyPrediction, HourlyWindow
from app.pipeline import load_raw_transactions, run_prediction_pipeline
from app.schemas import BulkTransactionsIn, HealthResponse, PipelineRunRequest, PipelineRunResponse


settings = get_settings()
app = FastAPI(title=settings.app_name, version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    Base.metadata.create_all(bind=engine)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", database="connected")


@app.post("/transactions/bulk")
def ingest_transactions(payload: BulkTransactionsIn, db: Session = Depends(get_db)) -> dict[str, int]:
    count = load_raw_transactions(db, [record.model_dump() for record in payload.records])
    return {"inserted": count}


@app.post("/transactions/upload-csv")
async def ingest_transactions_csv(file: UploadFile = File(...), db: Session = Depends(get_db)) -> dict[str, int]:
    if not file.filename.lower().endswith((".csv", ".txt")):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")
    frame = read_csv(file.file)
    records = frame.to_dict(orient="records")
    count = load_raw_transactions(db, records)
    return {"inserted": count}


@app.post("/pipeline/run", response_model=PipelineRunResponse)
def run_pipeline(payload: PipelineRunRequest, db: Session = Depends(get_db)) -> PipelineRunResponse:
    try:
        run, predictions = run_prediction_pipeline(
            db,
            predict_day=payload.predict_day,
            train_start=payload.train_start,
            train_end=payload.train_end,
            granularity=payload.granularity,
            model_version=payload.model_version,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return PipelineRunResponse(
        prediction_run_ts=run.prediction_run_ts,
        predict_day=run.predict_day,
        train_start=run.train_start,
        train_end=run.train_end,
        model_version=run.model_version,
        granularity=run.granularity,
        rows_written=len(predictions),
    )


@app.get("/predictions/{predict_day}")
def get_predictions(predict_day: date, db: Session = Depends(get_db)) -> list[dict]:
    rows = db.query(HourlyPrediction).filter(HourlyPrediction.predict_day == predict_day).order_by(HourlyPrediction.hour_of_day.asc()).all()
    return [
        {
            "id": row.id,
            "prediction_run_ts": row.prediction_run_ts,
            "predict_day": row.predict_day,
            "hour_of_day": row.hour_of_day,
            "window_start": row.window_start,
            "pred_downtime_sec": row.pred_downtime_sec,
            "pred_downtime_min": row.pred_downtime_min,
            "pred_high_sec": row.pred_high_sec,
            "proba_down": row.proba_down,
            "pred_availability_pct": row.pred_availability_pct,
            "true_downtime_sec": row.true_downtime_sec,
            "true_availability_pct": row.true_availability_pct,
            "error_abs_sec": row.error_abs_sec,
            "alert_level": row.alert_level,
            "model_version": row.model_version,
            "train_start": row.train_start,
            "train_end": row.train_end,
        }
        for row in rows
    ]


@app.get("/windows/latest")
def get_latest_windows(db: Session = Depends(get_db)) -> list[dict]:
    rows = db.query(HourlyWindow).order_by(HourlyWindow.window_start.asc()).all()
    return [
        {
            "id": row.id,
            "window_start": row.window_start,
            "hour_of_day": row.hour_of_day,
            "day_of_week": row.day_of_week,
            "n_transactions": row.n_transactions,
            "downtime_seconds": row.downtime_seconds,
            "downtime_minutes": row.downtime_minutes,
            "n_down_events": row.n_down_events,
            "down_rate": row.down_rate,
            "fail_rate": row.fail_rate,
            "mean_gap": row.mean_gap,
            "max_gap": row.max_gap,
            "p99_gap": row.p99_gap,
            "availability_pct": row.availability_pct,
            "rc_neg_rate": row.rc_neg_rate,
            "rc_915_rate": row.rc_915_rate,
            "standin_rate": row.standin_rate,
            "mean_ceiling": row.mean_ceiling,
        }
        for row in rows
    ]
