from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.models import HourlyPrediction, HourlyWindow, PredictionRun, RawTransaction


RAW_FEATURE_COLUMNS = [
    "downtime_seconds",
    "downtime_minutes",
    "n_down_events",
    "down_rate",
    "fail_rate",
    "mean_gap",
    "max_gap",
    "p99_gap",
    "n_transactions",
    "availability_pct",
    "rc_neg_rate",
    "rc_915_rate",
    "standin_rate",
    "mean_ceiling",
]


def transactions_to_frame(rows) -> pd.DataFrame:
    data = [
        {
            "universal_transaction_number": row.universal_transaction_number,
            "reversal_flag": row.reversal_flag,
            "svfe_trace_number": row.svfe_trace_number,
            "svfe_message_type": row.svfe_message_type,
            "response_code": row.response_code,
            "svfe_transaction_type": row.svfe_transaction_type,
            "svfe_response_code": row.svfe_response_code,
            "completion_status": row.completion_status,
            "stood_in_for": row.stood_in_for,
            "issuer_posted": row.issuer_posted,
            "ts": row.ts,
            "gap_to_next_sec": row.gap_to_next_sec,
            "gap_label": row.gap_label,
            "local_expected_gap_sec": row.local_expected_gap_sec,
            "local_ceiling_sec": row.local_ceiling_sec,
            "duration_to_expected_ratio": row.duration_to_expected_ratio,
            "hour_of_day": row.hour_of_day,
            "day_of_week": row.day_of_week,
            "is_active_hours": row.is_active_hours,
            "minutes_from_open": row.minutes_from_open,
            "minutes_from_close": row.minutes_from_close,
            "confidence_tier": row.confidence_tier,
        }
        for row in rows
    ]
    if not data:
        return pd.DataFrame()
    frame = pd.DataFrame(data)
    frame["ts"] = pd.to_datetime(frame["ts"], utc=False)
    return frame


def infer_downtime_seconds(frame: pd.DataFrame) -> pd.Series:
    expected = frame["local_expected_gap_sec"].fillna(frame["gap_to_next_sec"])
    gap = frame["gap_to_next_sec"].fillna(0)
    ceiling = frame["local_ceiling_sec"].fillna(np.inf)
    label_down = frame["gap_label"].astype(str).str.contains("down", case=False, na=False)
    forced_down = (
        label_down
        | (frame["completion_status"].fillna(1).astype(int) == 0)
        | (gap > ceiling)
    )
    downtime = (gap - expected).clip(lower=0)
    return downtime.where(forced_down, 0).fillna(0.0)


def build_hourly_windows(transactions: pd.DataFrame) -> pd.DataFrame:
    if transactions.empty:
        return pd.DataFrame()

    frame = transactions.copy()
    frame["window_start"] = frame["ts"].dt.floor("H")
    frame["downtime_seconds_row"] = infer_downtime_seconds(frame)
    frame["is_down_event"] = frame["downtime_seconds_row"] > 0
    frame["is_fail"] = frame["completion_status"].fillna(1).astype(int) == 0
    frame["is_rc_neg"] = frame["svfe_response_code"].astype(str) == "-01"
    frame["is_rc_915"] = frame["svfe_response_code"].astype(str) == "915"
    frame["is_standin"] = frame["stood_in_for"].fillna(0).astype(int) == 1

    grouped = frame.groupby("window_start", sort=True)
    windows = grouped.agg(
        n_transactions=("ts", "size"),
        downtime_seconds=("downtime_seconds_row", "sum"),
        n_down_events=("is_down_event", "sum"),
        fail_rate=("is_fail", "mean"),
        mean_gap=("gap_to_next_sec", "mean"),
        max_gap=("gap_to_next_sec", "max"),
        p99_gap=("gap_to_next_sec", lambda x: float(np.nanpercentile(x, 99)) if len(x.dropna()) else 0.0),
        rc_neg_rate=("is_rc_neg", "mean"),
        rc_915_rate=("is_rc_915", "mean"),
        standin_rate=("is_standin", "mean"),
        mean_ceiling=("local_ceiling_sec", "mean"),
    ).reset_index()

    windows["downtime_minutes"] = windows["downtime_seconds"] / 60.0
    windows["down_rate"] = windows["n_down_events"] / windows["n_transactions"].replace(0, np.nan)
    windows["availability_pct"] = (1 - windows["downtime_seconds"] / 3600.0).clip(lower=0) * 100.0
    windows["hour_of_day"] = windows["window_start"].dt.hour
    windows["day_of_week"] = windows["window_start"].dt.dayofweek
    windows["down_rate"] = windows["down_rate"].fillna(0.0)
    windows["availability_pct"] = windows["availability_pct"].fillna(100.0)
    windows["fail_rate"] = windows["fail_rate"].fillna(0.0)
    windows["rc_neg_rate"] = windows["rc_neg_rate"].fillna(0.0)
    windows["rc_915_rate"] = windows["rc_915_rate"].fillna(0.0)
    windows["standin_rate"] = windows["standin_rate"].fillna(0.0)
    windows["mean_gap"] = windows["mean_gap"].fillna(0.0)
    windows["max_gap"] = windows["max_gap"].fillna(0.0)
    windows["p99_gap"] = windows["p99_gap"].fillna(0.0)
    windows["mean_ceiling"] = windows["mean_ceiling"].fillna(0.0)

    return windows


def build_forecast_feature_frame(history_windows: pd.DataFrame, predict_day: date, granularity: str = "H") -> tuple[pd.DataFrame, int]:
    if history_windows.empty:
        raise ValueError("Training history is empty. Load raw transactions first.")

    gran_minutes = int(pd.Timedelta(granularity).total_seconds() / 60)
    windows_per_day = int(24 * 60 / gran_minutes)

    predict_starts = pd.date_range(pd.Timestamp(predict_day), periods=windows_per_day, freq=granularity)
    pred_frame = pd.DataFrame({"window_start": predict_starts})
    pred_frame["hour_of_day"] = pred_frame["window_start"].dt.hour
    pred_frame["day_of_week"] = pred_frame["window_start"].dt.dayofweek

    combined = pd.concat([history_windows.copy(), pred_frame], ignore_index=True, sort=False)
    combined = combined.sort_values("window_start").reset_index(drop=True)

    lag_1h = max(1, 60 // gran_minutes)
    lag_24h = windows_per_day
    lag_48h = windows_per_day * 2

    for col in RAW_FEATURE_COLUMNS:
        if col not in combined.columns:
            combined[col] = np.nan
        combined[f"{col}_lag_1h"] = combined[col].shift(lag_1h)
        combined[f"{col}_lag_24h"] = combined[col].shift(lag_24h)
        combined[f"{col}_lag_48h"] = combined[col].shift(lag_48h)

    for col in ["downtime_seconds", "down_rate", "fail_rate", "mean_gap", "max_gap", "availability_pct"]:
        combined[f"{col}_samedow_mean"] = (
            combined.groupby("hour_of_day")[col]
            .transform(lambda x: x.shift(1).expanding().mean())
        )
        combined[f"{col}_samedow_std"] = (
            combined.groupby("hour_of_day")[col]
            .transform(lambda x: x.shift(1).expanding().std().fillna(0))
        )

    combined["downtime_trend_1h"] = combined["downtime_seconds"].shift(lag_1h).diff(lag_1h)
    combined["downtime_trend_24h"] = combined["downtime_seconds"].shift(lag_24h).diff(lag_24h)
    combined["delta_vs_yesterday"] = combined["downtime_seconds_lag_1h"] - combined["downtime_seconds_lag_24h"]
    combined["hour_sin"] = np.sin(2 * np.pi * combined["hour_of_day"] / 24)
    combined["hour_cos"] = np.cos(2 * np.pi * combined["hour_of_day"] / 24)
    combined["dow_sin"] = np.sin(2 * np.pi * combined["day_of_week"] / 7)
    combined["dow_cos"] = np.cos(2 * np.pi * combined["day_of_week"] / 7)
    combined["is_weekend"] = (combined["day_of_week"] >= 5).astype(int)
    combined["is_night"] = ((combined["hour_of_day"] < 6) | (combined["hour_of_day"] >= 22)).astype(int)
    combined["is_peak"] = combined["hour_of_day"].between(8, 12) | combined["hour_of_day"].between(14, 18)
    combined["is_peak"] = combined["is_peak"].astype(int)

    pred_features = combined.iloc[len(history_windows) :].copy()
    pred_features = pred_features.fillna(0.0)
    pred_features["predict_day"] = pd.Timestamp(predict_day).date()
    return pred_features, windows_per_day


def forecast_hourly_windows(history_windows: pd.DataFrame, predict_day: date, model_version: str, granularity: str = "H") -> pd.DataFrame:
    pred_features, _ = build_forecast_feature_frame(history_windows, predict_day, granularity=granularity)

    same_hour_profile = (
        history_windows.groupby("hour_of_day")["downtime_seconds"]
        .agg(["mean", "std", "max"])
        .rename(columns={"mean": "hour_mean", "std": "hour_std", "max": "hour_max"})
        .reset_index()
    )
    dow_profile = history_windows.groupby("day_of_week")["downtime_seconds"].mean().rename("dow_mean").reset_index()
    pred_features = pred_features.merge(same_hour_profile, on="hour_of_day", how="left")
    pred_features = pred_features.merge(dow_profile, on="day_of_week", how="left")

    pred_features["hour_mean"] = pred_features["hour_mean"].fillna(0.0)
    pred_features["hour_std"] = pred_features["hour_std"].fillna(0.0)
    pred_features["hour_max"] = pred_features["hour_max"].fillna(0.0)
    pred_features["dow_mean"] = pred_features["dow_mean"].fillna(0.0)

    lag_score = pred_features["downtime_seconds_lag_24h"].fillna(0.0)
    trend_score = pred_features["downtime_trend_24h"].fillna(0.0).clip(lower=0)
    hour_score = pred_features["hour_mean"]
    dow_score = pred_features["dow_mean"]

    pred_downtime_sec = (0.45 * lag_score + 0.35 * hour_score + 0.20 * dow_score + 0.10 * trend_score).clip(lower=0)
    pred_high_sec = (pred_downtime_sec + pred_features["hour_std"].fillna(0.0) * 1.5).clip(lower=pred_downtime_sec)
    proba_down = (
        0.5 * pred_features["down_rate_lag_24h"].fillna(0.0)
        + 0.3 * pred_features["down_rate_samedow_mean"].fillna(0.0)
        + 0.2 * pred_features["fail_rate_lag_24h"].fillna(0.0)
    ).clip(0, 1)

    pred_availability_pct = (1 - pred_downtime_sec / 3600.0).clip(lower=0) * 100.0

    output = pd.DataFrame(
        {
            "prediction_run_ts": datetime.utcnow(),
            "predict_day": pd.Timestamp(predict_day).date(),
            "hour_of_day": pred_features["hour_of_day"].astype(int),
            "window_start": pred_features["window_start"],
            "pred_downtime_sec": pred_downtime_sec,
            "pred_downtime_min": pred_downtime_sec / 60.0,
            "pred_high_sec": pred_high_sec,
            "proba_down": proba_down,
            "pred_availability_pct": pred_availability_pct,
            "true_downtime_sec": np.nan,
            "true_availability_pct": np.nan,
            "error_abs_sec": np.nan,
            "alert_level": np.where(
                pred_downtime_sec >= 180,
                "DOWN",
                np.where(pred_downtime_sec >= 60, "WARN", "OK"),
            ),
            "model_version": model_version,
            "train_start": history_windows["window_start"].min().date(),
            "train_end": history_windows["window_start"].max().date(),
        }
    )
    return output


def load_raw_transactions(session: Session, records: list[dict]) -> int:
    if not records:
        return 0

    session.execute(delete(RawTransaction))
    session.commit()

    rows = [RawTransaction(**record) for record in records]
    session.add_all(rows)
    session.commit()
    return len(rows)


def fetch_raw_transactions(session: Session, start_date: date | None = None, end_date: date | None = None) -> pd.DataFrame:
    query = select(RawTransaction)
    if start_date is not None:
        query = query.where(RawTransaction.ts >= datetime.combine(start_date, datetime.min.time()))
    if end_date is not None:
        query = query.where(RawTransaction.ts < datetime.combine(end_date + timedelta(days=1), datetime.min.time()))
    rows = session.scalars(query.order_by(RawTransaction.ts.asc())).all()
    return transactions_to_frame(rows)


def save_hourly_windows(session: Session, windows: pd.DataFrame) -> int:
    session.execute(delete(HourlyWindow))
    session.commit()
    if windows.empty:
        return 0

    now = datetime.utcnow()
    entities = [
        HourlyWindow(
            window_start=row.window_start.to_pydatetime() if hasattr(row.window_start, "to_pydatetime") else row.window_start,
            hour_of_day=int(row.hour_of_day),
            day_of_week=int(row.day_of_week),
            n_transactions=int(row.n_transactions),
            downtime_seconds=float(row.downtime_seconds),
            downtime_minutes=float(row.downtime_minutes),
            n_down_events=int(row.n_down_events),
            down_rate=float(row.down_rate),
            fail_rate=float(row.fail_rate),
            mean_gap=float(row.mean_gap),
            max_gap=float(row.max_gap),
            p99_gap=float(row.p99_gap),
            availability_pct=float(row.availability_pct),
            rc_neg_rate=float(row.rc_neg_rate),
            rc_915_rate=float(row.rc_915_rate),
            standin_rate=float(row.standin_rate),
            mean_ceiling=float(row.mean_ceiling),
            created_at=now,
        )
        for row in windows.itertuples(index=False)
    ]
    session.add_all(entities)
    session.commit()
    return len(entities)


def run_prediction_pipeline(
    session: Session,
    predict_day: date,
    train_start: date | None = None,
    train_end: date | None = None,
    granularity: str = "H",
    model_version: str = "statistical_v1",
) -> tuple[PredictionRun, list[HourlyPrediction]]:
    raw_history = fetch_raw_transactions(session, start_date=train_start, end_date=train_end)
    if raw_history.empty:
        raise ValueError("No training data found. Load raw transactions before running the pipeline.")

    hourly_windows = build_hourly_windows(raw_history)
    if hourly_windows.empty:
        raise ValueError("Unable to build hourly windows from the loaded transactions.")

    save_hourly_windows(session, hourly_windows)

    if train_start is None:
        train_start = hourly_windows["window_start"].min().date()
    if train_end is None:
        train_end = hourly_windows["window_start"].max().date()

    run = PredictionRun(
        prediction_run_ts=datetime.utcnow(),
        predict_day=predict_day,
        train_start=train_start,
        train_end=train_end,
        model_version=model_version,
        granularity=granularity,
    )
    session.add(run)
    session.commit()
    session.refresh(run)

    predictions_df = forecast_hourly_windows(hourly_windows, predict_day, model_version, granularity=granularity)

    predicted_rows = [
        HourlyPrediction(
            prediction_run_ts=row.prediction_run_ts.to_pydatetime() if hasattr(row.prediction_run_ts, "to_pydatetime") else row.prediction_run_ts,
            predict_day=row.predict_day,
            hour_of_day=int(row.hour_of_day),
            window_start=row.window_start.to_pydatetime() if hasattr(row.window_start, "to_pydatetime") else row.window_start,
            pred_downtime_sec=float(row.pred_downtime_sec),
            pred_downtime_min=float(row.pred_downtime_min),
            pred_high_sec=float(row.pred_high_sec),
            proba_down=float(row.proba_down),
            pred_availability_pct=float(row.pred_availability_pct),
            true_downtime_sec=None,
            true_availability_pct=None,
            error_abs_sec=None,
            alert_level=str(row.alert_level),
            model_version=str(row.model_version),
            train_start=row.train_start,
            train_end=row.train_end,
        )
        for row in predictions_df.itertuples(index=False)
    ]
    session.add_all(predicted_rows)
    session.commit()
    return run, predicted_rows
