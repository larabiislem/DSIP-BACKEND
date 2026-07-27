from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from sqlalchemy import Boolean, Date, DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class RawTransaction(Base):
    __tablename__ = "raw_transactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    universal_transaction_number: Mapped[Optional[str]] = mapped_column(String(64), index=True, nullable=True)
    reversal_flag: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    svfe_trace_number: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    svfe_message_type: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    response_code: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    svfe_transaction_type: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    svfe_response_code: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    completion_status: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    stood_in_for: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    issuer_posted: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=False), index=True, nullable=False)
    gap_to_next_sec: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    gap_label: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    local_expected_gap_sec: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    local_ceiling_sec: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    duration_to_expected_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    hour_of_day: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    day_of_week: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    is_active_hours: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    minutes_from_open: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    minutes_from_close: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    confidence_tier: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)


class HourlyWindow(Base):
    __tablename__ = "hourly_windows"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=False), index=True, nullable=False)
    hour_of_day: Mapped[int] = mapped_column(Integer, nullable=False)
    day_of_week: Mapped[int] = mapped_column(Integer, nullable=False)
    n_transactions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    downtime_seconds: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    downtime_minutes: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    n_down_events: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    down_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    fail_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    mean_gap: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    max_gap: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    p99_gap: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    availability_pct: Mapped[float] = mapped_column(Float, nullable=False, default=100.0)
    rc_neg_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    rc_915_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    standin_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    mean_ceiling: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)


class PredictionRun(Base):
    __tablename__ = "prediction_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_run_ts: Mapped[datetime] = mapped_column(DateTime(timezone=False), index=True, nullable=False)
    predict_day: Mapped[date] = mapped_column(Date, index=True, nullable=False)
    train_start: Mapped[date] = mapped_column(Date, nullable=False)
    train_end: Mapped[date] = mapped_column(Date, nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    granularity: Mapped[str] = mapped_column(String(8), nullable=False)


class HourlyPrediction(Base):
    __tablename__ = "hourly_predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_run_ts: Mapped[datetime] = mapped_column(DateTime(timezone=False), index=True, nullable=False)
    predict_day: Mapped[date] = mapped_column(Date, index=True, nullable=False)
    hour_of_day: Mapped[int] = mapped_column(Integer, nullable=False)
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=False), index=True, nullable=False)
    pred_downtime_sec: Mapped[float] = mapped_column(Float, nullable=False)
    pred_downtime_min: Mapped[float] = mapped_column(Float, nullable=False)
    pred_high_sec: Mapped[float] = mapped_column(Float, nullable=False)
    proba_down: Mapped[float] = mapped_column(Float, nullable=False)
    pred_availability_pct: Mapped[float] = mapped_column(Float, nullable=False)
    true_downtime_sec: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    true_availability_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    error_abs_sec: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    alert_level: Mapped[str] = mapped_column(String(16), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    train_start: Mapped[date] = mapped_column(Date, nullable=False)
    train_end: Mapped[date] = mapped_column(Date, nullable=False)
