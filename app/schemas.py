from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field


class TransactionIn(BaseModel):
    universal_transaction_number: str | None = None
    reversal_flag: int | None = None
    svfe_trace_number: str | None = None
    svfe_message_type: str | None = None
    response_code: str | None = None
    svfe_transaction_type: str | None = None
    svfe_response_code: str | None = None
    completion_status: int | None = None
    stood_in_for: int | None = None
    issuer_posted: int | None = None
    ts: datetime
    gap_to_next_sec: float | None = None
    gap_label: str | None = None
    local_expected_gap_sec: float | None = None
    local_ceiling_sec: float | None = None
    duration_to_expected_ratio: float | None = None
    hour_of_day: float | None = None
    day_of_week: float | None = None
    is_active_hours: bool | None = None
    minutes_from_open: float | None = None
    minutes_from_close: float | None = None
    confidence_tier: str | None = None


class BulkTransactionsIn(BaseModel):
    records: list[TransactionIn] = Field(default_factory=list)


class PipelineRunRequest(BaseModel):
    predict_day: date
    train_start: date | None = None
    train_end: date | None = None
    granularity: str = "H"
    model_version: str = "statistical_v1"


class PipelineRunResponse(BaseModel):
    prediction_run_ts: datetime
    predict_day: date
    train_start: date
    train_end: date
    model_version: str
    granularity: str
    rows_written: int


class HealthResponse(BaseModel):
    status: str
    database: str
